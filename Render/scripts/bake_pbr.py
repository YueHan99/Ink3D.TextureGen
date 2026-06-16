#!/usr/bin/env python3
"""
PBR Texture Baking: Reconstruct textured GLB from rendered multi-view videos.

Uses xatlas UV unwrapping + nvdiffrast to bake albedo, metallic, and roughness
textures from H+V rendered condition videos back onto a GLB mesh.

Dependencies: torch, utils3d, nvdiffrast, trimesh, xatlas, pyvista, pymeshfix,
              igraph, cv2, imageio, numpy, scipy

Usage:
    python bake_pbr.py \
        --sha256 "000-058/41c7f9707c9744b1b8cc849c7dee4982_1024" \
        --glb_path ./glbs_normalized/41c7f9707c9744b1b8cc849c7dee4982_1024.glb \
        --albedo_h_video ./albedo_h.mp4 \
        --albedo_v_video ./albedo_v.mp4 \
        --mr_h_video ./mr.mp4 \
        --mr_v_video ./mr_v.mp4 \
        --h_meta ./meta_h.json \
        --v_meta ./meta_v.json \
        --output_dir ./output
"""
import argparse
import json
import os
from glob import glob
from typing import List, Tuple, Literal

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import trimesh.visual
import utils3d
import xatlas
from PIL import Image
from tqdm import tqdm


def load_observations_from_video(video_path: str, flip: bool = False) -> List[np.ndarray]:
    """Load all frames from a video file, returning RGB uint8 arrays."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    print(f"Loading video: {video_path}")
    reader = imageio.get_reader(video_path)
    observations = []
    for frame in reader:
        if frame.ndim == 3 and frame.shape[-1] == 4:
            rgb_frame = frame[:, :, :3]
        elif frame.ndim == 3 and frame.shape[-1] == 3:
            rgb_frame = frame
        else:
            raise ValueError(f"Unexpected frame shape: {frame.shape}")
        if flip:
            rgb_frame = np.fliplr(rgb_frame.astype(np.uint8))
            observations.append(rgb_frame)
        else:
            observations.append(rgb_frame.astype(np.uint8))
    reader.close()
    print(f"Loaded {len(observations)} frames from video.")
    return observations


def resize_observations_to_mask(observations: List[np.ndarray],
                                target_size: int = 1024) -> List[np.ndarray]:
    """Resize observations to target_size x target_size."""
    return [cv2.resize(obs, (target_size, target_size)) for obs in observations]


def split_mr(obs_list: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Split combined MR frames into metallic (channel B) and roughness (channel G) lists."""
    m_list, r_list = [], []
    for img in obs_list:
        r_ch = img[:, :, 1]  # roughness
        m_ch = img[:, :, 2]  # metallic
        r_list.append(np.stack([r_ch, r_ch, r_ch], axis=-1))
        m_list.append(np.stack([m_ch, m_ch, m_ch], axis=-1))
    return m_list, r_list


def generate_camera_positions_continuous(yaws, pitch, r):
    """Generate camera positions from yaw, pitch, radius on a sphere."""
    is_list = isinstance(yaws, list)
    if not is_list:
        yaws = [yaws]
    positions = []
    pitch_rad = torch.deg2rad(torch.tensor(float(pitch), dtype=torch.float32, device='cuda'))
    start = torch.tensor([
        0.0,
        torch.cos(pitch_rad).item(),
        torch.sin(pitch_rad).item()
    ], dtype=torch.float32, device='cuda') * r
    for yaw in yaws:
        yaw_rad = torch.tensor(float(yaw), dtype=torch.float32, device='cuda')
        cos_y = torch.cos(yaw_rad)
        sin_y = torch.sin(yaw_rad)
        R_z = torch.tensor([
            [cos_y, -sin_y, 0.0],
            [sin_y,  cos_y, 0.0],
            [0.0,     0.0, 1.0]
        ], dtype=torch.float32, device='cuda')
        pos = R_z @ start
        positions.append(pos)
    if not is_list:
        positions = positions[0]
    return positions


def _safe_look_at(pos, target, pitch_deg):
    """Look-at with robust up-vector for full-sphere camera orbits."""
    forward = F.normalize(target - pos, dim=-1)
    if 90 < pitch_deg < 270:
        up = torch.tensor([0., 0., -1.], dtype=torch.float32, device='cuda')
    else:
        up = torch.tensor([0., 0., 1.], dtype=torch.float32, device='cuda')
    return utils3d.torch.extrinsics_look_at(pos, target, up)


def yaw_pitch_r_fov_to_extrinsics_intrinsics_robust(yaws, pitchs, rs, fovs):
    """Convert spherical camera parameters to extrinsics and intrinsics matrices."""
    is_list = isinstance(yaws, list)
    if not is_list:
        yaws = [yaws]
        pitchs = [pitchs]
    if not isinstance(rs, list):
        rs = [rs] * len(yaws)
    if not isinstance(fovs, list):
        fovs = [fovs] * len(yaws)
    extrinsics = []
    intrinsics = []
    for yaw, pitch, r, fov in zip(yaws, pitchs, rs, fovs):
        yaw_shifted = -yaw
        pos = generate_camera_positions_continuous(yaw_shifted, pitch, r)
        target = torch.zeros(3, dtype=torch.float32, device='cuda')
        extr = _safe_look_at(pos, target,
                             pitch_deg=pitch.item() if torch.is_tensor(pitch) else float(pitch))
        fov_rad = torch.deg2rad(torch.tensor(float(fov), dtype=torch.float32, device='cuda'))
        intr = utils3d.torch.intrinsics_from_fov_xy(fov_rad, fov_rad)
        extrinsics.append(extr)
        intrinsics.append(intr)
    if not is_list:
        extrinsics = extrinsics[0]
        intrinsics = intrinsics[0]
    return extrinsics, intrinsics


def get_intrinsics_pixel(width: int, height: int, fov_x_deg: float) -> np.ndarray:
    """Generate pixel-space intrinsic matrix from FOV."""
    fov_x_rad = np.deg2rad(fov_x_deg)
    fx = fy = (width / 2) / np.tan(fov_x_rad / 2)
    cx, cy = width / 2, height / 2
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def get_intrinsics_normalized(width: int, height: int, fov_x_deg: float) -> np.ndarray:
    """Generate normalized intrinsic matrix (coordinates in [0, 1])."""
    K_pixel = get_intrinsics_pixel(width, height, fov_x_deg)
    K_norm = K_pixel.copy()
    K_norm[0, :] /= width
    K_norm[1, :] /= height
    return K_norm


def load_cameras_from_meta(meta_path: str) -> Tuple[List[float], List[float],
                                                      List[float], List[float]]:
    """Load yaw, pitch, radius, fov lists from meta.json."""
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    yaws, pitchs, rs, fovs = [], [], [], []
    for loc in meta['locations']:
        yaw_deg = np.degrees(loc['azimuth'])
        pitch_deg = np.degrees(loc['elevation'])
        T = np.array(loc['transform_matrix'])
        pos = T[:3, 3]
        r = np.linalg.norm(pos)
        fov_x_deg = np.degrees(loc['camera_angle_x'])
        yaws.append(yaw_deg)
        pitchs.append(pitch_deg)
        rs.append(r)
        fovs.append(fov_x_deg)
    return yaws, pitchs, rs, fovs


def bake_texture(
    vertices: np.array,
    faces: np.array,
    uvs: np.array,
    observations: List[np.array],
    masks: List[np.array],
    extrinsics: List[np.array],
    intrinsics: List[np.array],
    texture_size: int = 1024,
    near: float = 0.1,
    far: float = 10.0,
    mode: Literal['fast', 'opt'] = 'fast',
    lambda_tv: float = 1e-2,
    verbose: bool = False,
) -> np.ndarray:
    """Bake texture to a mesh from multiple observations using nvdiffrast."""
    vertices = torch.tensor(vertices).cuda()
    faces = torch.tensor(faces.astype(np.int32)).cuda()
    uvs = torch.tensor(uvs).cuda()
    observations = [torch.tensor(obs / 255.0).float().cuda() for obs in observations]
    masks = [torch.tensor(m > 0).bool().cuda() for m in masks]
    views = [utils3d.torch.extrinsics_to_view(torch.tensor(extr).cuda()) for extr in extrinsics]
    projections = [utils3d.torch.intrinsics_to_perspective(
        torch.tensor(intr).cuda(), near, far) for intr in intrinsics]

    if mode == 'fast':
        texture = torch.zeros((texture_size * texture_size, 3), dtype=torch.float32).cuda()
        texture_weights = torch.zeros((texture_size * texture_size), dtype=torch.float32).cuda()
        rastctx = utils3d.torch.RastContext(backend='cuda')
        for observation, view, projection in tqdm(
            zip(observations, views, projections), total=len(observations),
            disable=not verbose, desc='Texture baking (fast)'
        ):
            with torch.no_grad():
                rast = utils3d.torch.rasterize_triangle_faces(
                    rastctx, vertices[None], faces, observation.shape[1],
                    observation.shape[0], uv=uvs[None], view=view,
                    projection=projection
                )
                uv_map = rast['uv'][0].detach().flip(0)
                mask = rast['mask'][0].detach().bool() & masks[0]
            uv_map = (uv_map * texture_size).floor().long()
            obs = observation[mask]
            uv_map = uv_map[mask]
            idx = uv_map[:, 0] + (texture_size - uv_map[:, 1] - 1) * texture_size
            texture = texture.scatter_add(0, idx.view(-1, 1).expand(-1, 3), obs)
            texture_weights = texture_weights.scatter_add(
                0, idx, torch.ones((obs.shape[0]), dtype=torch.float32, device=texture.device))
        mask = texture_weights > 0
        texture[mask] /= texture_weights[mask][:, None]
        texture = np.clip(texture.reshape(texture_size, texture_size, 3).cpu().numpy() * 255,
                          0, 255).astype(np.uint8)
    elif mode == 'opt':
        import nvdiffrast.torch as dr
        rastctx = utils3d.torch.RastContext(backend='cuda')
        observations = [obs.flip(0) for obs in observations]
        masks = [m.flip(0) for m in masks]
        _uv, _uv_dr = [], []
        for observation, view, projection in tqdm(
            zip(observations, views, projections), total=len(views),
            disable=not verbose, desc='Texture baking (opt): UV'
        ):
            with torch.no_grad():
                rast = utils3d.torch.rasterize_triangle_faces(
                    rastctx, vertices[None], faces, observation.shape[1],
                    observation.shape[0], uv=uvs[None], view=view,
                    projection=projection
                )
                _uv.append(rast['uv'].detach())
                _uv_dr.append(rast['uv_dr'].detach())
        texture = torch.nn.Parameter(
            torch.zeros((1, texture_size, texture_size, 3), dtype=torch.float32).cuda())
        optimizer = torch.optim.Adam([texture], betas=(0.5, 0.9), lr=1e-2)

        def tv_loss(tex):
            return (torch.nn.functional.l1_loss(tex[:, :-1, :, :], tex[:, 1:, :, :]) +
                    torch.nn.functional.l1_loss(tex[:, :, :-1, :], tex[:, :, 1:, :]))

        def cosine_anealing(optimizer, step, total_steps, start_lr, end_lr):
            return end_lr + 0.5 * (start_lr - end_lr) * (1 + np.cos(np.pi * step / total_steps))

        total_steps = 2500
        with tqdm(total=total_steps, disable=not verbose,
                  desc='Texture baking (opt): optimizing') as pbar:
            for step in range(total_steps):
                optimizer.zero_grad()
                selected = np.random.randint(0, len(views))
                uv, uv_dr, observation, mask = (_uv[selected], _uv_dr[selected],
                                                observations[selected], masks[selected])
                render = dr.texture(texture, uv, uv_dr)[0]
                loss = torch.nn.functional.l1_loss(render[mask], observation[mask])
                if lambda_tv > 0:
                    loss += lambda_tv * tv_loss(texture)
                loss.backward()
                optimizer.step()
                optimizer.param_groups[0]['lr'] = cosine_anealing(
                    optimizer, step, total_steps, 1e-2, 1e-5)
                pbar.set_postfix({'loss': loss.item()})
                pbar.update()
        texture = np.clip(texture[0].flip(0).detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f'Unknown mode: {mode}')
    return texture


def process_camera_set(video_path: str, meta_path: str, debug_single_frame: bool = False):
    """Load observations and camera parameters for one orbit (H or V)."""
    observations = load_observations_from_video(video_path, flip=False)
    if debug_single_frame:
        observations = [observations[0]]
    observations = resize_observations_to_mask(observations)

    yaws, pitchs, rs, fovs = load_cameras_from_meta(meta_path)
    if debug_single_frame:
        yaws, pitchs, rs, fovs = yaws[:1], pitchs[:1], rs[:1], fovs[:1]

    print(f"Camera 0: yaw={yaws[0]:.1f} pitch={pitchs[0]:.1f} r={rs[0]:.2f} fov={fovs[0]:.1f}")

    ang = np.pi
    yaws = [-y / 360 * 2 * ang for y in yaws]
    diff = np.float64(ang) - yaws[0]
    yaws = [y + diff for y in yaws]

    extrinsics, _ = yaw_pitch_r_fov_to_extrinsics_intrinsics_robust(
        yaws=yaws, pitchs=pitchs, rs=rs, fovs=fovs)

    with open(meta_path, 'r') as f:
        meta = json.load(f)
    img_width = meta["width"]
    img_height = meta["height"]
    fov_x_deg_list = [np.degrees(loc["camera_angle_x"]) for loc in meta["locations"]]

    intrinsics_normalized = [
        get_intrinsics_normalized(img_width, img_height, fov_x_deg)
        for fov_x_deg in fov_x_deg_list
    ]

    masks = [np.any(obs > 0, axis=-1) for obs in observations]
    return observations, masks, extrinsics, intrinsics_normalized


def main():
    parser = argparse.ArgumentParser(
        description="PBR Texture Baking: Reconstruct textured GLB from rendered videos")
    parser.add_argument("--sha256", type=str, required=True,
                        help="Model identifier (e.g., '000-058/41c7f9707c9744b1b8cc849c7dee4982_1024')")
    parser.add_argument("--glb_path", type=str, required=True,
                        help="Path to normalized GLB mesh")
    parser.add_argument("--albedo_h_video", type=str, required=True,
                        help="Path to H albedo video (albedo_h.mp4)")
    parser.add_argument("--albedo_v_video", type=str, required=True,
                        help="Path to V albedo video (albedo_v.mp4)")
    parser.add_argument("--mr_h_video", type=str, required=True,
                        help="Path to H MR video (mr.mp4)")
    parser.add_argument("--mr_v_video", type=str, required=True,
                        help="Path to V MR video (mr_v.mp4)")
    parser.add_argument("--h_meta", type=str, required=True,
                        help="Path to H meta.json")
    parser.add_argument("--v_meta", type=str, required=True,
                        help="Path to V meta.json")
    parser.add_argument("--output_dir", type=str, default="output_bake",
                        help="Output directory")
    parser.add_argument("--texture_resolution", type=int, default=1024,
                        help="Texture atlas resolution (default: 1024)")
    parser.add_argument("--bake_mode", type=str, choices=["fast", "opt"], default="fast",
                        help="Baking mode: fast (~3s for 240 views) or opt (~5min)")
    parser.add_argument("--debug_single_frame", action="store_true",
                        help="Bake only 1 frame for camera alignment check")
    args = parser.parse_args()

    texture_resolution = args.texture_resolution
    debug_single_frame = args.debug_single_frame

    output_dir = os.path.join(args.output_dir, f"{args.sha256[:7]}_{args.sha256[8:]}")
    os.makedirs(output_dir, exist_ok=True)

    # Load mesh
    mesh = trimesh.load(args.glb_path, force='mesh')
    vertices = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.uint32)

    # Convert to Blender coordinate system
    R_blender = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
    vertices = vertices @ R_blender
    print(f"Mesh bounds: min={vertices.min():.3f}, max={vertices.max():.3f}")

    # UV unwrap with xatlas
    vmapping, indices, uvs = xatlas.parametrize(vertices, faces)
    vertices = vertices[vmapping]
    faces = indices

    # Process H and V observations
    v_obs, v_masks, v_extrinsics, v_intrinsics = process_camera_set(
        args.albedo_v_video, args.v_meta, debug_single_frame)
    h_obs, h_masks, h_extrinsics, h_intrinsics = process_camera_set(
        args.albedo_h_video, args.h_meta, debug_single_frame)

    Image.fromarray(v_obs[0]).save(f'{output_dir}/v_frame0.png')

    # Load MR observations
    hmr_obs = load_observations_from_video(args.mr_h_video)
    vmr_obs = load_observations_from_video(args.mr_v_video)
    if debug_single_frame:
        hmr_obs = [hmr_obs[0]]
        vmr_obs = [vmr_obs[0]]
    hm_obs, hr_obs = split_mr(hmr_obs)
    vm_obs, vr_obs = split_mr(vmr_obs)

    output_mesh_path = f'{output_dir}/bake_{texture_resolution}_a.glb'

    if os.path.exists(output_mesh_path):
        print(f"Output already exists: {output_mesh_path}")
        return

    # Bake albedo
    texture = bake_texture(
        vertices, faces, uvs,
        v_obs + h_obs, v_masks + h_masks,
        v_extrinsics + h_extrinsics,
        v_intrinsics + h_intrinsics,
        texture_size=texture_resolution, mode=args.bake_mode,
        lambda_tv=0.01, verbose=True,
    )
    Image.fromarray(texture).save(f'{output_dir}/texture.png')

    # Bake metallic + roughness
    mr_views_ext = v_extrinsics + h_extrinsics
    mr_views_int = v_intrinsics + h_intrinsics
    mr_masks = v_masks + h_masks
    m_obs = vm_obs + hm_obs
    r_obs = vr_obs + hr_obs

    m_texture = bake_texture(vertices, faces, uvs, m_obs, mr_masks,
                             mr_views_ext, mr_views_int,
                             texture_size=texture_resolution, mode='fast',
                             lambda_tv=0.01, verbose=True)
    r_texture = bake_texture(vertices, faces, uvs, r_obs, mr_masks,
                             mr_views_ext, mr_views_int,
                             texture_size=texture_resolution, mode='fast',
                             lambda_tv=0.01, verbose=True)

    Image.fromarray(m_texture).save(f'{output_dir}/texture_metallic.png')
    Image.fromarray(r_texture).save(f'{output_dir}/texture_roughness.png')

    # Combine metallic (R) + roughness (G) into RGBA texture
    mr_combined = np.dstack([
        m_texture[:, :, 0],
        r_texture[:, :, 0],
        np.zeros_like(r_texture[:, :, 0]),
        np.full(r_texture[:, :, 0].shape, 255, dtype=np.uint8),
    ])

    # Export PBR GLB
    vertices_out = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(texture),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(mr_combined),
        metallicFactor=1.0,
        roughnessFactor=1.0,
    )
    glb = trimesh.Trimesh(vertices_out, faces,
                          visual=trimesh.visual.TextureVisuals(uv=uvs, material=material))
    glb.export(output_mesh_path)
    print(f"Done: {output_mesh_path}")


if __name__ == "__main__":
    main()
