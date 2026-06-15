"""
PBR render from vxz + mesh → mp4 video.

Usage:
    # With pickle mesh
    conda run -n trellis2 python render_vxz.py --vxz path/to.vxz --mesh path/to.pickle -o out.mp4

    # With GLB/OBJ mesh
    conda run -n trellis2 python render_vxz.py --vxz path/to.vxz --mesh path/to.glb -o out.mp4

    # Custom options
    conda run -n trellis2 python render_vxz.py --vxz f.vxz --mesh f.pickle -o out.mp4 \
        --resolution 512 --num_frames 60 --envmap assets/hdri/forest.exr
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import argparse
import pickle
import numpy as np
import torch
import cv2
import imageio
import trimesh
from PIL import Image
import o_voxel
from trellis2.representations import MeshWithVoxel
from trellis2.renderers import EnvMap
from trellis2.utils.render_utils import render_video, render_frames, make_pbr_vis_frames, yaw_pitch_r_fov_to_extrinsics_intrinsics


def load_mesh_pickle(path):
    """Load pickle mesh (Blender Z-up) → vertices/faces converted to Y-up."""
    with open(path, 'rb') as f:
        dump = pickle.load(f)
    objects = [obj for obj in dump['objects'] if obj['vertices'].size > 0 and obj['faces'].size > 0]
    if not objects:
        raise ValueError(f"No valid geometry in {path}")
    all_vertices = np.concatenate([obj['vertices'] for obj in objects], axis=0)
    all_faces = []
    offset = 0
    for obj in objects:
        all_faces.append(obj['faces'] + offset)
        offset += len(obj['vertices'])
    all_faces = np.concatenate(all_faces, axis=0)
    # Use coordinates as-is (pickle from our pipeline is already in the same space as vxz)
    return torch.tensor(all_vertices, dtype=torch.float32), torch.tensor(all_faces, dtype=torch.int32)


def load_mesh_file(path):
    """Load GLB/OBJ/PLY mesh via trimesh → vertices/faces."""
    mesh = trimesh.load(path, force='mesh')
    return (
        torch.tensor(np.array(mesh.vertices), dtype=torch.float32),
        torch.tensor(np.array(mesh.faces), dtype=torch.int32),
    )


def load_vxz(path, metallic=0.0, roughness=0.9):
    """Load vxz → coords (N,3), attrs (N,6) in [0,1]."""
    coords, attr = o_voxel.io.read_vxz(path, num_threads=4)
    if 'top6' in attr:
        attr['base_color'] = attr['top6'][:, :3]
        attr['roughness'] = torch.ones_like(attr['base_color'][:, :1]) * 255 * roughness
        attr['metallic'] = torch.ones_like(attr['base_color'][:, :1]) * 255 * metallic
        attr['alpha'] = torch.ones_like(attr['base_color'][:, :1]) * 255
    attrs = torch.cat([
        attr['base_color'], attr['metallic'], attr['roughness'], attr['alpha']
    ], dim=-1).float() / 255.0
    return coords, attrs


def build_mesh_with_voxel(vertices, faces, coords, attrs, resolution=1024):
    """Assemble MeshWithVoxel from components."""
    return MeshWithVoxel(
        vertices=vertices,
        faces=faces,
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1.0 / resolution,
        coords=coords,
        attrs=attrs,
        voxel_shape=torch.Size([1, 6, resolution, resolution, resolution]),
        layout={
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        },
    )


def render_mp4(mesh_voxel, envmap, output_path, resolution=1024, num_frames=120, fps=15, bg_color=(0, 0, 0), shaded_only=False, turntable=False, elevation=20.0):
    """Render MeshWithVoxel → mp4 with PBR shading."""
    mesh_voxel = mesh_voxel.to('cuda')
    if turntable:
        yaws = torch.linspace(np.pi, np.pi + 2 * np.pi, num_frames + 1)[:num_frames].tolist()
        pitch_rad = elevation / 180.0 * np.pi
        pitch = [pitch_rad] * num_frames
        extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, 2, 40)
        result = render_frames(mesh_voxel, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, envmap=envmap)
    else:
        result = render_video(mesh_voxel, resolution=resolution, num_frames=num_frames, envmap=envmap)
    # Apply background color using alpha channel
    if bg_color != (0, 0, 0):
        bg = np.array(bg_color, dtype=np.float32) * 255
        for key in ['shaded', 'normal', 'base_color']:
            if key in result:
                for i in range(len(result[key])):
                    alpha = result['alpha'][i].astype(np.float32) / 255.0
                    if alpha.ndim == 2:
                        alpha = alpha[..., None]
                    img = result[key][i].astype(np.float32)
                    result[key][i] = np.clip(img * alpha + bg[None, None, :] * (1 - alpha), 0, 255).astype(np.uint8)
    if shaded_only:
        video = []
        for i in range(len(result['shaded'])):
            img = Image.fromarray(result['shaded'][i]).resize((resolution, resolution))
            video.append(np.array(img))
    else:
        video = make_pbr_vis_frames(result)
    imageio.mimsave(output_path, video, fps=fps)
    print(f"Saved: {output_path} ({len(video)} frames)")


def main():
    parser = argparse.ArgumentParser(description="PBR render: vxz + mesh → mp4")
    parser.add_argument("--vxz", type=str, required=True, help="Path to .vxz file")
    parser.add_argument("--mesh", type=str, required=True, help="Mesh file (.pickle / .glb / .obj / .ply)")
    parser.add_argument("-o", "--output", type=str, default="render.mp4", help="Output mp4 path")
    parser.add_argument("--envmap", type=str, default=None, help="HDR envmap (.exr). Default: assets/hdri/forest.exr")
    parser.add_argument("--resolution", type=int, default=1024, help="Render resolution")
    parser.add_argument("--voxel_resolution", type=int, default=1024, help="Voxel grid resolution")
    parser.add_argument("--num_frames", type=int, default=120, help="Number of video frames")
    parser.add_argument("--fps", type=int, default=15, help="Video FPS")
    parser.add_argument("--metallic", type=float, default=0.0, help="Default metallic value (0-1)")
    parser.add_argument("--roughness", type=float, default=0.9, help="Default roughness value (0-1)")
    parser.add_argument("--white_bg", action="store_true", help="Use white background")
    parser.add_argument("--shaded_only", action="store_true", help="Output only PBR shaded result (no debug panels)")
    parser.add_argument("--turntable", action="store_true", help="Horizontal turntable rotation (fixed elevation)")
    parser.add_argument("--elevation", type=float, default=20.0, help="Camera elevation in degrees (used with --turntable)")
    args = parser.parse_args()

    # Envmap
    if args.envmap is None:
        args.envmap = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets/hdri/forest.exr")
    print(f"Loading envmap: {args.envmap}")
    envmap = EnvMap(torch.tensor(
        cv2.cvtColor(cv2.imread(args.envmap, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
        dtype=torch.float32, device='cuda',
    ))

    # Mesh (auto-detect by extension)
    ext = os.path.splitext(args.mesh)[1].lower()
    if ext == '.pickle':
        print(f"Loading pickle mesh: {args.mesh}")
        vertices, faces = load_mesh_pickle(args.mesh)
    elif ext in ('.glb', '.gltf', '.obj', '.ply', '.stl'):
        print(f"Loading mesh: {args.mesh}")
        vertices, faces = load_mesh_file(args.mesh)
    else:
        raise ValueError(f"Unsupported mesh format: {ext}")
    print(f"  Vertices: {vertices.shape[0]}, Faces: {faces.shape[0]}")

    # Vxz
    print(f"Loading vxz: {args.vxz}")
    coords, attrs = load_vxz(args.vxz, metallic=args.metallic, roughness=args.roughness)
    print(f"  Voxels: {coords.shape[0]}, Attrs: {attrs.shape[1]}ch")

    # Build & render
    mesh_voxel = build_mesh_with_voxel(vertices, faces, coords, attrs, args.voxel_resolution)
    render_mp4(mesh_voxel, envmap, args.output, args.resolution, args.num_frames, args.fps,
               bg_color=(1, 1, 1) if args.white_bg else (0, 0, 0),
               shaded_only=args.shaded_only,
               turntable=args.turntable, elevation=args.elevation)


if __name__ == "__main__":
    main()
