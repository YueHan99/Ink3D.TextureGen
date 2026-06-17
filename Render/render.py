#!/usr/bin/env python3
"""
Ink3D Render — Multi-pass GLB rendering with configurable camera orbits.

Renders a single GLB model with horizontal (H) or vertical (V) camera orbit,
producing color, depth, normal, albedo, and position outputs as images and videos.

Usage:
    python render.py --input_file model.glb --output_dir ./output --orbit horizontal
    python render.py --input_file model.glb --output_dir ./output --orbit vertical --num_cameras 60
"""

import json
import os
import argparse
from glob import glob

import bpy
import numpy as np
import imageio

imageio.plugins.freeimage.download()

from ink3d_render import SceneManager
from ink3d_render.camera_utils import add_camera
from ink3d_render.camera import get_camera_positions_on_sphere, get_vertical_camera_path_360
from ink3d_render.engine import init_render_engine
from ink3d_render.environment import set_env_map
from ink3d_render.importer import load_file
from ink3d_render.render_output import (
    enable_color_output,
    enable_depth_output,
    enable_normals_output,
    enable_position_output,
)
from ink3d_render.render_output_a import enable_pbr_output
from ink3d_render.utils import convert_depth_to_webp, convert_normal_to_webp


def normalize_frame_with_fixed_params(img, min_val=-0.5, max_val=0.5):
    """Normalize position map frame to [0, 255] range."""
    if img is None:
        return None
    if len(img.shape) == 3 and img.shape[2] >= 3:
        data_to_process = img[:, :, :3]
    else:
        data_to_process = img
    normalized = np.clip((data_to_process - min_val) / (max_val - min_val), 0, 1)
    return (normalized * 255).astype(np.uint8)


def convert_position_to_png(exr_files, png_files):
    """Convert position EXR files to PNG."""
    for exr_file, png_file in zip(exr_files, png_files):
        pos_img = imageio.imread(exr_file, format='EXR-FI')
        pos_img_normalized = normalize_frame_with_fixed_params(pos_img, min_val=-0.5, max_val=0.5)
        os.makedirs(os.path.dirname(png_file), exist_ok=True)
        imageio.imwrite(png_file, pos_img_normalized)


def clear_animation_data():
    """Clear all animation data from the scene."""
    print("Clearing animation data...")
    actions = list(bpy.data.actions)
    for action in actions:
        bpy.data.actions.remove(action, do_unlink=True)
    print(f"  Removed {len(actions)} actions")

    for obj in bpy.data.objects:
        if obj.animation_data:
            nla_tracks = list(obj.animation_data.nla_tracks)
            for track in nla_tracks:
                strips = list(track.strips)
                for strip in strips:
                    track.strips.remove(strip)
                obj.animation_data.nla_tracks.remove(track)
            obj.animation_data.action = None

    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.data.shape_keys:
            shape_keys = obj.data.shape_keys.key_blocks
            for i in range(len(shape_keys) - 1, 0, -1):
                obj.shape_key_remove(shape_keys[i])

    for obj in bpy.data.objects:
        if obj.type == "ARMATURE":
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode='POSE')
            bpy.ops.pose.transforms_clear()
            bpy.ops.pose.select_all(action='SELECT')
            bpy.ops.pose.transforms_clear()
            bpy.ops.object.mode_set(mode='OBJECT')

    for obj in bpy.data.objects:
        constraints = list(obj.constraints)
        for constraint in constraints:
            obj.constraints.remove(constraint)

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 1
    bpy.context.scene.frame_current = 1
    bpy.context.view_layer.update()
    print("Animation data cleared.")


def render_single_model(model_path, render_dir, args):
    """Render a single model with all configured outputs.

    Output layout:
        render_dir/images/   — per-frame images
        render_dir/          — videos (.mp4) and meta.json
    """
    image_output_dir = os.path.join(render_dir, "images")
    video_output_dir = render_dir
    metadata_output_dir = render_dir

    os.makedirs(image_output_dir, exist_ok=True)

    # 1. Init engine and scene
    init_render_engine(args.engine)
    scene_manager = SceneManager()
    scene_manager.clear(reset_keyframes=True)

    # 2. Import model
    load_file(model_path)
    clear_animation_data()

    # 3. Process scene
    scene_manager.smooth()
    scene_manager.normalize_scene(args.scene_scale)
    scene_manager.set_materials_opaque()
    scene_manager.set_material_transparency(False)

    # 4. Set environment
    env_path = args.env_map
    if not os.path.isabs(env_path):
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), env_path)
    if os.path.exists(env_path):
        set_env_map(env_path)
    else:
        print(f"Warning: Env map not found at {env_path}")

    # 5. Setup cameras based on orbit type
    if args.orbit == "horizontal":
        cam_pos, cam_mats, elevations, azimuths = get_camera_positions_on_sphere(
            center=args.camera_center,
            radius=args.camera_radius,
            elevations=[0],
            num_camera_per_layer=args.num_cameras,
            azimuth_offset=args.azimuth_offset,
        )
    elif args.orbit == "vertical":
        cam_pos, cam_mats, elevations, azimuths = get_vertical_camera_path_360(
            center=tuple(args.camera_center),
            radius=args.camera_radius,
            num_cameras=args.num_cameras,
            full_circle=True,
            trajectory_type='meridian_270',
            start_angle_offset=np.pi / 2,
            flip_x=args.flip_x,
        )
    else:
        raise ValueError(f"Unknown orbit type: {args.orbit}")

    cameras = []
    for i, camera_mat in enumerate(cam_mats):
        camera = add_camera(camera_mat, add_frame=i < len(cam_mats) - 1)
        cameras.append(camera)

    # 6. Configure render outputs
    enable_color_output(
        width=args.width, height=args.height,
        output_dir=image_output_dir, mode="PNG",
        film_transparent=True,
    )
    enable_depth_output(image_output_dir)
    enable_normals_output(image_output_dir)
    enable_pbr_output(
        output_dir=image_output_dir,
        attr_name="Base Color",
        file_prefix="albedo_",
        color_mode="RGBA",
    )
    if args.mr:
        enable_pbr_output(output_dir=image_output_dir, attr_name="Metallic", file_prefix="metallic_", color_mode="BW")
        enable_pbr_output(output_dir=image_output_dir, attr_name="Roughness", file_prefix="roughness_", color_mode="BW")
    enable_position_output(
        output_dir=image_output_dir,
        file_prefix="position_",
        space="WORLD",
        file_format="OPEN_EXR",
    )

    # 7. Render
    scene_manager.render()

    # 8. Post-process
    meta_info = {
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "orbit": args.orbit,
        "num_cameras": args.num_cameras,
        "pbr_channels": {},
        "locations": [],
    }

    min_depth, scale = None, None

    # Depth: EXR → PNG
    render_files = sorted(glob(os.path.join(image_output_dir, "depth_*.exr")))
    output_files = [f.replace(".exr", ".png") for f in render_files]
    if render_files:
        min_depth, scale = convert_depth_to_webp(render_files, output_files)

    # Normal: EXR → PNG
    for file in os.listdir(image_output_dir):
        if file.startswith("normal_") and file.endswith(".exr"):
            filepath = os.path.join(image_output_dir, file)
            render_filepath = filepath.replace("normal_", "render_")
            png_path = filepath.replace(".exr", ".png")
            convert_normal_to_webp(filepath, png_path, render_filepath)

    # Create videos
    video_params = ['-crf', '10', '-preset', 'medium', '-pix_fmt', 'yuv444p']

    # RGB + Mask video
    render_files = sorted(glob(os.path.join(image_output_dir, "render_*.png")))
    if render_files:
        rgb_path = os.path.join(video_output_dir, "rgb.mp4")
        mask_path = os.path.join(video_output_dir, "mask.mp4")
        with imageio.get_writer(rgb_path, fps=args.fps, codec='libx264', output_params=video_params) as rgb_w, \
             imageio.get_writer(mask_path, fps=args.fps, codec='libx264', output_params=video_params) as mask_w:
            for file in render_files:
                image = imageio.imread(file)
                mask = image[:, :, 3]
                white_bg = np.ones((args.height, args.width, 3), dtype=np.uint8) * 255
                alpha = image[:, :, 3:4] / 255.0
                white_image = image[:, :, :3] * alpha + white_bg * (1 - alpha)
                rgb_w.append_data(white_image.astype(np.uint8))
                mask_w.append_data(mask)
        meta_info["pbr_channels"]["color"] = "rgb.mp4"
        meta_info["pbr_channels"]["mask"] = "mask.mp4"

    # Normal video
    normal_files = sorted(glob(os.path.join(image_output_dir, "normal_*.png")))
    if normal_files:
        video_path = os.path.join(video_output_dir, "normal.mp4")
        with imageio.get_writer(video_path, fps=args.fps, codec='libx264', output_params=video_params) as writer:
            for file in normal_files:
                writer.append_data(imageio.imread(file))
        meta_info["pbr_channels"]["normal"] = "normal.mp4"

    # Depth video
    depth_files = sorted(glob(os.path.join(image_output_dir, "depth_*.png")))
    if depth_files:
        video_path = os.path.join(video_output_dir, "depth.mp4")
        with imageio.get_writer(video_path, fps=args.fps, codec='libx264', output_params=video_params) as writer:
            for file in depth_files:
                writer.append_data(imageio.imread(file))
        meta_info["pbr_channels"]["depth"] = "depth.mp4"

    # Albedo video
    albedo_files = sorted(glob(os.path.join(image_output_dir, "albedo_*.png")))
    if albedo_files:
        video_path = os.path.join(video_output_dir, "albedo.mp4")
        with imageio.get_writer(video_path, fps=args.fps, codec='libx264', output_params=video_params) as writer:
            for file in albedo_files:
                writer.append_data(imageio.imread(file))
        meta_info["pbr_channels"]["albedo"] = "albedo.mp4"

    # Position video
    position_files = sorted(glob(os.path.join(image_output_dir, "position_*.exr")))
    if position_files:
        png_files = [f.replace(".exr", ".png") for f in position_files]
        convert_position_to_png(position_files, png_files)
        video_path = os.path.join(video_output_dir, "position.mp4")
        with imageio.get_writer(video_path, fps=args.fps, codec='libx264', output_params=video_params) as writer:
            for file in png_files:
                writer.append_data(imageio.imread(file))
        meta_info["pbr_channels"]["position"] = "position.mp4"

    # MR (Metallic + Roughness) video
    if args.mr:
        metallic_files = sorted(glob(os.path.join(image_output_dir, "metallic_*.png")))
        roughness_files = sorted(glob(os.path.join(image_output_dir, "roughness_*.png")))
        if metallic_files and roughness_files:
            mr_path = os.path.join(video_output_dir, "mr.mp4")
            video_params = ['-crf', '10', '-preset', 'medium', '-pix_fmt', 'yuv444p']
            with imageio.get_writer(mr_path, fps=args.fps, codec='libx264', output_params=video_params) as writer:
                for mf, rf in zip(metallic_files, roughness_files):
                    m = imageio.imread(mf)
                    r = imageio.imread(rf)
                    # AOV output is 16-bit PNG (color_depth=16). Normalize to 8-bit.
                    if m.dtype == np.uint16 or m.max() > 255:
                        m = (m.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
                    if r.dtype == np.uint16 or r.max() > 255:
                        r = (r.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
                    m_flat = m if m.ndim == 2 else m[:, :, 0]
                    r_flat = r if r.ndim == 2 else r[:, :, 0]
                    # R=255 unused, G=roughness, B=metallic
                    mr_frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                    mr_frame[:, :, 0] = 255  # R: unused
                    mr_frame[:, :, 1] = r_flat  # G: roughness
                    mr_frame[:, :, 2] = m_flat  # B: metallic
                    writer.append_data(mr_frame)
            meta_info["pbr_channels"]["mr"] = "mr.mp4"

    # Save metadata
    for i in range(len(cam_pos)):
        meta_info["locations"].append({
            "index": f"{i:04d}",
            "projection_type": cameras[i].data.type,
            "ortho_scale": cameras[i].data.ortho_scale,
            "camera_angle_x": cameras[i].data.angle_x,
            "elevation": float(elevations[i]),
            "azimuth": float(azimuths[i]),
            "transform_matrix": cam_mats[i].tolist(),
            "depth_min": float(min_depth) if min_depth is not None else None,
            "depth_scale": float(scale) if scale is not None else None,
        })

    meta_file_path = os.path.join(metadata_output_dir, "meta.json")
    with open(meta_file_path, "w") as f:
        json.dump(meta_info, f, indent=4)

    print(f"Done: {model_path} → {video_output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ink3D Render: Multi-pass GLB rendering with H/V camera orbits"
    )

    # I/O
    parser.add_argument("--input_file", type=str, required=True, help="Input GLB file path")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Base output directory")
    parser.add_argument("--env_map", type=str, default="assets/env_textures/brown_photostudio_02_1k.exr",
                        help="HDR environment map path")

    # Render engine
    parser.add_argument("--engine", type=str, choices=["CYCLES_GPU", "CYCLES_CPU", "BLENDER_EEVEE"],
                        default="CYCLES_GPU", help="Render engine")

    # Image settings
    parser.add_argument("--width", type=int, default=1024, help="Render width")
    parser.add_argument("--height", type=int, default=1024, help="Render height")
    parser.add_argument("--fps", type=int, default=24, help="Video frame rate")

    # Camera orbit
    parser.add_argument("--orbit", type=str, choices=["horizontal", "vertical"],
                        default="horizontal", help="Camera orbit type: horizontal (H) or vertical (V)")
    parser.add_argument("--num_cameras", type=int, default=120, help="Number of camera positions")
    parser.add_argument("--camera_center", type=float, nargs=3, default=[0, 0, 0], help="Orbit center")
    parser.add_argument("--camera_radius", type=float, default=1.5, help="Orbit radius")
    parser.add_argument("--azimuth_offset", type=float, default=-90, help="Azimuth offset in degrees")

    # Scene
    parser.add_argument("--scene_scale", type=float, default=1.0, help="Scene normalization scale")
    parser.add_argument("--model_name", type=str, default=None, help="Override model name (default: GLB filename)")
    parser.add_argument("--flip_x", action="store_true", help="Flip right axis (left-right mirror) for vertical orbit")
    parser.add_argument("--mr", action="store_true", help="Enable Metallic/Roughness AOV output (mr.mp4)")

    return parser.parse_args()


def main():
    args = parse_args()

    model_path = args.input_file
    if not os.path.exists(model_path):
        print(f"Error: File not found: {model_path}")
        return
    if not model_path.lower().endswith(('.glb', '.obj', '.fbx', '.ply')):
        print(f"Error: Unsupported format: {model_path}")
        return

    model_name = args.model_name or os.path.splitext(os.path.basename(model_path))[0]
    orbit_prefix = "h" if args.orbit == "horizontal" else "v"
    render_dir = os.path.join(args.output_dir, model_name, f"{orbit_prefix}{args.num_cameras}")

    render_single_model(model_path, render_dir, args)


if __name__ == "__main__":
    main()
