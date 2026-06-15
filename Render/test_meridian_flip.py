#!/usr/bin/env python3
"""
Test script: render a single frame from meridian_270 vs meridian_90 to verify
that switching meridian achieves the same result as left-right flipping.

Usage:
    conda run -n bpy40 python test_meridian_flip.py \\
        --input_file /path/to/model.glb --output_dir ./test_flip --frame_idx 0
"""
import os, argparse, json

import bpy
import numpy as np
import imageio
imageio.plugins.freeimage.download()

from ink3d_render import SceneManager
from ink3d_render.camera_utils import add_camera
from ink3d_render.camera import get_vertical_camera_path_360
from ink3d_render.engine import init_render_engine
from ink3d_render.environment import set_env_map
from ink3d_render.importer import load_file
from ink3d_render.render_output import enable_color_output
from ink3d_render.utils import convert_normal_to_webp


def render_frame(model_path, output_path, args, trajectory_type):
    """Render a single frame at args.frame_idx for the given trajectory_type."""

    # Init
    init_render_engine(args.engine)
    scene_manager = SceneManager()
    scene_manager.clear(reset_keyframes=True)
    load_file(model_path)

    # Scene setup
    scene_manager.smooth()
    scene_manager.normalize_scene(args.scene_scale)
    scene_manager.set_materials_opaque()
    scene_manager.set_material_transparency(False)

    env_path = args.env_map
    if not os.path.isabs(env_path):
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), env_path)
    if os.path.exists(env_path):
        set_env_map(env_path)

    # Generate V cameras
    cam_pos, cam_mats, elevations, azimuths = get_vertical_camera_path_360(
        center=tuple(args.camera_center),
        radius=args.camera_radius,
        num_cameras=args.num_cameras,
        full_circle=True,
        trajectory_type=trajectory_type,
        start_angle_offset=np.pi / 2,
    )

    cameras = []
    for i, camera_mat in enumerate(cam_mats):
        camera = add_camera(camera_mat, add_frame=i < len(cam_mats) - 1)
        cameras.append(camera)

    os.makedirs(output_path, exist_ok=True)
    enable_color_output(
        width=args.width, height=args.height,
        output_dir=output_path, mode="PNG",
        film_transparent=True,
    )

    # Only render the target frame
    target_frame = args.frame_idx + 1  # Blender frames are 1-indexed
    bpy.context.scene.frame_set(target_frame)

    # Build composite
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    tree.nodes.clear()

    render_layers = tree.nodes.new("CompositorNodeRLayers")
    alpha_over = tree.nodes.new("CompositorNodeAlphaOver")
    composite = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(render_layers.outputs["Image"], alpha_over.inputs[2])
    tree.links.new(alpha_over.outputs["Image"], composite.inputs["Image"])

    bpy.ops.render.render(write_still=True)

    # Find rendered file
    import glob
    rendered = sorted(glob.glob(os.path.join(output_path, "render_*.png")))
    if rendered:
        print(f"  {trajectory_type}: {rendered[-1]}")
        return rendered[-1]
    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./test_flip")
    parser.add_argument("--frame_idx", type=int, default=0, help="Which camera index to render (0-based)")
    parser.add_argument("--num_cameras", type=int, default=120)
    parser.add_argument("--engine", type=str, default="CYCLES_GPU")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--camera_center", type=float, nargs=3, default=[0, 0, 0])
    parser.add_argument("--camera_radius", type=float, default=1.5)
    parser.add_argument("--scene_scale", type=float, default=1.0)
    parser.add_argument("--env_map", type=str, default="assets/env_textures/brown_photostudio_02_1k.exr")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Rendering frame {args.frame_idx}: meridian_270 vs meridian_90")

    out_270 = os.path.join(args.output_dir, "meridian_270")
    out_90 = os.path.join(args.output_dir, "meridian_90")

    render_frame(args.input_file, out_270, args, "meridian_270")
    render_frame(args.input_file, out_90, args, "meridian_90")

    print(f"\nCompare outputs:\n  {out_270}/\n  {out_90}/")


if __name__ == "__main__":
    main()
