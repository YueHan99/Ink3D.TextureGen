#!/usr/bin/env python3
"""
Render one frame from meridian_270, comparing: normal vs right-axis-flipped.
Normal: tangent=[1,0,0] → current output
Flipped: rot_matrix[:, 0] *= -1 → pure left-right mirror
"""
import os, sys, argparse, glob
from math import pi
import numpy as np
from scipy.spatial.transform import Rotation as R

import bpy
import imageio
imageio.plugins.freeimage.download()

from ink3d_render import SceneManager
from ink3d_render.camera_utils import add_camera
from ink3d_render.engine import init_render_engine
from ink3d_render.environment import set_env_map
from ink3d_render.importer import load_file
from ink3d_render.render_output import enable_color_output


def render_v_frame(model_path, output_dir, args, flip_x=False):
    init_render_engine(args.engine)
    scene_manager = SceneManager()
    scene_manager.clear(reset_keyframes=True)
    load_file(model_path)

    scene_manager.smooth()
    scene_manager.normalize_scene(args.scene_scale)
    scene_manager.set_materials_opaque()
    scene_manager.set_material_transparency(False)

    env_path = args.env_map
    if not os.path.isabs(env_path):
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), env_path)
    if os.path.exists(env_path):
        set_env_map(env_path)

    center = np.array(args.camera_center)
    radius = args.camera_radius
    num_cameras = args.num_cameras
    start_angle_offset = pi / 2
    azimuth_rad = 3 * pi / 2  # meridian_270

    angles = np.linspace(0, 2 * pi, num_cameras, endpoint=False) + start_angle_offset
    points, mats = [], []

    for angle in angles:
        elevation_rad = angle - pi / 2
        x = center[0] + radius * np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = center[1] + radius * np.cos(elevation_rad) * np.sin(azimuth_rad)
        z = center[2] + radius * np.sin(elevation_rad)
        cam_pos = np.array([x, y, z])
        points.append(cam_pos)

        direction = center - cam_pos
        direction = direction / np.linalg.norm(direction)
        tangent = np.array([1., 0., 0.])
        tangent = tangent - np.dot(tangent, direction) * direction
        tangent = tangent / (np.linalg.norm(tangent) + 1e-8)
        right = np.cross(tangent, direction)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(direction, right)
        rot_matrix = np.column_stack([right, up, -direction])
        rot_90 = R.from_rotvec(-pi / 2 * direction).as_matrix()
        rot_matrix = rot_90 @ rot_matrix

        if flip_x:
            rot_matrix[:, 0] = -rot_matrix[:, 0]

        mat = np.eye(4)
        mat[:3, :3] = rot_matrix
        mat[:3, 3] = cam_pos
        mats.append(mat)

    cameras = []
    for i, camera_mat in enumerate(mats):
        camera = add_camera(camera_mat, add_frame=i < len(mats) - 1)
        cameras.append(camera)

    os.makedirs(output_dir, exist_ok=True)
    enable_color_output(
        width=args.width, height=args.height,
        output_dir=output_dir, mode="PNG",
        film_transparent=True,
    )

    target_frame = args.frame_idx + 1
    bpy.context.scene.frame_start = target_frame
    bpy.context.scene.frame_end = target_frame
    bpy.context.scene.frame_set(target_frame)

    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    if tree is None:
        tree = bpy.data.node_groups.new('Compositing', 'CompositorNodeTree')
        bpy.context.scene.node_tree = tree
    if "Render Layers" not in tree.nodes:
        tree.nodes.new("CompositorNodeRLayers")

    bpy.ops.render.render(animation=True, write_still=True)

    rendered = sorted(glob.glob(os.path.join(output_dir, "render_*.png")))
    return rendered[-1] if rendered else None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_file", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./test_flip_render")
    p.add_argument("--frame_idx", type=int, default=30)
    p.add_argument("--num_cameras", type=int, default=120)
    p.add_argument("--engine", type=str, default="CYCLES_GPU")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--camera_center", type=float, nargs=3, default=[0., 0., 0.])
    p.add_argument("--camera_radius", type=float, default=1.5)
    p.add_argument("--scene_scale", type=float, default=1.0)
    p.add_argument("--env_map", type=str, default="assets/env_textures/brown_photostudio_02_1k.exr")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Rendering frame {args.frame_idx}: normal vs flip-x")
    r1 = render_v_frame(args.input_file, os.path.join(args.output_dir, "normal"), args, flip_x=False)
    r2 = render_v_frame(args.input_file, os.path.join(args.output_dir, "flipped"), args, flip_x=True)
    print(f"Normal:  {r1}")
    print(f"Flipped: {r2}")


if __name__ == "__main__":
    main()
