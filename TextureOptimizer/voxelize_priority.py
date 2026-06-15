#!/usr/bin/env python3
"""
Priority-based baking: H frames first (high priority), then V fills gaps, then BFS.
"""
import os, sys, argparse, tempfile, shutil, pickle, numpy as np, torch, cv2, time
from subprocess import call
from PIL import Image
from pathlib import Path
import o_voxel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BLENDER_PATH = os.environ.get('BLENDER_PATH', 'blender')
DUMP_PBR_SCRIPT = os.path.join(SCRIPT_DIR, 'blender_script', 'dump_pbr_revise.py')


def extract_frames_from_video(video_path, col_index=2, num_cols=4):
    """Extract frames from multi-column video, taking column col_index."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames = {}
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        col_w = w // num_cols
        x1, x2 = col_index * col_w, (col_index + 1) * col_w
        panel = frame[:, x1:x2, :]
        panel = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
        frames[idx] = Image.fromarray(panel)
        idx += 1
    cap.release()
    print(f"  Extracted {len(frames)} frames from {video_path} (col {col_index}/{num_cols})")
    return frames


def install_blender():
    if os.path.exists(BLENDER_PATH):
        return
    print("Installing Blender...")
    os.system('sudo apt-get update')
    os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6 libxfixes3 libgl1')
    raise RuntimeError("Blender not found. Set BLENDER_PATH env var.")


def extract_pbr_with_blender(input_glb, output_pickle):
    if not os.path.exists(DUMP_PBR_SCRIPT):
        raise FileNotFoundError(f"Blender script not found: {DUMP_PBR_SCRIPT}")
    args = [BLENDER_PATH, '-b', '-P', DUMP_PBR_SCRIPT, '--',
            '--object', os.path.expanduser(input_glb),
            '--output_path', os.path.expanduser(output_pickle)]
    env = os.environ.copy()
    blender_lib_dir = os.path.join(os.path.dirname(os.path.abspath(BLENDER_PATH)), 'lib')
    env['LD_LIBRARY_PATH'] = blender_lib_dir + ':' + env.get('LD_LIBRARY_PATH', '')
    ret = call(args, env=env)
    if ret != 0 or not os.path.exists(output_pickle):
        raise RuntimeError(f"Blender failed to extract PBR from {input_glb}")


def _to_tensor(x):
    return torch.from_numpy(np.array(x)).float()


def color_voxels_with_views(voxel_rgb, voxel_conf, visibility, coord, extr, intr, frames,
                             view_list, start_voxel_idx=0):
    """Color voxels from a list of views. Only colors voxels with conf==0 (uncolored)."""
    H, W = 4096, 4096
    voxel_count = coord.shape[0]
    newly_colored = 0

    for view_idx, (mode, cam_idx) in enumerate(view_list):
        vf = cam_idx  # For stride=1, camera index = video frame index
        if vf not in frames:
            continue

        img = np.array(frames[vf]).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).cuda()

        # Project voxels to this camera
        ext = extr[cam_idx].cuda()
        intrin = intr[cam_idx].cuda()

        # Camera center in world
        R = ext[:3, :3]
        t = ext[:3, 3]
        cam_center = -R.T @ t

        # Project each voxel
        world_coord = coord.cuda()
        cam_coord = (R @ world_coord.T + t.unsqueeze(1)).T  # [N, 3]

        # Only process front-facing voxels (z > 0)
        z = cam_coord[:, 2]
        valid_depth = z > 0

        # Project to image
        u = intrin[0, 0] * cam_coord[:, 0] / z + intrin[0, 2]
        v = intrin[1, 1] * cam_coord[:, 1] / z + intrin[1, 2]

        u = u.long()
        v = v.long()
        in_frame = (u >= 0) & (u < W) & (v >= 0) & (v < H) & valid_depth

        # For visible voxels that are currently uncolored, assign color
        visible = visibility[cam_idx].cuda() & in_frame
        uncolored = (voxel_conf == 0)
        to_color = visible & uncolored

        if to_color.sum() == 0:
            continue

        # Read pixel colors
        pixel_colors = img_tensor[v[to_color], u[to_color]]  # [K, 3]
        voxel_rgb[to_color] = pixel_colors
        voxel_conf[to_color] = 1.0
        newly_colored += to_color.sum().item()

    return newly_colored


def main():
    parser = argparse.ArgumentParser(description="Priority baking: H first, then V, then BFS")
    parser.add_argument("input_glb")
    parser.add_argument("--video", default=None, help="H video")
    parser.add_argument("--video_v", default=None, help="V video")
    parser.add_argument("--video_num_cols", type=int, default=4)
    parser.add_argument("--video_col", type=int, default=2)
    parser.add_argument("--video_v_num_cols", type=int, default=4)
    parser.add_argument("--video_v_col", type=int, default=2)
    parser.add_argument("--output_vxz", help="Output .vxz path")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--max_lock_angle", type=float, default=60.0)
    args = parser.parse_args()

    install_blender()

    # Step 0: Blender PBR dump
    tmp_dir = tempfile.mkdtemp()
    temp_pickle = os.path.join(tmp_dir, 'pbr_dump.pickle')
    output_pickle = args.output_vxz.replace('.vxz', '.pickle')

    if os.path.exists(output_pickle):
        print(f"✅ Reusing existing pickle: {output_pickle}")
    else:
        extract_pbr_with_blender(args.input_glb, temp_pickle)
        shutil.copy(temp_pickle, output_pickle)
        print(f"✅ Saved PBR dump to: {output_pickle}")

    with open(output_pickle, 'rb') as f:
        data = pickle.load(f)

    # Load camera data
    h_extr = _to_tensor(data['h_extrinsics_filtered'])
    h_intr = _to_tensor(data['h_intrinsics_normalized_filtered'])
    v_extr = _to_tensor(data['v_extrinsics_filtered'])
    v_intr = _to_tensor(data['v_intrinsics_normalized_filtered'])
    num_h, num_v = h_extr.shape[0], v_extr.shape[0]

    # Voxelize
    visibility, coord = o_voxel.voxelize(temp_pickle, args.resolution)
    N = coord.shape[0]
    print(f"✅ Voxelized: {N} voxels at resolution {args.resolution}")

    # Load H frames
    h_frames = extract_frames_from_video(args.video, args.video_col, args.video_num_cols)
    n_h_video = len(h_frames)

    # Load V frames
    v_frames = None
    if args.video_v:
        v_frames = extract_frames_from_video(args.video_v, args.video_v_col, args.video_v_num_cols)

    # Auto-detect stride
    h_stride = 1
    if n_h_video < num_h and abs(n_h_video * 2 - num_h) <= 2:
        h_stride = 2
        remapped = {}
        for vf, img in h_frames.items():
            cam_idx = vf * 2
            if cam_idx < num_h:
                remapped[cam_idx] = img
        h_frames = remapped
    print(f"📹 H: {len(h_frames)} frames, {num_h} cameras, stride={h_stride}")

    n_v_video = len(v_frames) if v_frames else 0
    v_stride = 1
    if v_frames and n_v_video < num_v and abs((n_v_video-1) * 2 - num_v) <= 2:
        v_stride = 2
        remapped = {}
        for vf, img in v_frames.items():
            if vf >= n_v_video - 1:
                break
            cam_idx = vf * 2
            if cam_idx < num_v:
                remapped[cam_idx] = img
        v_frames = remapped
    if v_frames:
        print(f"📹 V: {len(v_frames)} frames, {num_v} cameras, stride={v_stride}")

    # Build view lists
    h_views = [('h', i) for i in range(0, num_h, h_stride) if i in h_frames]
    v_views = []
    if v_frames:
        for i in range(0, num_v, v_stride):
            if i in v_frames:
                v_views.append(('v', i))

    print(f"📹 Total: {len(h_views)} H views + {len(v_views)} V views")

    # Initialize voxel color arrays
    voxel_rgb = torch.zeros(N, 3, device='cuda')
    voxel_conf = torch.zeros(N, device='cuda')

    # === Phase 1: H priority ===
    print(f"\n🔧 Phase 1: H-track ({len(h_views)} views)")
    n_h_colored = color_voxels_with_views(voxel_rgb, voxel_conf, visibility,
                                            coord, h_extr, h_intr, h_frames, h_views)
    h_uncolored = (voxel_conf == 0).sum().item()
    print(f"  H colored: {n_h_colored}/{N} voxels ({100*n_h_colored/N:.1f}%), remaining: {h_uncolored}")

    # === Phase 2: V fill gaps ===
    if v_views and h_uncolored > 0:
        print(f"\n🔧 Phase 2: V-track fills remaining {h_uncolored} voxels ({len(v_views)} views)")
        n_v_colored = color_voxels_with_views(voxel_rgb, voxel_conf, visibility,
                                                coord, v_extr, v_intr, v_frames, v_views)
        v_uncolored = (voxel_conf == 0).sum().item()
        print(f"  V colored: {n_v_colored} voxels, remaining: {v_uncolored}")
    else:
        v_uncolored = h_uncolored

    # === Phase 3: BFS gap-fill ===
    if v_uncolored > 0:
        print(f"\n🔧 Phase 3: BFS gap-fill for {v_uncolored} voxels")
        colored_mask = voxel_conf > 0
        # BFS: for each uncolored voxel, find nearest colored neighbor
        t0 = time.time()
        uncolored_idx = torch.where(~colored_mask)[0].cpu().numpy()
        colored_coord = coord[colored_mask.cpu().numpy()]
        colored_rgb = voxel_rgb[colored_mask]

        for uc_idx in uncolored_idx:
            uc_pos = coord[uc_idx]
            dists = ((colored_coord - uc_pos.unsqueeze(0)) ** 2).sum(dim=1)
            nearest = dists.argmin()
            voxel_rgb[uc_idx] = colored_rgb[nearest]
            voxel_conf[uc_idx] = 0.5  # Mark as gap-filled

        filled = (voxel_conf > 0).sum().item()
        print(f"  BFS filled: {filled - (N - v_uncolored)} voxels ({time.time()-t0:.1f}s)")
        print(f"  After gap-fill: {filled}/{N} covered ({100*filled/N:.1f}%)")

    # Save VXZ
    voxel_rgb_np = (voxel_rgb.cpu().numpy() * 255).astype(np.uint8)
    o_voxel.save_vxz(args.output_vxz, voxel_rgb_np, coord.numpy(), {})
    print(f"\n✅ Saved VXZ: {args.output_vxz}")

    # Render
    render_script = os.path.join(SCRIPT_DIR, 'render_vxz1024.py')
    if os.path.exists(render_script):
        print("🎨 Rendering...")
        py = sys.executable
        mp4_out = args.output_vxz.replace('.vxz', '.mp4')
        ret = call([py, render_script, args.output_vxz])
        if ret == 0:
            print(f"✅ Render: {mp4_out}")


if __name__ == "__main__":
    main()
