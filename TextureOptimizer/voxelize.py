#!/usr/bin/env python3
"""
Voxelize a .glb mesh with per-voxel color from the best of rendered views.

First-frame-fixed variant:
  - Phase 1: Voxelize mesh ONCE → get voxel coords + normals
  - Phase 2: Process first frame → lock all visible voxels (unconditional trust)
  - Phase 3: For remaining unlocked voxels, process other views and
             run greedy/Graph-Cut optimization.
  - Phase 4: Gap-fill via BFS.
"""
import os
import sys
import argparse
import tempfile
import shutil
import pickle
import numpy as np
import torch
from subprocess import call, DEVNULL
import utils3d
from PIL import Image
import torch.nn.functional as F
import o_voxel
from pathlib import Path
import subprocess
import gco
import time
import cv2

def extract_frames_from_video(video_path, num_cols=4, col_index=2):
    """
    从视频中提取帧，每帧分成 num_cols 等宽列，取第 col_index 列作为 h 视角图像。
    默认: 4列取第3列 (col_index=2)，兼容旧的4列视频。
    8列视频 (hv格式) 应传 num_cols=8, col_index=4 (第5列)。

    Returns:
        frames: dict {frame_idx: np.ndarray [H, W, 3] uint8 BGR→RGB}
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    frames = {}
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        if w % num_cols != 0:
            frame_idx += 1
            continue
        sub_w = w // num_cols
        img_h = frame[:, sub_w * col_index : sub_w * (col_index + 1)]
        # BGR → RGB
        frames[frame_idx] = cv2.cvtColor(img_h, cv2.COLOR_BGR2RGB)
        frame_idx += 1

    cap.release()
    print(f"📹 从视频提取 {len(frames)} 帧 (h 视角, 第{col_index+1}/{num_cols}列): {video_path}")
    return frames


def extract_frames_from_video_v(video_path, col_index=5, num_cols=8):
    """
    从 8 列拼接视频中提取指定列作为 v（竖直轨道）视角图像。
    默认取第 6 列（col_index=5）。

    Returns:
        frames: dict {frame_idx: np.ndarray [H, W, 3] uint8 RGB}
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    frames = {}
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        if w % num_cols != 0:
            frame_idx += 1
            continue
        sub_w = w // num_cols
        img_v = frame[:, sub_w * col_index : sub_w * (col_index + 1)]
        frames[frame_idx] = cv2.cvtColor(img_v, cv2.COLOR_BGR2RGB)
        frame_idx += 1

    cap.release()
    print(f"📹 从视频提取 {len(frames)} 帧 (v 视角, 第{col_index+1}/{num_cols}列): {video_path}")
    return frames


# --- 配置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BLENDER_PATH = os.environ.get('BLENDER_PATH', 'blender')
DUMP_PBR_SCRIPT = os.path.join(SCRIPT_DIR, 'blender_script', 'dump_pbr_revise.py')


def install_blender():
    if os.path.exists(BLENDER_PATH):
        return
    print("Installing Blender dependencies and binary...")
    os.system('sudo apt-get update')
    os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6 libxfixes3 libgl1')
    os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
    os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-4.5.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')


def extract_pbr_with_blender(input_glb, output_pickle):
    if not os.path.exists(DUMP_PBR_SCRIPT):
        raise FileNotFoundError(f"Blender script not found: {DUMP_PBR_SCRIPT}")
    args = [
        BLENDER_PATH, '-b', '-P', DUMP_PBR_SCRIPT,
        '--',
        '--object', os.path.expanduser(input_glb),
        '--output_path', os.path.expanduser(output_pickle)
    ]
    env = os.environ.copy()
    blender_lib_dir = os.path.join(os.path.dirname(os.path.abspath(BLENDER_PATH)), 'lib')
    env['LD_LIBRARY_PATH'] = blender_lib_dir + ':' + env.get('LD_LIBRARY_PATH', '')
    ret = call(args, env=env)
    if ret != 0 or not os.path.exists(output_pickle):
        error_file = output_pickle + '_error.txt'
        if os.path.exists(error_file):
            with open(error_file, 'r') as f:
                print("Blender error:", f.read())
        raise RuntimeError(f"Blender failed to extract PBR from {input_glb}")


def prepare_and_voxelize(pickle_path, resolution):
    """
    Phase 1: Load mesh from pickle, normalize, voxelize ONCE.
    Returns: coord [N,3], normals [N,3], vertices [V,3], faces [F,3] (int32)
    """
    with open(pickle_path, 'rb') as f:
        dump = pickle.load(f)

    # 修复 alphaMode
    for mat in dump['materials']:
        if mat.get('alphaTexture') is not None and mat.get('alphaMode') == 'OPAQUE':
            mat['alphaMode'] = 'BLEND'

    # 默认材质（用于 mat_id == -1 的面）
    default_mat = {
        "baseColorFactor": [0.8, 0.8, 0.8, 1.0],
        "alphaFactor": 1.0,
        "metallicFactor": 0.0,
        "roughnessFactor": 0.5,
        "alphaMode": "OPAQUE",
        "alphaCutoff": 0.5,
        "baseColorTexture": None,
        "alphaTexture": None,
        "metallicTexture": None,
        "roughnessTexture": None,
    }
    dump['materials'].append(default_mat)

    # 过滤空 mesh
    dump['objects'] = [
        o for o in dump['objects']
        if o['vertices'].size > 0 and o['faces'].size > 0
    ]
    if not dump['objects']:
        raise ValueError("No valid geometry found.")

    # 归一化到 [-0.5, 0.5]
    all_verts = np.concatenate([o['vertices'] for o in dump['objects']], axis=0)
    vt = torch.from_numpy(all_verts).float()
    vmin, vmax = vt.min(0).values, vt.max(0).values
    center = (vmin + vmax) / 2.0
    scale = 0.99999 / (vmax - vmin).max()

    for obj in dump['objects']:
        v = torch.from_numpy(obj['vertices']).float()
        obj['vertices'] = ((v - center) * scale).numpy()
        obj['mat_ids'] = np.where(obj['mat_ids'] == -1, len(dump['materials']) - 1, obj['mat_ids'])

    # 拼接 vertices 和 faces
    pre_max = 0
    vert_list, face_list = [], []
    for obj in dump['objects']:
        vert_list.append(torch.tensor(obj['vertices'], dtype=torch.float32))
        face_list.append(torch.tensor(obj['faces'] + pre_max, dtype=torch.int32))
        pre_max += obj['vertices'].shape[0]
    vertices = torch.cat(vert_list, dim=0)
    faces = torch.cat(face_list, dim=0)

    # 体素化（使用原始纹理/UV，只取 coord 和 normals）
    coord, attr = o_voxel.convert.blender_dump_to_volumetric_attr(
        dump,
        grid_size=resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        mip_level_offset=0,
        verbose=False,
        timing=False,
    )

    # 解码 normals
    normals_f = attr['normal'].float() / 255.0 * 2.0 - 1.0
    normals = F.normalize(normals_f, dim=1)

    print(f"✅ Voxelized: {coord.shape[0]} voxels at resolution {resolution}")
    return coord, normals, vertices, faces


@torch.no_grad()
def process_single_view(coord, normals, vertices_gpu, faces_gpu, resolution,
                        extr_t, intr_t, obs_np, rastctx, H=4096, W=4096,
                        displacement=None,
                        cumulative_flow=None, fb_map=None,
                        anchor_mvp=None, search_radius=2, depth_eps=2e-3):
    """
    Phase 2 per-view: rasterize depth → project voxels → visibility → sample image → confidence.

    Args:
        displacement: optional (Hd, Wd, 2) float tensor — cumulative per-pixel displacement
            field from flow-geometry calibration. When provided, the sampling position for
            each voxel is shifted by -D to correct for temporal-geometric drift
            (Deformable Texture Atlas).
        cumulative_flow: optional (Hf, Wf, 2) float tensor — cumulative flow chain from
            anchor frame. cumflow[y, x] = pixel coordinates in THIS frame where anchor
            pixel (y, x) has been tracked to.
        fb_map: optional (Hf, Wf) float tensor — forward-backward consistency error at
            tracked positions. Lower = more reliable.
        anchor_mvp: optional (4, 4) float tensor — MVP matrix of the anchor frame.
            Required when cumulative_flow is provided, used to project voxels into anchor
            frame to look up the flow chain.
        search_radius: int — neighborhood search radius in flow resolution pixels.

    Returns:
        confidence [N, 1] uint8 CPU
        rgb        [N, 3] uint8 CPU
    """
    device = vertices_gpu.device

    # 1. 光栅化 mesh → depth map
    view = utils3d.torch.extrinsics_to_view(extr_t[None]).squeeze(0)
    projection = utils3d.torch.intrinsics_to_perspective(intr_t[None], near=0.1, far=10.0).squeeze(0)
    mvp = projection @ view

    rast = utils3d.torch.rasterize_triangle_faces(
        rastctx, vertices_gpu[None], faces_gpu,
        width=W, height=H, view=view[None], projection=projection[None]
    )
    depth_rast = rast['depth'][0]        # [H, W]
    mask_rast = rast['mask'][0].bool()   # [H, W]

    # 2. 投影 voxel 中心 → NDC → 像素坐标
    coord_dev = coord.to(device)
    voxel_world = (coord_dev.float() + 0.5) / resolution - 0.5   # [N, 3]
    voxel_homo = torch.cat([voxel_world, torch.ones(voxel_world.shape[0], 1, device=device)], dim=1)
    voxel_clip = (mvp @ voxel_homo.T).T
    w_clip = voxel_clip[:, 3:4].clamp(min=1e-8)
    voxel_ndc = voxel_clip[:, :3] / w_clip

    voxel_u = (voxel_ndc[:, 0] + 1.0) * W * 0.5
    voxel_v = (1.0 - voxel_ndc[:, 1]) * H * 0.5   # Y 轴翻转
    voxel_depth = (voxel_ndc[:, 2] + 1.0) * 0.5    # NDC depth [0, 1]

    voxel_u_idx = torch.clamp(voxel_u.round().long(), 0, W - 1)
    voxel_v_idx = torch.clamp(voxel_v.round().long(), 0, H - 1)

    # 3. 可见性：深度测试
    surface_depth = depth_rast[voxel_v_idx, voxel_u_idx]
    has_surface = mask_rast[voxel_v_idx, voxel_u_idx]
    # depth_eps: passed as parameter（太小→俯视黑缝, 太大→穿透拿到背面色）
    visible = (has_surface
               & (voxel_depth >= surface_depth - depth_eps)
               & (voxel_depth <= surface_depth + depth_eps)
               & (voxel_depth > 0))

    # 4. 采样观测图像（支持三种模式）
    if cumulative_flow is not None and anchor_mvp is not None:
        # === 光流链引导 + 邻域搜索 ===
        # 思路：
        #   几何投影 (voxel_u, voxel_v) 是 GT 位置 → 确定哪个 voxel 接收颜色
        #   但视频帧中，这个 3D point 的纹理内容可能漂移了
        #   光流链追踪 anchor 帧中的对应像素到此帧 → 找到内容的实际语义位置
        #   在该位置附近邻域搜索，用 FB consistency 选最可靠像素来采色

        N = coord.to(device).shape[0]
        cf_gpu = cumulative_flow.to(device)     # (Hf, Wf, 2)
        Hf, Wf = cf_gpu.shape[:2]

        # A. 投影 voxel 到 anchor 帧 → anchor 帧像素坐标（光流分辨率）
        anchor_mvp_dev = anchor_mvp.to(device)
        anchor_clip = (anchor_mvp_dev @ voxel_homo.T).T        # (N, 4)
        anchor_w = anchor_clip[:, 3:4].clamp(min=1e-8)
        anchor_ndc = anchor_clip[:, :3] / anchor_w
        anchor_u = (anchor_ndc[:, 0] + 1.0) * Wf * 0.5        # (N,)
        anchor_v = (1.0 - anchor_ndc[:, 1]) * Hf * 0.5        # (N,)
        anchor_u_idx = torch.clamp(anchor_u.round().long(), 0, Wf - 1)
        anchor_v_idx = torch.clamp(anchor_v.round().long(), 0, Hf - 1)

        # B. 查累积光流 → 语义位置 (光流分辨率)
        sem_u = cf_gpu[anchor_v_idx, anchor_u_idx, 0]          # (N,)
        sem_v = cf_gpu[anchor_v_idx, anchor_u_idx, 1]          # (N,)

        # C. 上采样到光栅化分辨率
        scale_u = W / Wf
        scale_v = H / Hf
        sem_u_hi = sem_u * scale_u
        sem_v_hi = sem_v * scale_v

        # D. 准备观测图像和 FB map
        obs_gpu = torch.tensor(obs_np, dtype=torch.float32, device=device) / 255.0  # (H, W, 3)

        if fb_map is not None:
            fb_gpu = fb_map.to(device)
            # 上采样 FB map 到光栅化分辨率
            fb_hi = F.interpolate(
                fb_gpu.unsqueeze(0).unsqueeze(0), size=(H, W),
                mode='bilinear', align_corners=False
            ).squeeze()  # (H, W)
        else:
            fb_hi = None

        # E. 邻域搜索
        r_hi = max(1, int(round(search_radius * scale_u)))
        best_color = torch.zeros(N, 3, device=device)
        best_fb = torch.full((N,), float('inf'), device=device)

        for dy in range(-r_hi, r_hi + 1):
            for dx in range(-r_hi, r_hi + 1):
                cu = torch.clamp((sem_u_hi + dx).round().long(), 0, W - 1)
                cv = torch.clamp((sem_v_hi + dy).round().long(), 0, H - 1)
                color = obs_gpu[cv, cu]        # (N, 3)

                if fb_hi is not None:
                    fb_score = fb_hi[cv, cu]    # (N,)
                else:
                    # Fallback: prefer center
                    fb_score = ((dx ** 2 + dy ** 2) ** 0.5) * torch.ones(N, device=device)

                better = (fb_score < best_fb) & visible
                best_color[better] = color[better]
                best_fb[better] = fb_score[better]

        rgb = (best_color * 255).clamp(0, 255).to(torch.uint8)  # [N, 3]

    elif displacement is not None:
        # Upscale displacement from flow resolution to rasterization resolution
        disp_gpu = displacement.to(device)
        Hd, Wd = disp_gpu.shape[:2]
        if Hd != H or Wd != W:
            disp_gpu = F.interpolate(
                disp_gpu.permute(2, 0, 1).unsqueeze(0),
                size=(H, W), mode='bilinear', align_corners=False
            ).squeeze(0).permute(1, 2, 0)
            # Scale displacement values proportionally
            disp_gpu[..., 0] *= W / Wd
            disp_gpu[..., 1] *= H / Hd

        # Lookup displacement at each voxel's projected position
        disp_at_voxel = disp_gpu[voxel_v_idx, voxel_u_idx]  # [N, 2]

        # Corrected sampling: projected_pos - D (undo the drift)
        sample_u = voxel_u - disp_at_voxel[:, 0]
        sample_v = voxel_v - disp_at_voxel[:, 1]

        # Normalize to [-1, 1] for grid_sample
        grid_x = (sample_u / (W - 1)) * 2 - 1
        grid_y = (sample_v / (H - 1)) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, -1, 2)  # [1, 1, N, 2]

        obs_gpu = torch.tensor(obs_np, dtype=torch.float32, device=device) / 255.0
        obs_perm = obs_gpu.permute(2, 0, 1).unsqueeze(0)   # [1, 3, H, W]
        sampled = F.grid_sample(
            obs_perm, grid, mode='bilinear', padding_mode='zeros', align_corners=True
        )   # [1, 3, 1, N]
        rgb = (sampled[0, :, 0, :].T * 255).clamp(0, 255).to(torch.uint8)  # [N, 3]
    else:
        # Original nearest-neighbor sampling (no displacement)
        obs_gpu = torch.tensor(obs_np, dtype=torch.uint8, device=device)   # [H, W, 3]
        rgb = obs_gpu[voxel_v_idx, voxel_u_idx]   # [N, 3]
    rgb[~visible] = 0

    # 5. 计算 view_cos 置信度
    normals_dev = normals.to(device).float()
    R, t_vec = extr_t[:3, :3], extr_t[:3, 3]
    C = -R.T @ t_vec                                       # 相机中心（世界坐标）
    view_dirs = F.normalize(C.unsqueeze(0) - voxel_world, dim=1)
    cos_theta = (normals_dev * view_dirs).sum(dim=1, keepdim=True)
    view_cos = torch.clamp(cos_theta + 1, min=0.0, max=2.0)
    confidence = (view_cos * 255.0 / 2).clamp(0, 255).round().to(torch.uint8)
    confidence[~visible] = 0

    return confidence.cpu(), rgb.cpu()


# ====================================================================
# Phase 3  helpers: Graph-Cut (MRF) view selection
# ====================================================================

def build_voxel_adjacency(coord, resolution):
    """
    Build a 6-connected adjacency graph over occupied voxels.
    Fully vectorized for speed.
    
    Args:
        coord: [N, 3] int tensor  (voxel integer coordinates)
        resolution: int
    Returns:
        edges: np.ndarray [E, 2] int32, each row (i, j) with i < j
    """
    N = coord.shape[0]
    coord_np = coord.numpy().astype(np.int64)
    
    # Spatial hash: pack (x,y,z) into a single int64
    R = np.int64(resolution + 2)
    keys = coord_np[:, 0] * (R * R) + coord_np[:, 1] * R + coord_np[:, 2]
    key_to_idx = dict(zip(keys.tolist(), range(N)))
    
    # Vectorized: for each offset, compute all neighbor keys at once
    offsets_key = np.array([1, R, R * R], dtype=np.int64)
    all_edges = []
    for dk in offsets_key:
        neighbor_keys = keys + dk
        # Vectorized lookup: check which neighbors exist
        found = np.array([key_to_idx.get(k, -1) for k in neighbor_keys.tolist()], dtype=np.int32)
        valid = found >= 0
        src = np.arange(N, dtype=np.int32)[valid]
        dst = found[valid]
        # Stack and ensure i < j
        pairs = np.stack([np.minimum(src, dst), np.maximum(src, dst)], axis=1)
        all_edges.append(pairs)
    
    if all_edges:
        edges = np.concatenate(all_edges, axis=0)
    else:
        edges = np.empty((0, 2), dtype=np.int32)
    return edges


def graphcut_view_selection(coord, resolution, all_rgb, all_conf,
                           lambda_smooth=5.0, max_views=10, gc_iter=-1):
    """
    Per-voxel top-K → MRF Graph-Cut view selection.

    Each voxel keeps its K (=max_views) highest-confidence views as feasible
    candidates.  This guarantees every visible voxel retains its best view —
    no coverage loss, no fallback needed.

    Union of all per-voxel candidates forms the global label set for
    alpha-expansion.  For each label expansion, voxels that have INF cost for
    that label trivially keep their current assignment → fast convergence.

    Energy:  E = Σ_i D_i(l_i)  +  λ · Σ_{(i,j)∈E} w_ij · [l_i ≠ l_j]

    Args:
        max_views: per-voxel top-K (each voxel keeps its K best views)
    """
    N, V = all_conf.shape
    K = min(max_views, V)
    print(f"\n🔧 Graph-Cut MRF  (N={N:,}, V={V}, per_voxel_K={K}, λ={lambda_smooth})")

    conf_u8 = all_conf.numpy()   # [N, V] uint8  (zero-copy from torch)

    # --- 0. Greedy pre-solve (from ALL V views) ---
    t0 = time.time()
    greedy_best = conf_u8.argmax(axis=1)   # [N]
    print(f"  Greedy pre-solve  ({time.time()-t0:.1f}s)")

    # --- 1. Per-voxel top-K by confidence ---
    t0 = time.time()
    if K >= V:
        topk_idx = np.tile(np.arange(V, dtype=np.int32), (N, 1))   # [N, V]
    else:
        neg_conf = -(conf_u8.astype(np.int16))          # int16 saves memory vs float64
        topk_idx = np.argpartition(neg_conf, K, axis=1)[:, :K].astype(np.int32)  # [N, K]
        del neg_conf

    # Union of all per-voxel top-K → global candidate label set
    candidate_views = np.unique(topk_idx.ravel()).astype(np.int32)   # sorted
    V_cand = len(candidate_views)
    print(f"  Per-voxel top-{K}: union → {V_cand}/{V} labels  ({time.time()-t0:.1f}s)")

    # --- 2. Index mapping: original view → candidate label ---
    orig_to_cand = np.full(V, -1, dtype=np.int32)
    for ci, oi in enumerate(candidate_views):
        orig_to_cand[oi] = ci

    greedy_cand = orig_to_cand[greedy_best]   # [N], in [0..V_cand-1]
    assert (greedy_cand >= 0).all(), "BUG: greedy best not in its own top-K"

    # --- 3. Unary cost [N, V_cand] ---
    #   INF_COST for views NOT in per-voxel top-K or with conf=0.
    #   Real cost = 2.0 - conf/255 for feasible visible views.
    t0 = time.time()
    INF_COST = 10.0
    unary = np.full((N, V_cand), INF_COST, dtype=np.float64)

    # Set real cost only for each voxel's top-K entries
    rows = np.repeat(np.arange(N, dtype=np.int64), K)
    orig_cols = topk_idx.ravel().astype(np.int64)
    cand_cols = orig_to_cand[orig_cols].astype(np.int64)
    conf_vals = conf_u8[rows, orig_cols].astype(np.float64)
    real_cost = np.where(conf_vals > 0, 2.0 - conf_vals / 255.0, INF_COST)
    unary[rows, cand_cols] = real_cost

    n_feasible = (unary < INF_COST).sum()
    print(f"  Unary: [{N:,} × {V_cand}], feasible {n_feasible:,}/{N*V_cand:,} "
          f"({100*n_feasible/(N*V_cand):.1f}%)  ({time.time()-t0:.1f}s)")

    # --- 4. Adjacency ---
    t0 = time.time()
    edges = build_voxel_adjacency(coord, resolution)
    E = edges.shape[0]
    print(f"  Adjacency: {E:,} edges  ({time.time()-t0:.1f}s)")

    # --- 5. Edge weights (color gradient from greedy assignment) ---
    t0 = time.time()
    greedy_rgb = all_rgb[np.arange(N), greedy_best].numpy().astype(np.float64)  # [N, 3]

    if E > 0:
        c_i = greedy_rgb[edges[:, 0]]
        c_j = greedy_rgb[edges[:, 1]]
        color_diff = np.linalg.norm(c_i - c_j, axis=1)
        beta = 1.0 / (2.0 * max(color_diff.mean(), 1e-3))
        edge_weights = (lambda_smooth / (1.0 + beta * color_diff)).astype(np.float64)
    else:
        edge_weights = np.empty(0, dtype=np.float64)
    print(f"  Edge weights  ({time.time()-t0:.1f}s)")

    # --- 6. Pairwise (Potts) ---
    pairwise = (1.0 - np.eye(V_cand)).astype(np.float64)

    # --- 7. Alpha-expansion ---
    t0 = time.time()
    print(f"  Alpha-expansion ({V_cand} labels, {N:,} nodes, {E:,} edges, iter={gc_iter}) ...")
    labels_cand = gco.cut_general_graph(
        edges, edge_weights, unary, pairwise,
        n_iter=gc_iter, algorithm='expansion',
        init_labels=greedy_cand.astype(np.int32),
    )
    print(f"  Graph-Cut solved  ({time.time()-t0:.1f}s)")

    # --- 8. Map back to original view indices ---
    labels = candidate_views[labels_cand]
    changed = (labels != greedy_best).sum()
    print(f"  Changed from greedy: {changed:,}/{N:,} ({100*changed/N:.1f}%)")

    return labels

# /path/to/dataset/.../model.glb
def main():
    parser = argparse.ArgumentParser(
        description="Convert .glb to PBR .vxz using Graph-Cut optimized view selection")
    parser.add_argument("input_glb", help="Input .glb file path")
    parser.add_argument("--image", default=None, help="Image directory (with h/ and v/ subdirs)")
    parser.add_argument("--video", default=None, help="Input video file (替代 --image，自动拆帧取 h 视角)")
    parser.add_argument("--output_vxz", help="Output .vxz file path (default: auto)")
    parser.add_argument("--output_pickle", help="Output .pickle file path (default: auto)")
    parser.add_argument("--resolution", type=int, default=1024, help="Voxel grid resolution")
    parser.add_argument("--lambda_smooth", type=float, default=5.0,
                        help="Graph-Cut smoothness weight (higher = fewer seams, default=5)")
    parser.add_argument("--max_views", type=int, default=30,
                        help="Per-voxel top-K: each voxel keeps its K best views (default=10)")
    parser.add_argument("--gc_iter", type=int, default=-1,
                        help="Alpha-expansion max iterations (-1=until convergence, default=-1)")
    parser.add_argument("--mode", choices=['graphcut', 'greedy'], default='greedy',
                        help="'graphcut': MRF optimization (default), 'greedy': simple top-1 per voxel")
    parser.add_argument("--pickle_in", default=None,
                        help="Pre-existing PBR pickle (skip Blender extraction)")
    parser.add_argument("--frames", default=None,
                        help="Comma-separated frame indices to use, e.g. '0,30,60,90' (default: all)")
    parser.add_argument("--displacement_dir", default=None,
                        help="Directory with per-frame displacement .npy files (D_XXX.npy) "
                             "from generate_uv.py --save_displacement. Enables deformable "
                             "texture atlas: each voxel samples at projected_pos - D.")
    parser.add_argument("--max_lock_angle", type=float, default=60.0,
                        help="Max view-normal angle (degrees) for first-frame locking. "
                             "Voxels with angle > this are not locked by the first frame (default: 60)")
    parser.add_argument("--flow_chain_dir", default=None,
                        help="Directory with cumulative flow chain data: cumflow_XXX.npy and "
                             "fb_XXX.npy from precompute_flow_chain.py. Enables flow-guided "
                             "semantic sampling: each voxel's color is sampled at the position "
                             "tracked by the optical flow chain from the anchor frame, with "
                             "neighborhood search for robustness.")
    parser.add_argument("--search_radius", type=int, default=2,
                        help="Neighborhood search radius in flow-resolution pixels (default: 2). "
                             "Only used with --flow_chain_dir.")
    parser.add_argument("--video_num_cols", type=int, default=4,
                        help="Number of columns in the H video (default: 4 for 4-col, use 8 for hv format)")
    parser.add_argument("--video_col", type=int, default=2,
                        help="Which column (0-indexed) in the H video to use (default: 2 = 3rd column of 4-col)")
    parser.add_argument("--video_v", default=None,
                        help="Vertical-track video (8-col concatenated). "
                             "Uncovered voxels after H-track will be filled from V-track.")
    parser.add_argument("--video_v_col", type=int, default=5,
                        help="Which column (0-indexed) in the V video to use (default: 5 = 6th column)")
    parser.add_argument("--video_v_num_cols", type=int, default=8,
                        help="Number of columns in the V video (default: 8 for hv format)")
    parser.add_argument("--camera_mode", choices=['h', 'v'], default='h',
                        help="Which camera to use for --video: h (horizontal) or v (vertical)")
    parser.add_argument("--priority_mode", action='store_true',
                        help="H priority: color all with H first, then fill gaps with V, then BFS")
    parser.add_argument("--video_frame_indices", default=None,
                        help="Comma-separated camera indices mapping to video frames (e.g. '30,90')")
    parser.add_argument("--video_v_frame_indices", default=None,
                        help="Comma-separated camera indices for V video (same format as --video_frame_indices)")
    parser.add_argument("--no_bfs", action='store_true',
                        help="Skip BFS gap-fill (show only view-colored voxels)")
    parser.add_argument("--depth_eps", type=float, default=2e-3,
                        help="NDC depth tolerance (smaller=less bleeding, larger=less cracks, default=2e-3)")
    parser.add_argument("--first_frame", default=None,
                        help="Override first frame for locking, e.g. 'v-030'")
    parser.add_argument("--flip_v", default=None,
                        help="Comma-separated V camera indices to flip right axis, e.g. '28,30,32'")
    args = parser.parse_args()

    input_glb = args.input_glb

    if not args.image and not args.video:
        parser.error("必须指定 --image 或 --video 之一")
    if args.image and args.video:
        parser.error("--image 和 --video 不能同时指定")

    if not os.path.exists(input_glb):
        raise FileNotFoundError(f"Input file not found: {input_glb}")

    # 如果使用 --video，提取帧到内存
    video_frames = None
    if args.video:
        if not os.path.exists(args.video):
            raise FileNotFoundError(f"Video not found: {args.video}")
        video_frames = extract_frames_from_video(args.video, num_cols=args.video_num_cols, col_index=args.video_col)
        token = os.path.splitext(os.path.basename(args.video))[0]
    else:
        token = args.image.split('/')[-1]
    base = os.path.splitext(input_glb)[0]
    output_vxz = args.output_vxz or base + f'_{token}.vxz'
    output_pickle = args.output_pickle or os.path.splitext(output_vxz)[0] + '.pickle'

    os.makedirs(os.path.dirname(output_vxz) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(output_pickle) or '.', exist_ok=True)
    # import pdb;pdb.set_trace()
    # --- 加载相机参数 ---
    save_path = os.path.join(SCRIPT_DIR, "parameters_hv.pt")
    print('start')
    data = torch.load(save_path, weights_only=False)
    def _to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, list):
            return torch.stack([_to_tensor(i) for i in x])
        return torch.tensor(x)
    h_extr = _to_tensor(data['h_extrinsics_filtered'])                     # [61, 4, 4]
    h_intr = _to_tensor(data['h_intrinsics_normalized_filtered'])          # [61, 3, 3]
    v_extr = _to_tensor(data['v_extrinsics_filtered'])                     # [61, 4, 4]
    v_intr = _to_tensor(data['v_intrinsics_normalized_filtered'])           # [61, 3, 3]

    if args.flip_v:
        # Flip only the cameras in the specified comma-separated list of V indices
        flip_v_indices = [int(x.strip()) for x in args.flip_v.split(",") if x.strip()]
        for fi in flip_v_indices:
            v_extr[fi, :3, 0] = -v_extr[fi, :3, 0]
        print(f"🔧 Flipped V cameras: {flip_v_indices}")

    num_h, num_v = h_extr.shape[0], v_extr.shape[0]

    # --- Step 1: Blender 提取 PBR (or reuse existing pickle) ---
    if args.pickle_in and os.path.exists(args.pickle_in):
        print(f"♻️  Reusing existing pickle: {args.pickle_in}")
        pickle_path = args.pickle_in
        # --- Phase 1: 体素化一次 ---
        coord, normals, vertices, faces = prepare_and_voxelize(
            pickle_path, args.resolution
        )
    else:
        install_blender()
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_pickle_path = os.path.join(tmp_dir, 'pbr_dump.pickle')
            extract_pbr_with_blender(input_glb, temp_pickle_path)
            shutil.copy(temp_pickle_path, output_pickle)
            print(f"✅ Saved PBR dump to: {output_pickle}")

            # --- Phase 1: 体素化一次 ---
            coord, normals, vertices, faces = prepare_and_voxelize(
                temp_pickle_path, args.resolution
            )

    N = coord.shape[0]
    H, W = 4096, 4096

    # --- 动态判断视频帧数与相机参数的对应关系 ---
    if args.camera_mode == 'v':
        cam_count = num_v
        cam_label = 'v'
    else:
        cam_count = num_h
        cam_label = 'h'

    stride = 1
    if video_frames is not None:
        n_video = len(video_frames)
        # Explicit frame→camera mapping overrides stride detection
        if args.video_frame_indices:
            cam_indices = [int(x) for x in args.video_frame_indices.split(",")]
            remapped = {}
            for vf_idx, cam_idx in enumerate(cam_indices):
                if vf_idx < n_video and cam_idx < cam_count:
                    remapped[cam_idx] = video_frames[vf_idx]
            video_frames = remapped
            print(f"📹 {cam_label}-track: {n_video}帧→相机[{args.video_frame_indices}], {len(video_frames)} views")
        elif n_video < cam_count and abs(n_video * 2 - cam_count) <= 2:
            stride = 2
            remapped = {}
            for vf_idx, img in video_frames.items():
                cam_idx = vf_idx * 2
                if cam_idx < cam_count:
                    remapped[cam_idx] = img
            video_frames = remapped
            print(f"📹 {cam_label}-track: 视频{n_video}帧 vs 相机{cam_count}组, stride=2, 映射后{len(video_frames)}帧")
        else:
            print(f"📹 {cam_label}-track: 视频{n_video}帧 vs 相机{cam_count}组, stride=1 (1:1)")

    all_views = [(cam_label, i) for i in range(0, cam_count, stride) if video_frames is None or i in video_frames]

    # V-track: 加入 V 视角，与 H 平等参与 view selection
    video_frames_v = None
    video_frames_v_simple = None
    if args.video_v:
        video_frames_v = extract_frames_from_video_v(
            args.video_v, col_index=args.video_v_col, num_cols=args.video_v_num_cols)
        # V-track: auto-detect stride or use explicit frame indices
        n_v_video = len(video_frames_v)
        if args.video_v_frame_indices:
            v_cam_indices = [int(x) for x in args.video_v_frame_indices.split(",")]
            video_frames_v_simple = {}
            v_view_list = []
            for cam_idx in v_cam_indices:
                if cam_idx < num_v and cam_idx in video_frames_v:
                    video_frames_v_simple[cam_idx] = video_frames_v[cam_idx]
                    v_view_list.append(('v', cam_idx, cam_idx))
        else:
            n_v_video_adj = n_v_video - 1  # exclude last duplicate frame
            v_stride = 1
            if n_v_video_adj < num_v and abs(n_v_video_adj * 2 - num_v) <= 2:
                v_stride = 2
            v_view_list = []
            for vf in range(n_v_video_adj):
                cam_idx = vf * v_stride
                if cam_idx < num_v and vf in video_frames_v:
                    v_view_list.append(('v', cam_idx, vf))  # (mode, cam_idx, video_frame_idx)
            video_frames_v_simple = None
        # Add V views to all_views using ('v', cam_idx) format
        for _, cam_idx, _ in v_view_list:
            all_views.append(('v', cam_idx))
        # Build mapping: ('v', cam_idx) → video_frame_idx
        v_cam_to_vf = {cam_idx: vf for _, cam_idx, vf in v_view_list}
        print(f"📹 V-track: {len(v_view_list)} views added, total views: {len(all_views)}")

    # Override first frame for locking
    if args.first_frame:
        parts = args.first_frame.split('-')
        ff_mode, ff_idx = parts[0], int(parts[1])
        for i, (m, idx) in enumerate(all_views):
            if m == ff_mode and idx == ff_idx:
                all_views.insert(0, all_views.pop(i))
                print(f"🔒 First frame overridden to: {args.first_frame}")
                break

    # 如果指定了 --frames，只保留指定帧
    if args.frames is not None:
        selected = set(int(x.strip()) for x in args.frames.split(','))
        all_views = [(m, i) for m, i in all_views if i in selected]
        print(f"🎯 Using {len(all_views)} selected frames: {sorted(selected)}")

    V = len(all_views)

    # 持久化 GPU 数据（所有 view 共享）
    vertices_gpu = vertices.cuda()
    faces_gpu = faces.cuda()
    rastctx = utils3d.torch.RastContext(backend='cuda')

    # ============================================================
    # Phase 2a: 处理第一帧 → 无条件信任，锁定可见 voxel
    # ============================================================
    best_rgb  = torch.zeros(N, 3, dtype=torch.uint8)
    best_conf = torch.zeros(N,    dtype=torch.uint8)

    first_mode, first_idx = all_views[0]
    extr_first = (v_extr if first_mode == 'v' else h_extr)[first_idx]
    intr_first = (v_intr if first_mode == 'v' else h_intr)[first_idx]

    # 加载第一帧图像
    if first_mode == 'v' and video_frames_v_simple is not None:
        if first_idx not in video_frames_v_simple:
            raise RuntimeError(f"First frame v-{first_idx} not found in V video!")
        frame_rgb = video_frames_v_simple[first_idx]
        obs_first = np.array(Image.fromarray(frame_rgb).resize((H, W), Image.BILINEAR))
    elif first_mode == 'v' and video_frames_v is not None:
        if first_idx not in video_frames_v:
            raise RuntimeError(f"First frame v-{first_idx} not found in V video!")
        frame_rgb = video_frames_v[first_idx]
        obs_first = np.array(Image.fromarray(frame_rgb).resize((H, W), Image.BILINEAR))
    elif video_frames is not None:
        if first_idx not in video_frames:
            raise RuntimeError(f"First frame {first_idx} not found in video!")
        frame_rgb = video_frames[first_idx]
        obs_first = np.array(Image.fromarray(frame_rgb).resize((H, W), Image.BILINEAR))
    else:
        image_path = f"{args.image}/{first_mode}/{first_idx:03d}.png"
        if not os.path.exists(image_path):
            raise RuntimeError(f"First frame image not found: {image_path}")
        obs_first = np.array(Image.open(image_path).resize((H, W), Image.BILINEAR).convert('RGB'))

    extr_t = extr_first.float().cuda()
    intr_t = intr_first.float().cuda()

    disp = None
    if args.displacement_dir and first_mode == 'h':
        disp_path = os.path.join(args.displacement_dir, f"D_{first_idx:03d}.npy")
        if os.path.exists(disp_path):
            disp = torch.from_numpy(np.load(disp_path)).float()

    conf_first, rgb_first = process_single_view(
        coord, normals, vertices_gpu, faces_gpu,
        args.resolution, extr_t, intr_t, obs_first, rastctx, H, W,
        displacement=disp, depth_eps=args.depth_eps
    )
    conf_first = conf_first.squeeze(-1)  # [N]

    # 计算第一帧 view-normal 夹角，过滤掉角度过大的 voxel
    max_lock_angle = args.max_lock_angle  # 默认 70 度
    cos_threshold = np.cos(np.deg2rad(max_lock_angle))  # cos(70°) ≈ 0.342
    # 重新计算 view direction 和 normal 的夹角
    coord_world = (coord.float() + 0.5) / args.resolution - 0.5  # [N, 3]
    R_cam, t_cam = extr_t[:3, :3].cpu(), extr_t[:3, 3].cpu()
    cam_center = -R_cam.T @ t_cam  # 相机中心（世界坐标）
    view_dirs = F.normalize(cam_center.unsqueeze(0) - coord_world, dim=1)  # [N, 3]
    cos_angle = (normals.float() * view_dirs).sum(dim=1)  # [N]
    angle_ok = cos_angle >= cos_threshold  # True = 角度 <= max_lock_angle

    # 锁定第一帧可见且角度合适的 voxel
    locked = (conf_first > 0) & angle_ok
    best_rgb[locked]  = rgb_first[locked]
    best_conf[locked] = conf_first[locked]
    n_locked = locked.sum().item()
    n_visible = (conf_first > 0).sum().item()
    n_angle_filtered = n_visible - n_locked
    print(f"\n🔒 First frame ({first_mode}-{first_idx:03d}): "
          f"visible {n_visible:,}, angle filtered {n_angle_filtered:,} (>{max_lock_angle}°), "
          f"locked {n_locked:,}/{N:,} voxels ({100*n_locked/N:.1f}%)")

    # Precompute anchor MVP for flow-chain guided sampling
    anchor_mvp = None
    if args.flow_chain_dir:
        anchor_view = utils3d.torch.extrinsics_to_view(extr_first.float().cuda()[None]).squeeze(0)
        anchor_proj = utils3d.torch.intrinsics_to_perspective(
            intr_first.float().cuda()[None], near=0.1, far=10.0).squeeze(0)
        anchor_mvp = anchor_proj @ anchor_view  # (4, 4), on cuda
        print(f"  Anchor MVP precomputed for flow-chain sampling")

    # ============================================================
    # Phase 2b: 处理剩余帧 → 只对未锁定的 voxel 收集数据
    # ============================================================
    if n_locked == 0:
        remaining_views = all_views  # first frame contributed nothing, re-include it
    else:
        remaining_views = all_views[1:]  # 跳过第一帧
    V_rest = len(remaining_views)

    if V_rest > 0 and n_locked < N:
        # 未锁定 voxel 的索引
        unlocked_mask = ~locked
        unlocked_idx = torch.where(unlocked_mask)[0]  # 全局索引
        N_unlocked = unlocked_idx.shape[0]
        print(f"\n🔄 Processing {V_rest} remaining views for {N_unlocked:,} unlocked voxels...")

        # 未锁定 voxel 的 coord 和 normals
        coord_unlocked = coord[unlocked_idx]
        normals_unlocked = normals[unlocked_idx]

        # 存储未锁定 voxel 的结果
        rest_rgb  = torch.zeros(N_unlocked, V_rest, 3, dtype=torch.uint8)
        rest_conf = torch.zeros(N_unlocked, V_rest,    dtype=torch.uint8)
        print(f"📦 Allocated rest-view storage: "
              f"rgb {rest_rgb.nbytes/1e9:.2f} GB, conf {rest_conf.nbytes/1e9:.2f} GB")

        for view_i, (mode, idx) in enumerate(remaining_views):
            extr = (v_extr if mode == 'v' else h_extr)[idx]
            intr = (v_intr if mode == 'v' else h_intr)[idx]

            if mode == 'v':
                # Use simplified mapping if available, else fall back to original
                if video_frames_v_simple is not None:
                    if idx not in video_frames_v_simple:
                        continue
                    frame_rgb = video_frames_v_simple[idx]
                elif video_frames_v is not None:
                    vf_idx = v_cam_to_vf.get(idx)
                    if vf_idx is None or vf_idx not in video_frames_v:
                        continue
                    frame_rgb = video_frames_v[vf_idx]
                else:
                    continue
                obs = np.array(Image.fromarray(frame_rgb).resize((H, W), Image.BILINEAR))
            elif video_frames is not None:
                if idx not in video_frames:
                    continue
                frame_rgb = video_frames[idx]
                obs = np.array(Image.fromarray(frame_rgb).resize((H, W), Image.BILINEAR))
            else:
                image_path = f"{args.image}/{mode}/{idx:03d}.png"
                if not os.path.exists(image_path):
                    continue
                obs = np.array(Image.open(image_path).resize((H, W), Image.BILINEAR).convert('RGB'))

            extr_t = extr.float().cuda()
            intr_t = intr.float().cuda()

            disp = None
            if args.displacement_dir and mode == 'h':
                disp_path = os.path.join(args.displacement_dir, f"D_{idx:03d}.npy")
                if os.path.exists(disp_path):
                    disp = torch.from_numpy(np.load(disp_path)).float()

            # 加载光流链数据
            cumflow = None
            fb = None
            if args.flow_chain_dir and mode == 'h':
                cf_path = os.path.join(args.flow_chain_dir, f"cumflow_{idx:03d}.npy")
                fb_path = os.path.join(args.flow_chain_dir, f"fb_{idx:03d}.npy")
                if os.path.exists(cf_path):
                    cumflow = torch.from_numpy(np.load(cf_path)).float()
                if os.path.exists(fb_path):
                    fb = torch.from_numpy(np.load(fb_path)).float()

            # 只对未锁定 voxel 做投影
            confidence, rgb = process_single_view(
                coord_unlocked, normals_unlocked, vertices_gpu, faces_gpu,
                args.resolution, extr_t, intr_t, obs, rastctx, H, W,
                displacement=disp,
                cumulative_flow=cumflow, fb_map=fb,
                anchor_mvp=anchor_mvp, depth_eps=args.depth_eps,
                search_radius=args.search_radius,
            )

            rest_conf[:, view_i] = confidence.squeeze(-1)
            rest_rgb[:, view_i]  = rgb

            if (view_i + 1) % 10 == 0 or view_i == V_rest - 1:
                vis_this = (confidence > 0).sum().item()
                covered_rest = (rest_conf.max(dim=1).values > 0).sum().item()
                print(f"  [{view_i+1:3d}/{V_rest}] {mode}-{idx:03d} | "
                      f"this_view: {vis_this:,} visible | "
                      f"rest_covered: {covered_rest:,}/{N_unlocked:,} ({100*covered_rest/N_unlocked:.1f}%)")

        # --- Phase 3: 对未锁定 voxel 做 view selection ---
        if args.priority_mode:
            # Split remaining views into H and V
            h_indices = [i for i, (mode, _) in enumerate(remaining_views) if mode == 'h']
            v_indices = [i for i, (mode, _) in enumerate(remaining_views) if mode == 'v']
            print(f"\n🔧 Priority mode: H-first ({len(h_indices)} views) → V-fill ({len(v_indices)} views)")

            labels_rest = np.full(N_unlocked, -1, dtype=np.int32)
            rest_best_rgb = torch.zeros(N_unlocked, 3, dtype=torch.uint8)
            rest_best_conf = torch.zeros(N_unlocked, dtype=torch.uint8)

            # Step 1: H priority - pick best H for each voxel
            if h_indices:
                h_rgb = rest_rgb[:, h_indices]
                h_conf = rest_conf[:, h_indices]
                h_labels_np = h_conf.numpy().argmax(axis=1)
                h_labels_t = torch.from_numpy(h_labels_np).long()
                h_max_conf_np = h_conf[torch.arange(N_unlocked), h_labels_t].numpy()
                h_covered = h_max_conf_np > 0
                h_covered_idx = np.where(h_covered)[0]
                h_best = h_labels_np[h_covered_idx].astype(int)
                labels_rest[h_covered_idx] = np.array(h_indices)[h_best]
                rest_best_rgb[h_covered_idx] = h_rgb[torch.from_numpy(h_covered_idx).long(), torch.from_numpy(h_best).long()]
                rest_best_conf[h_covered_idx] = torch.from_numpy(h_max_conf_np[h_covered_idx]).byte()
                print(f"  H covered: {len(h_covered_idx):,}/{N_unlocked:,} ({100*len(h_covered_idx)/N_unlocked:.1f}%)")

            # Step 2: V fills remaining
            v_uncovered = np.where(~h_covered)[0]
            if v_indices and len(v_uncovered) > 0:
                v_rgb = rest_rgb[v_uncovered][:, v_indices]
                v_conf = rest_conf[v_uncovered][:, v_indices]
                v_labels_np = v_conf.numpy().argmax(axis=1)
                v_labels_t = torch.from_numpy(v_labels_np).long()
                v_max_conf_np = v_conf[torch.arange(len(v_uncovered)), v_labels_t].numpy()
                v_covered_mask = v_max_conf_np > 0
                v_covered_local = np.where(v_covered_mask)[0]
                v_global = v_uncovered[v_covered_local]
                v_best = v_labels_np[v_covered_local].astype(int)
                labels_rest[v_global] = np.array(v_indices)[v_best]
                rest_best_rgb[v_global] = v_rgb[torch.from_numpy(v_covered_local).long(), torch.from_numpy(v_best).long()]
                rest_best_conf[v_global] = torch.from_numpy(v_max_conf_np[v_covered_local]).byte()
                print(f"  V covered: {len(v_covered_local):,}/{len(v_uncovered):,} remaining ({100*len(v_covered_local)/len(v_uncovered):.1f}%)")

            uncovered_count = (rest_best_conf == 0).sum().item()
            print(f"  Total covered: {N_unlocked - uncovered_count:,}/{N_unlocked:,} ({100*(N_unlocked-uncovered_count)/N_unlocked:.1f}%)")

        elif args.mode == 'graphcut':
            labels_rest = graphcut_view_selection(
                coord_unlocked, args.resolution, rest_rgb, rest_conf,
                lambda_smooth=args.lambda_smooth,
                max_views=args.max_views,
                gc_iter=args.gc_iter,
            )
        else:
            print(f"\n🔧 Greedy top-1 mode for unlocked voxels")
            labels_rest = rest_conf.numpy().argmax(axis=1)

        # 填充未锁定 voxel 的结果
        labels_rest_t = torch.from_numpy(labels_rest).long()
        rest_best_rgb  = rest_rgb[torch.arange(N_unlocked), labels_rest_t]
        rest_best_conf = rest_conf[torch.arange(N_unlocked), labels_rest_t]

        # Fallback: 如果 GC/greedy 选的 conf=0，回退到全局 greedy
        rest_uncovered = (rest_best_conf == 0)
        if rest_uncovered.any():
            greedy_all = rest_conf.numpy().argmax(axis=1)
            greedy_all_t = torch.from_numpy(greedy_all).long()
            greedy_rgb = rest_rgb[torch.arange(N_unlocked), greedy_all_t]
            greedy_conf = rest_conf[torch.arange(N_unlocked), greedy_all_t]
            patch = rest_uncovered & (greedy_conf > 0)
            rest_best_rgb[patch]  = greedy_rgb[patch]
            rest_best_conf[patch] = greedy_conf[patch]
            print(f"  Fallback: patched {patch.sum().item():,} unlocked voxels from greedy")

        # 写回到全局 best_rgb / best_conf
        best_rgb[unlocked_idx]  = rest_best_rgb
        best_conf[unlocked_idx] = rest_best_conf

        del rest_rgb, rest_conf
    else:
        if n_locked == N:
            print(f"\n✅ All voxels covered by first frame, no further processing needed")
        else:
            print(f"\n⚠️  No remaining views to process")

    # 释放 GPU 资源
    del vertices_gpu, faces_gpu, rastctx
    torch.cuda.empty_cache()

    # --- 统计 ---
    uncovered = (best_conf == 0).sum().item()
    print(f"\n📊 Before gap-fill: {N - uncovered:,}/{N:,} voxels covered "
          f"({100*(N-uncovered)/N:.1f}%), {uncovered:,} uncovered "
          f"(locked by frame0: {n_locked:,})")

    # --- Gap-fill: BFS 从已覆盖 voxel 向未覆盖邻居扩散颜色 ---
    if uncovered > 0 and not args.no_bfs:
        t0 = time.time()
        coord_np = coord.numpy().astype(np.int64)
        R = np.int64(args.resolution + 2)
        keys = coord_np[:, 0] * (R * R) + coord_np[:, 1] * R + coord_np[:, 2]
        key_to_idx = dict(zip(keys.tolist(), range(N)))
        offsets_key = [1, -1, R, -R, R * R, -(R * R)]  # 6-connected

        # Build per-voxel neighbor list
        neighbors = [[] for _ in range(N)]
        for dk in offsets_key:
            neighbor_keys = keys + dk
            for i in range(N):
                j = key_to_idx.get(neighbor_keys[i])
                if j is not None:
                    neighbors[i].append(j)

        # BFS from all covered voxels
        from collections import deque
        covered_mask = best_conf.numpy() > 0
        best_rgb_np = best_rgb.numpy().copy()
        queue = deque()
        for i in range(N):
            if covered_mask[i]:
                queue.append(i)

        filled = 0
        while queue:
            i = queue.popleft()
            for j in neighbors[i]:
                if not covered_mask[j]:
                    covered_mask[j] = True
                    best_rgb_np[j] = best_rgb_np[i]
                    filled += 1
                    queue.append(j)

        best_rgb = torch.from_numpy(best_rgb_np)
        best_conf = torch.where(torch.from_numpy(covered_mask),
                                torch.clamp(best_conf, min=1), best_conf)
        print(f"  Gap-fill: filled {filled:,} voxels via BFS  ({time.time()-t0:.1f}s)")
        uncovered_after = (best_conf == 0).sum().item()
        print(f"📊 After gap-fill: {N - uncovered_after:,}/{N:,} covered "
              f"({100*(N-uncovered_after)/N:.1f}%)")
    else:
        print("  No gap-fill needed (100% covered)")

    # --- 保存 .vxz ---
    attr = {}
    attr['top6'] = torch.cat([best_rgb, best_conf.unsqueeze(-1)], dim=-1)   # [N, 4], uint8

    o_voxel.io.write_vxz(output_vxz, coord.cpu(), attr)
    print(f"✅ Saved PBR voxel grid to: {output_vxz}")

    # --- 渲染 ---
    render_script = os.path.join(SCRIPT_DIR, "render_vxz1024.py")
    cmd = [sys.executable, render_script, output_vxz]
    print(f"🎨 启动渲染: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ 渲染完成: {output_vxz[:-4]+'.mp4'}")
    else:
        print(f"❌ 渲染失败: {result.stderr.strip()}")


if __name__ == '__main__':
    main()
