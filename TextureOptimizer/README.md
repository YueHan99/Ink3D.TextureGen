# TextureOptimizer

Multi-view video to PBR texture baking pipeline. Takes a 3D mesh (GLB) and generated H/V orbit videos, produces a voxelized PBR texture grid (.vxz), turntable render (.mp4), and exportable GLB.

## Pipeline

```
GLB mesh + H video + V video → Blender PBR dump → Voxelize → View selection → VXZ → Render + GLB
```

Videos are expected as multi-panel MP4 files. The `--video_col` / `--video_num_cols` arguments specify which panel to use (e.g. column 2 of a 4-panel `[ref | condition | generated | albedo]` video).

## Environment

- Python 3.10 with CUDA
- Blender 4.5+ (set `BLENDER_PATH` env var or install via script)
- `o_voxel`, `torch`, `numpy`, `opencv-python`, `Pillow`, `gco`
- Conda env recommended: `conda create -n texopt python=3.10 && pip install o_voxel torch numpy opencv-python Pillow gco`

```bash
# Install Blender locally (if not in PATH)
azcopy cp --recursive "https://<account>.blob.core.windows.net/<container>/path/to/blender-4.5.1-linux-x64?<SAS>" /tmp/blender-4.5.1
cd /tmp/blender-4.5.1/blender-4.5.1-linux-x64/lib && python3 -c "
import os, glob
for f in glob.glob('lib*.so*'):
    parts = f.split('.so.')
    if len(parts)==2 and '.' in parts[1]:
        vp = parts[1].split('.')
        major, minor = f'{parts[0]}.so.{vp[0]}', f'{parts[0]}.so.{vp[0]}.{vp[1]}'
        [os.path.exists(x) or os.symlink(f, x) for x in (major, minor) if not os.path.exists(x)]
"
export BLENDER_PATH=/tmp/blender-4.5.1/blender-4.5.1-linux-x64/blender
```

## Quick Start

```bash
# 1. Generate videos with your video model (H + V orbit, 4-panel format)
# Output: h_spider_G000.mp4, v_spider_G000.mp4

# 2. Copy videos and mesh locally
mkdir -p test_outputs/spider_G000
cp /path/to/h_spider_G000.mp4 test_outputs/spider_G000/
cp /path/to/v_spider_G000.mp4 test_outputs/spider_G000/

# 3. Run texture baking (priority mode: H-first, V fills gaps)
BLENDER_PATH=/tmp/blender-4.5.1/blender-4.5.1-linux-x64/blender python3 voxelize.py \
    /path/to/spider/mesh.glb \
    --video test_outputs/spider_G000/h_spider_G000.mp4 \
    --video_v test_outputs/spider_G000/v_spider_G000.mp4 \
    --video_num_cols 4 --video_col 2 \
    --video_v_num_cols 4 --video_v_col 2 \
    --priority_mode --depth_eps 5e-4 \
    --output_vxz test_outputs/spider_G000/spider_G000.vxz \
    --resolution 1024

# 4. Optional: select specific V frames (15-45 + 75-105) with flip correction
INDICES="15,16,...,45,75,76,...,105"
python3 voxelize.py mesh.glb \
    --video h_video.mp4 --video_num_cols 4 --video_col 2 \
    --video_v v_video.mp4 --video_v_num_cols 4 --video_v_col 2 \
    --video_v_frame_indices "$INDICES" \
    --priority_mode --depth_eps 5e-4 --flip_v 30 \
    --output_vxz output.vxz --resolution 1024
```

## Modes

| Flag | Description |
|------|-------------|
| `--mode greedy` (default) | Per-voxel top-1 confidence across all H+V views |
| `--mode graphcut` | MRF graph-cut for spatial smoothness |
| `--priority_mode` | **H-first**: color all voxels from H, then fill gaps from V, then BFS |

### Priority Mode

```
Phase 1: H views → color all visible voxels (H has priority)
Phase 2: V views → fill remaining uncovered voxels from V
Phase 3: BFS gap-fill → any remaining holes
```

Example output with 120 H + 120 V views on spider mesh:

| Phase | Voxels covered | % of unlocked |
|-------|---------------|---------------|
| H-first | 3,469,114 | 93.1% |
| V-fill | 249,618 | 6.3% |
| BFS | 6,556 | 0.6% |
| **Total** | **4,391,683** | **100%** |

### Optimized H+V config (034 example, 120 H + 60 V center frames)

```bash
# 60 V frames from ranges 15-45 and 75-105 (center region, avoids pole flipping)
INDICES="15,16,...,45,75,76,...,105"
python3 voxelize.py mesh.glb \
    --video h_video.mp4 --video_num_cols 4 --video_col 2 \
    --video_v v_video.mp4 --video_v_num_cols 4 --video_v_col 2 \
    --video_v_frame_indices "$INDICES" \
    --priority_mode --depth_eps 5e-4 --flip_v 30 \
    --output_vxz output.vxz --resolution 1024
```

| Phase | Voxels covered | % total |
|-------|---------------|---------|
| Frame0 lock (H-000) | 612,088 | 6.3% |
| H-first greedy | 4,558,592 | 46.7% |
| V-fill (62 views) | 4,446,201 | 45.6% |
| BFS | 135,804 | 1.4% |
| **Total** | **9,752,685** | **100%** |

60 V frames achieve 98.6% bare coverage (same as 120 V frames with half the compute).

## Options

| Argument | Default | Description |
|----------|---------|-------------|
| `input_glb` | (required) | Input mesh (.glb) |
| `--video` | — | H-track video (multi-panel) |
| `--video_v` | — | V-track video (multi-panel) |
| `--video_num_cols` | 4 | Columns in H video |
| `--video_col` | 2 | Which column to use (0-indexed) |
| `--video_v_num_cols` | 8 | Columns in V video |
| `--video_v_col` | 5 | Which column in V (0-indexed) |
| `--output_vxz` | auto | Output .vxz path |
| `--resolution` | 1024 | Voxel grid resolution |
| `--mode` | greedy | View selection: `greedy` or `graphcut` |
| `--priority_mode` | false | H-first priority baking |
| `--camera_mode` | h | Camera for `--video`: `h` or `v` |
| `--video_frame_indices` | — | Comma-separated camera indices for main video (e.g. `30,90`) |
| `--video_v_frame_indices` | — | Comma-separated camera indices for V video |
| `--max_lock_angle` | 60.0 | Max view-normal angle for first-frame locking |
| `--depth_eps` | 2e-3 | NDC depth tolerance (smaller=less bleeding, e.g. `5e-4`) |
| `--flip_v` | — | Comma-separated V camera indices to flip right axis (e.g. `30`) |
| `--no_bfs` | false | Skip BFS gap-fill (show only view-colored voxels) |
| `--first_frame` | — | Override first frame for locking, e.g. `v-030` |
| `--lambda_smooth` | 5.0 | Graph-cut smoothness weight |
| `--max_views` | 10 | Per-voxel top-K for graph-cut |
| `--pickle_in` | — | Pre-existing PBR pickle (skip Blender) |
| `--frames` | — | Comma-separated frame indices |

## Output

```
test_outputs/spider_G000/
├── spider_G000.vxz       # Voxelized PBR grid
├── spider_G000.pickle     # Blender PBR dump
└── spider_G000.mp4        # Turntable render
```

## Troubleshooting

- **Blender not found**: Set `BLENDER_PATH` env var or install to `/tmp/blender-4.5.1/`
- **Missing libIex/libImath etc.**: azcopy from blob doesn't preserve symlinks. Run the symlink fix script above
- **V video used wrong column**: Use `--video_v_num_cols 4 --video_v_col 2` for 4-panel V videos
- **`o_voxel` not found**: Install with `pip install o_voxel` in your conda env

## PBR Rendering

Two render scripts are provided for visualizing baked `.vxz` results:

| Script | Output | Use case |
|--------|--------|----------|
| `render_vxz_single.py` (TRELLIS.2) | Single PNG | Preview at a specific camera angle |
| `render_vxz.py` (local) | MP4 video | Turntable animation or full orbit video |

Both share the same core pipeline: **vxz + mesh → MeshWithVoxel → GPU PBR render with HDR envmap**.

### Pipeline overview

```
.vxz (voxel grid)  ─┐
                     ├─→ MeshWithVoxel ─→ PBR Renderer ─→ PNG / MP4
.glb / .pickle mesh ─┘        │                    │
                               │              ┌─────┴─────┐
                         voxel attrs      envmap HDR    camera params
                      [base_color, metal,   (.exr)      (yaw, pitch, r, fov)
                       roughness, alpha]
```

#### Data flow

1. **Load envmap** — HDR environment map (`.exr`) → `EnvMap` tensor on GPU, provides PBR lighting
2. **Load mesh** — `.pickle` (TRELLIS dump) via `pickle.load()` or `.glb/.obj/.ply` via `trimesh.load()` → `(vertices, faces)`
3. **Load vxz** — `.vxz` via `o_voxel.io.read_vxz()` → `(coords, attrs)`
   - `base_color` from `top6` attribute (first 3 channels)
   - `metallic` / `roughness` are global constants applied to all voxels (not per-voxel from vxz)
   - Final attrs: `[base_color(3), metallic(1), roughness(1), alpha(1)]` ∈ [0,1]
4. **Build MeshWithVoxel** — combines mesh geometry + voxel texture grid into a single object
   - `origin=[-0.5, -0.5, -0.5]`, `voxel_size=1/resolution`
   - Layout maps 6-channel attrs to PBR material channels
5. **Render** — `render_frames()` with camera extrinsics/intrinsics + envmap
   - Camera: `yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r=2, fov=40°)`
   - Yaw offset by π to match turntable convention (0° = front)
   - Alpha-composited onto background color

### Single image (render_vxz_single.py)

Located at `TRELLIS.2/render_vxz_single.py` in the TRELLIS repo.

```bash
conda run -n trellis2 python /path/to/TRELLIS.2/render_vxz_single.py \
    --vxz input.vxz --mesh input.glb \
    --yaw 0 --elevation 20 --resolution 1024 \
    --metallic 0.3 --roughness 0.15 --white_bg
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--vxz` | (required) | Input .vxz file |
| `--mesh` | (required) | Mesh file (.pickle / .glb / .obj / .ply) |
| `--yaw` | 0 | Horizontal angle (0=front, 90=right, 180=back) |
| `--elevation` | 20 | Vertical angle in degrees (positive = looking down) |
| `--resolution` | 1024 | Output image resolution |
| `--voxel_resolution` | 1024 | Internal voxel grid resolution |
| `--metallic` | 0.3 | Global metallic value (0-1) |
| `--roughness` | 0.15 | Global roughness value (0-1) |
| `--white_bg` | false | White background instead of black |
| `--envmap` | assets/hdri/forest.exr | HDR environment map |
| `-o` / `--output` | auto | Output PNG path (default: `{vxz_base}_y{yaw}_e{elevation}.png`) |

### Turntable video (render_vxz.py)

Located at `render_vxz.py` in TextureOptimizer. Requires `trellis2` in PYTHONPATH.

```bash
# Set TRELLIS.2 path for trellis2 module
export TRELLIS_ROOT=/path/to/TRELLIS.2.train.gen/TRELLIS.2
PYTHONPATH=$TRELLIS_ROOT:$PYTHONPATH python render_vxz.py \
    --vxz input.vxz --mesh input.pickle \
    -o output.mp4 --turntable --elevation 20 \
    --num_frames 120 --fps 15 --resolution 1024 \
    --roughness 0.15 --metallic 0.3 --shaded_only
```

Example with spider:

```bash
# Build on blobmnt
TRELLIS_ROOT=/home/v-hanyue/blobmnt/workspace/TRELLIS.2.train.gen/TRELLIS.2
PYTHONPATH=$TRELLIS_ROOT:$PYTHONPATH python TextureOptimizer/render_vxz.py \
    --vxz test_outputs/spider_G000/spider_priority.vxz \
    --mesh test_outputs/spider_G000/spider_priority.pickle \
    -o test_outputs/spider_G000/spider_priority_pbr.mp4 \
    --roughness 0.15 --metallic 0.3 --turntable --shaded_only
# Output: 120-frame PBR turntable with HDR envmap lighting
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--vxz` | (required) | Input .vxz file |
| `--mesh` | (required) | Mesh file (.pickle / .glb / .obj / .ply) |
| `-o` / `--output` | render.mp4 | Output video path |
| `--num_frames` | 120 | Number of frames |
| `--fps` | 15 | Video FPS |
| `--resolution` | 1024 | Render resolution |
| `--voxel_resolution` | 1024 | Internal voxel grid resolution |
| `--metallic` | 0.0 | Global metallic value |
| `--roughness` | 0.9 | Global roughness value |
| `--white_bg` | false | White background |
| `--envmap` | assets/hdri/forest.exr | HDR environment map |
| `--turntable` | false | Horizontal 360° rotation at fixed elevation |
| `--elevation` | 20 | Camera elevation (only with `--turntable`) |
| `--shaded_only` | false | Output only shaded result (no debug panels) |

### Camera convention

```
        Z (up)
        |
        |
        +────── Y (front, yaw=0)
       /
      /
     X (right, yaw=90°)

yaw:     rotation around Z axis (0=front, 90=right, 180=back, -90=left)
elevation: angle above X-Y plane (positive = looking down at object)
r:         camera distance from origin (default 2.0)
fov:       vertical field of view in degrees (default 40°)
```

The script applies a π offset internally: `yaw_rad = π + yaw_deg * π/180` to match turntable convention where 0° faces the front of the model.

### Voxel renderer (o_voxel)

An alternative renderer `render_vxz1024.py` uses `o_voxel.rasterize.VoxelRenderer` for direct voxel splatting (no mesh, no PBR). This is faster but lower quality — useful for quick previews of voxel coverage.

```bash
python render_vxz1024.py input.vxz
# Output: input.mp4 (H+V orbit video, 242 frames)
```

### Prerequisites

```bash
conda create -n trellis2 python=3.10
conda activate trellis2
pip install torch numpy opencv-python Pillow imageio trimesh o_voxel utils3d
pip install trellis2  # or clone from TRELLIS.2 repo
```
