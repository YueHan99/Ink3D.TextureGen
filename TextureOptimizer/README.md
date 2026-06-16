# TextureOptimizer

Multi-view video → PBR texture baking. Takes a GLB mesh and generated H/V orbit videos, produces a voxelized PBR texture grid (.vxz) and turntable render (.mp4).

```
GLB mesh + H video + V video → Blender PBR → Voxelize → View selection → .vxz + .mp4
```

## Environment

```bash
conda create -n trellis2 python=3.10
conda activate trellis2
pip install o_voxel torch numpy opencv-python Pillow imageio trimesh utils3d gco-wrapper

# Blender 4.5+ (set BLENDER_PATH or install)
wget https://download.blender.org/release/Blender4.5/blender-4.5.1-linux-x64.tar.xz
tar -xf blender-4.5.1-linux-x64.tar.xz -C /tmp/
export BLENDER_PATH=/tmp/blender-4.5.1-linux-x64/blender
```

## Quick Start

Download example data from Hugging Face and run end-to-end:

```bash
# 1. Download example data (mesh + generated videos)
pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Yuehavingfun/ink3d-example-data', repo_type='dataset',
                  allow_patterns='034/*', local_dir='./example_data')
"

# 2. Bake PBR texture
BLENDER_PATH=/tmp/blender-4.5.1/blender-4.5.1-linux-x64/blender python3 voxelize.py \
    ./example_data/034/mesh.glb \
    --video ./example_data/034/h_034_ref.mp4 \
    --video_v ./example_data/034/v_034_ref.mp4 \
    --video_num_cols 4 --video_col 2 --video_v_num_cols 4 --video_v_col 2 \
    --priority_mode --depth_eps 5e-4 \
    --output_vxz 034.vxz --resolution 1024

# 3. PBR render 
python3 render_vxz.py \
    --vxz 034.vxz --mesh 034.pickle \
    -o 034_pbr.mp4 --roughness 0.15 --metallic 0.3 --turntable --shaded_only

# 034.vxz — voxelized PBR grid
# 034_pbr.mp4 — PBR turntable with HDR lighting
```

Videos use 4-panel format `[ref | condition | generated | albedo]`. `--video_col 2` selects the generated panel. Each case directory on Hugging Face contains:

- `mesh.glb`, `ref.png` — Render inputs
- `h120/`, `v120/` — Render outputs (condition videos: position, normal, albedo)
- `h_*.mp4`, `v_*.mp4` — OrbitVideoGen outputs (generated videos, used as TextureOptimizer input)

## Baking Modes

| Flag | Description |
|------|-------------|
| `--mode greedy` (default) | Per-voxel top-1 confidence across all views |
| `--mode graphcut` | MRF graph-cut for spatial smoothness |
| `--priority_mode` | **H-first**: H colors all voxels, V fills gaps, then BFS |

Priority mode with H120+V120 on spider: H 93.1%, V 6.3%, BFS 0.6%, total 100%.

## Key Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--video`, `--video_v` | — | H and V track videos |
| `--video_num_cols`, `--video_col` | 4, 2 | Video panel layout |
| `--video_v_num_cols`, `--video_v_col` | 4, 2 | V video panel layout |
| `--video_v_frame_indices` | — | Select specific V camera indices (e.g. `15,16,...,45,75,...,105`) |
| `--priority_mode` | false | H-first baking |
| `--depth_eps` | 2e-3 | Depth tolerance (5e-4 for less bleeding) |
| `--max_lock_angle` | 60 | First-frame locking angle (180 to disable) |
| `--flip_v` | — | Flip specific V camera right-axis (e.g. `30`) |
| `--resolution` | 1024 | Voxel grid resolution |
| `--output_vxz` | auto | Output .vxz path |

## Output

```
output/
├── output.vxz       # Voxelized PBR grid (base color per voxel)
├── output.pickle    # Blender PBR dump (camera params, geometry)
└── output.mp4       # Auto-generated turntable preview
```

## PBR Rendering

```bash
# Voxel splatter (fast, included): auto-runs after baking
python render_vxz1024.py output.vxz

# Full PBR render (slower, requires TRELLIS.2 repo in PYTHONPATH):
python render_vxz.py \
    --vxz output.vxz --mesh output.pickle -o pbr.mp4 \
    --roughness 0.15 --metallic 0.3 --turntable --shaded_only
```

## Example: Optimized H+V config

60 V frames (15-45 + 75-105) with flip correction for camera 30:

```bash
INDICES="15,16,...,45,75,76,...,105"
python3 voxelize.py mesh.glb \
    --video h_video.mp4 --video_v v_video.mp4 \
    --video_num_cols 4 --video_col 2 --video_v_num_cols 4 --video_v_col 2 \
    --video_v_frame_indices "$INDICES" \
    --priority_mode --depth_eps 5e-4 --flip_v 30 \
    --output_vxz output.vxz --resolution 1024
```

| Phase | Voxels | % |
|-------|--------|---|
| H-first + frame0 | 5,170,680 | 53.0% |
| V-fill (62 views) | 4,446,201 | 45.6% |
| BFS | 135,804 | 1.4% |
| **Total** | 9,752,685 | 100% |

60 V frames match the coverage of all 120 V frames at half the compute.

## UV Atlas Baking (Alternative)

Classic UV-unwrap + nvdiffrast PBR baking. Produces texture atlases (`.png`) with metallic/roughness channels and `.glb` with PBR material.

### Pipeline

```
GLB mesh + H albedo video + V albedo video + H MR video + V MR video
    → UV unwrap (xatlas) → bake albedo → bake metallic → bake roughness → PBR GLB
```

### Quick Start

```bash
conda activate trellis2
python bake_pbr.py \
    --sha256 "000-000/967fe402e33942188b9cf34f8f9be431" \
    --video_path "/path/to/albedo_h.mp4" \
    --output_dir ./output
```

### Input Files

Each UUID directory needs:
```
{uuid}/
├── albedo_h.mp4     # H albedo video
├── albedo_v.mp4     # V albedo video
├── mr.mp4           # H MR video (metallic + roughness)
├── mr_v.mp4         # V MR video
└── meta.json        # H/V camera parameters (auto-detected from Objaverse paths)
```

### Output

```
output/{bucket}_{uuid}/
├── bake_1024_a.glb      # PBR GLB (baseColor + metallicRoughness)
├── texture.png          # Albedo atlas
├── texture_metallic.png # Metallic atlas
└── texture_roughness.png# Roughness atlas
```

### Mode

`bake_texture()` supports two modes:

| Mode | Speed | Quality | Use case |
|------|-------|---------|----------|
| `fast` | ~3s for 240 views | Good | Quick iteration |
| `opt` | ~5min for 2500 iters | Best | Final output |

## Troubleshooting
- **Depth bleeding on thin surfaces**: Reduce `--depth_eps` (try 5e-4 or even 1e-4)
