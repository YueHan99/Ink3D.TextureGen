# TextureOptimizer

Multi-view video → PBR texture baking. Takes a GLB mesh and generated H/V orbit videos, produces a voxelized PBR texture grid (.vxz) and turntable render (.mp4).

```
GLB mesh + H video + V video → Blender PBR → Voxelize → View selection → .vxz + .mp4
```

## Environment

```bash
conda create -n texopt python=3.10
conda activate texopt
pip install o_voxel torch numpy opencv-python Pillow imageio trimesh utils3d gco-wrapper

# Blender 4.5+ (set BLENDER_PATH or install)
wget https://download.blender.org/release/Blender4.5/blender-4.5.1-linux-x64.tar.xz
tar -xf blender-4.5.1-linux-x64.tar.xz -C /tmp/
export BLENDER_PATH=/tmp/blender-4.5.1-linux-x64/blender
```

## Quick Start

```bash
BLENDER_PATH=/tmp/blender-4.5.1/blender-4.5.1-linux-x64/blender python3 voxelize.py \
    mesh.glb \
    --video h_video.mp4 --video_v v_video.mp4 \
    --video_num_cols 4 --video_col 2 --video_v_num_cols 4 --video_v_col 2 \
    --priority_mode --depth_eps 5e-4 \
    --output_vxz output.vxz --resolution 1024
```

Videos use 4-panel format `[ref | condition | generated | albedo]`. `--video_col 2` selects the generated panel.

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

# Full PBR render (slower, requires TRELLIS.2 in PYTHONPATH):
PYTHONPATH=/path/to/TRELLIS.2:$PYTHONPATH python render_vxz.py \
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

## Troubleshooting

- **Blender not found**: Set `BLENDER_PATH` or `wget` from blender.org
- **V video wrong column**: Use `--video_v_num_cols 4 --video_v_col 2` for 4-panel V videos
- **`o_voxel` not found**: `pip install o_voxel`
- **`trellis2` not found**: Only needed for `render_vxz.py` PBR render; `render_vxz1024.py` works standalone
- **Depth bleeding on thin surfaces**: Reduce `--depth_eps` (try 5e-4 or even 1e-4)
