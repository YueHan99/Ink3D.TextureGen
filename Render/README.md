# Ink3D Render

Multi-pass 3D model rendering pipeline based on Blender. Supports horizontal (H) and vertical (V) camera orbits, outputting color, depth, normal, albedo, and position maps as images and videos.

## Features

- **Horizontal orbit (H)**: Camera circles the equator at a fixed elevation
- **Vertical orbit (V)**: Camera travels along a meridian (full 360° vertical loop)
- **Multi-pass output**: RGB, depth, normal, albedo, position, mask
- **Video export**: Auto-generates MP4 videos for each channel
- **GPU rendering**: CYCLES with CUDA/OptiX GPU acceleration
- **Batch rendering**: Process entire directories of GLB files

## Environment Setup

```bash
# 1. Create conda environment with Python 3.10
conda create -n bpy40 python=3.10
conda activate bpy40

# 2. Download and install bpy==4.0.0 (not available on PyPI)
#    Download from Hugging Face:
wget https://huggingface.co/datasets/Yuehavingfun/ink3d-example-data/resolve/main/bpy-4.0.0-py310-linux-x86_64.tar.gz
tar -xzf bpy-4.0.0-py310-linux-x86_64.tar.gz -C $CONDA_PREFIX/lib/python3.10/site-packages/

# 3. Install dependencies
pip install numpy imageio[ffmpeg] Pillow scipy tqdm requests zstandard

# 4. Install this package
cd Render
pip install -e .
```

> **Note**: `bpy==4.0.0` is no longer available on PyPI. We provide a pre-packaged version on Hugging Face. Requires Python 3.10 and Linux x86_64.

## Quick Start

### Example Data

Download example meshes from Hugging Face:

```bash
pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Yuehavingfun/ink3d-example-data', repo_type='dataset',
                  allow_patterns='034/*', local_dir='./example_data')
"
# Downloads mesh.glb + pre-rendered condition videos for case 034 (cup model)
```

### Single Model Rendering

```bash
conda activate bpy40

# Horizontal orbit, 120 cameras, GPU rendering
python3 render.py \
    --input_file ./example_data/034/mesh.glb \
    --output_dir ./output \
    --orbit horizontal \
    --num_cameras 120

# Vertical orbit
python3 render.py \
    --input_file ./example_data/034/mesh.glb \
    --output_dir ./output \
    --orbit vertical \
    --num_cameras 120

# CPU rendering (for machines without compatible GPU)
python3 render.py \
    --input_file /path/to/model.glb \
    --output_dir ./output \
    --engine CYCLES_CPU
```

### Batch Rendering

```bash
# Render all GLBs in a directory
python render_batch.py \
    --input_dir /path/to/glb_folder \
    --output_dir ./output \
    --orbit horizontal

# Batch render with vertical orbit + flip_x
python render_batch.py \
    --input_dir /path/to/glb_folder \
    --output_dir ./output \
    --orbit vertical \
    --flip_x

# Multi-threaded batch rendering
python render_batch.py \
    --input_dir /path/to/glb_folder \
    --output_dir ./output \
    --threads 4 \
    --skip 10  # skip first 10 files
```

## Output Structure

```
output/
└── <model_name>/
    └── h120/  (or v120)
        ├── images/
        │   ├── render_0001.png     # RGBA color
        │   ├── depth_0001.exr      # Depth (EXR)
        │   ├── depth_0001.png      # Depth (normalized PNG)
        │   ├── normal_0001.png     # Normal map
        │   ├── albedo_0001.png     # Albedo (base color)
        │   ├── position_0001.exr   # Position (EXR)
        │   └── position_0001.png   # Position (normalized PNG)
        ├── rgb.mp4                 # Color video (white background)
        ├── mask.mp4                # Alpha mask video
        ├── depth.mp4               # Depth video
        ├── normal.mp4              # Normal video
        ├── albedo.mp4              # Albedo video
        ├── position.mp4            # Position video
        └── meta.json               # Camera parameters & render info
```

## Dataset Preparation (Batch)

For large-scale dataset rendering, two batch scripts handle multi-threaded, resumable processing of GLB lists.

### Albedo / Normal / Position pass

Renders multi-pass channels (rgb, albedo, normal, position, depth, mask) for H and V orbits.

```bash
python scripts/batch_render_albedo.py \
    --input_json selected_glb_paths.json \
    --base_glb_path /path/to/glbs \
    --output_dir_h ./output_h --output_dir_v ./output_v \
    --num_camera 120 --threads 4 --skip 0
```

### Metallic / Roughness pass

Renders PBR material properties (mr.mp4). Requires `render_mr.py`.

```bash
python scripts/batch_render_mr.py \
    --input_json selected_glb_paths.json \
    --base_glb_path /path/to/glbs \
    --output_dir_h ./output_h --output_dir_v ./output_v \
    --num_camera 120 --threads 4
```

| Argument | Description |
|----------|-------------|
| `--input_json` | JSON list of relative GLB paths |
| `--base_glb_path` | Root directory for GLB files |
| `--output_dir_h/v` | Output directories for H/V orbits |
| `--num_camera` | Cameras per orbit (120) |
| `--threads` | Parallel workers (1) |
| `--skip` | Resume from index N |

### Reconstruction: Bake Videos Back to GLB

验证数据集质量——用渲染视频 + 相机参数反向重建带 PBR 纹理的 GLB。

```bash
conda activate trellis2
# 参考代码: OrbitVideoGen/examples/ 下的 bake_pbr.py (UV atlas + nvdiffrast)
python bake_pbr.py \
    --sha256 "000-000/{uuid}" \
    --video_path "./albedo_h.mp4" \
    --output_dir ./output
```

| 输入 | 来源 |
|------|------|
| `albedo_h.mp4`, `albedo_v.mp4` | H/V 渲染产出的 albedo 视频 |
| `mr.mp4`, `mr_v.mp4` | H/V 的 metallic/roughness 视频 |
| `{uuid}.glb` | `glbs_normalized/` 目录下的归一化网格 |
| meta.json | 自动从 Objaverse metadata 路径读取 |

| 输出 | 说明 |
|------|------|
| `bake_1024_a.glb` | PBR GLB (baseColor + metallicRoughness) |
| `texture.png` | Albedo 纹理图集 |
| `texture_metallic.png` | Metallic 纹理 |
| `texture_roughness.png` | Roughness 纹理 |

## Command-Line Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_file` | (required) | Path to input GLB/OBJ/FBX/PLY file |
| `--output_dir` | `outputs` | Base output directory |
| `--model_name` | GLB filename | Custom model name for output subdirectory |
| `--orbit` | `horizontal` | Camera orbit: `horizontal` or `vertical` |
| `--num_cameras` | `120` | Number of camera positions |
| `--flip_x` | `false` | Flip right axis (left-right mirror) for vertical orbit |
| `--engine` | `CYCLES_GPU` | Render engine: `CYCLES_GPU`, `CYCLES_CPU`, `BLENDER_EEVEE` |
| `--width` | `1024` | Render width in pixels |
| `--height` | `1024` | Render height in pixels |
| `--fps` | `24` | Video frame rate |
| `--camera_radius` | `1.5` | Camera orbit radius |
| `--azimuth_offset` | `-90` | Azimuth offset in degrees |
| `--scene_scale` | `1.0` | Scene normalization scale |
| `--env_map` | `assets/env_textures/...` | HDR environment map path |

> **Note**: `--flip_x` is available as an optional correction if the vertical orbit produces flipped images relative to the horizontal orbit. Most models trained on non-flipped V data do not require this flag.

## Requirements

- Python 3.10
- bpy 4.0.0 (Blender Python API)
- numpy, imageio[ffmpeg], Pillow, scipy, tqdm
- CUDA-compatible GPU (for `CYCLES_GPU`)
