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

### Single Model Rendering

```bash
conda activate bpy40

# Horizontal orbit, 120 cameras, GPU rendering
python3 render.py \
    --input_file /path/to/model.glb \
    --output_dir ./output \
    --orbit horizontal \
    --num_cameras 120

# Vertical orbit with left-right flip (recommended for V track)
python3 render.py \
    --input_file /path/to/model.glb \
    --output_dir ./output \
    --orbit vertical \
    --num_cameras 120 \
    --flip_x

# Custom model name (overrides GLB filename)
python3 render.py \
    --input_file /path/to/model.glb \
    --output_dir ./output \
    --model_name my_model

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

## Command-Line Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_file` | (required) | Path to input GLB/OBJ/FBX/PLY file |
| `--output_dir` | `outputs` | Base output directory |
| `--model_name` | GLB filename | Custom model name for output subdirectory |
| `--orbit` | `horizontal` | Camera orbit: `horizontal` or `vertical` |
| `--num_cameras` | `120` | Number of camera positions |
| `--flip_x` | `false` | Flip right axis (left-right mirror), recommended for vertical orbit |
| `--engine` | `CYCLES_GPU` | Render engine: `CYCLES_GPU`, `CYCLES_CPU`, `BLENDER_EEVEE` |
| `--width` | `1024` | Render width in pixels |
| `--height` | `1024` | Render height in pixels |
| `--fps` | `24` | Video frame rate |
| `--camera_radius` | `1.5` | Camera orbit radius |
| `--azimuth_offset` | `-90` | Azimuth offset in degrees |
| `--scene_scale` | `1.0` | Scene normalization scale |
| `--env_map` | `assets/env_textures/...` | HDR environment map path |

> **Why `--flip_x`?** By default, vertical orbit cameras on `meridian_270` produce rendered images that are mirrored relative to the horizontal orbit's first frame. Use `--flip_x` to correct this so H and V first frames share the same viewpoint.

## Requirements

- Python 3.10
- bpy 4.0.0 (Blender Python API)
- numpy, imageio[ffmpeg], Pillow, scipy, tqdm
- CUDA-compatible GPU (for `CYCLES_GPU`)
