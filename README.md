<p align="center">
  <img src="assets/teaser.png" alt="Ink3D Teaser" width="60%">
</p>

<h3 align="center">Ink3D: Sculpting 3D Assets with Extremely Complex Textures via Video Generative Models</h3>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/arXiv-coming_soon-b31b1b"></a>
  <a href="https://yuehan99.github.io/Ink3D-TextureGen/"><img src="https://img.shields.io/badge/Project-Page-blue"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green"></a>
</p>

---

## 📰 News

- **[2026/06/15]** 🔥🔥🔥 Initial release. See [`Render/`](Render/README.md) for condition video rendering, [`OrbitVideoGen/`](OrbitVideoGen/README.md) for video generation training & inference, and [`TextureOptimizer/`](TextureOptimizer/README.md) for PBR texture baking.
- **[2026/06/15]** 📦 Example data & pretrained weights available on [Hugging Face](https://huggingface.co/datasets/Yuehavingfun/ink3d-example-data).
- **[2026/06/15]** 🚀 Four training modes: H-only, H/V random, spatial concat, temporal interleave — see [`OrbitVideoGen/scripts/`](OrbitVideoGen/scripts/).
- **[2026/06/15]** 🎨 Priority baking with depth-aware view selection: H-first → V-fill → BFS. 99.9% coverage on complex meshes.

---

## Overview

**Ink3D** is an end-to-end pipeline for generating high-quality PBR textures on arbitrary 3D meshes using video generative models. Given a 3D mesh and a reference image, the system:

1. **Renders** multi-pass condition videos (position, normal, albedo) from horizontal and vertical camera orbits
2. **Generates** textured appearance videos using a fine-tuned 14B video diffusion model with appearance-conditioned control
3. **Bakes** the generated multi-view videos into a voxelized PBR texture grid via GPU-accelerated depth-aware view selection

```
  GLB Mesh  ──→  Render  ──→  Video Gen  ──→  Texture Baking  ──→  .vxz + .glb
  (geometry)     (H/V)        (WAN 14B)        (priority mode)     (PBR asset)
```

## Directory Structure

```
Ink3D/
├── Render/               # Blender-based H/V orbit rendering
│   ├── render.py         #   Single model / batch rendering
│   └── src/ink3d_render/ #   Camera orbits, multi-pass outputs
├── OrbitVideoGen/        # Video generation & training
│   ├── tests/            #   Inference scripts (H, V, H+V)
│   ├── diffsynth/        #   WAN pipeline, LoRA, VAE
│   ├── examples/         #   Training scripts (14B + LoRA)
│   └── TRAINING_CONFIG.md
├── TextureOptimizer/     # PBR texture baking
│   ├── voxelize.py       #   Depth-aware voxel baking
│   ├── render_vxz.py     #   PBR render with HDR envmap
│   └── README.md
└── assets/               # Teaser & media
```

## Getting Started

### Prerequisites

- Linux with CUDA GPU (A100/H100 recommended)
- Conda (Miniconda or Anaconda)

### Environment

Each module has its own conda environment:

```bash
# Render (module 1): Python 3.10 + bpy 4.0.0
conda create -n bpy40 python=3.10
conda activate bpy40
pip install numpy imageio[ffmpeg] Pillow scipy tqdm
# bpy 4.0.0: download from Hugging Face (see Render/README.md)

# OrbitVideoGen (module 2): Python 3.10 + PyTorch
conda create -n orbitgen python=3.10
conda activate orbitgen
pip install torch torchvision torchaudio
pip install diffusers accelerate transformers imageio[ffmpeg] opencv-python-headless scipy numpy Pillow pandas

# TextureOptimizer (module 3): Python 3.10 + o_voxel
conda create -n trellis2 python=3.10
conda activate trellis2
pip install o_voxel torch numpy opencv-python Pillow imageio trimesh utils3d gco-wrapper
```

### One-time Setup

```bash
# Blender 4.5+ for PBR preprocessing (TextureOptimizer)
wget https://download.blender.org/release/Blender4.5/blender-4.5.1-linux-x64.tar.xz
tar -xf blender-4.5.1-linux-x64.tar.xz -C /tmp/
export BLENDER_PATH=/tmp/blender-4.5.1-linux-x64/blender

# RMBG-2.0 (background removal, OrbitVideoGen)
# Download from Hugging Face to /path/to/RMBG-2.0
```

## Quick Start

### Download Example Data

```bash
pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Yuehavingfun/ink3d-example-data', repo_type='dataset',
                  allow_patterns='034/*', local_dir='./example_data')
"
```

### Render → Generate → Bake

See module-level READMEs for detailed instructions:

| Module | Description | Documentation |
|--------|-------------|---------------|
| `Render/` | Condition video rendering | [`Render/README.md`](Render/README.md) |
| `OrbitVideoGen/` | Video generation & training | [`OrbitVideoGen/TRAINING_CONFIG.md`](OrbitVideoGen/TRAINING_CONFIG.md) |
| `TextureOptimizer/` | PBR texture baking | [`TextureOptimizer/README.md`](TextureOptimizer/README.md) |

### Minimal Pipeline (Spider Example)

```bash
# 1. Render condition videos
python3 Render/render.py --input_file spider/mesh.glb --output_dir ./output \
    --orbit horizontal --num_cameras 120 --model_name spider
python3 Render/render.py --input_file spider/mesh.glb --output_dir ./output \
    --orbit vertical --num_cameras 120 --model_name spider --flip_x

# 2. Generate textured videos (requires trained LoRA weights)
python3 OrbitVideoGen/tests/test_single_h.py \
    --ref_image spider/ref.png --video_dir output/spider/h120 \
    --model_ckpt_high /path/to/high.safetensors \
    --model_ckpt_low /path/to/low.safetensors \
    --output h_spider.mp4
python3 OrbitVideoGen/tests/test_single_v_hv.py \
    --ref_image spider/ref.png --video_dir output/spider/v120 \
    --model_ckpt_high /path/to/high.safetensors \
    --model_ckpt_low /path/to/low.safetensors \
    --output v_spider.mp4

# 3. Bake PBR texture
python3 TextureOptimizer/voxelize.py spider/mesh.glb \
    --video h_spider.mp4 --video_v v_spider.mp4 \
    --video_num_cols 4 --video_col 2 --video_v_num_cols 4 --video_v_col 2 \
    --priority_mode --depth_eps 5e-4 \
    --output_vxz spider.vxz --resolution 1024
```

## Key Features

- **Dual-orbit coverage**: Horizontal (360°) + vertical (360°) camera paths for complete surface coverage
- **Priority view selection**: H-first strategy with V gap-filling, achieving 99%+ voxel coverage
- **Depth-aware projection**: NDC depth testing with tunable tolerance prevents texture bleeding across thin surfaces
- **Voxel PBR output**: `.vxz` format with per-voxel base color, ready for metallic/roughness PBR rendering
- **Open-source friendly**: All shader-free; Blender 4.5+ for mesh preprocessing only

## Results

| Mesh | Voxels | H coverage | V coverage | Total (before BFS) |
|------|--------|------------|------------|---------------------|
| Spider | 4.39M | 93.1% | — | 99.9% |
| Cup (034) | 9.75M | 58.9% | 94.9% | 98.6% |

## Citation

```bibtex
@article{ink3d2026,
  title   = {Ink3D: Sculpting 3D Assets with Extremely Complex Textures via Video Generative Models},
  author  = {},
  journal = {},
  year    = {2026},
}
```

## License

MIT License. See `LICENSE` for details.
