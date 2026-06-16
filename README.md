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

- **[2026/06/15]** 📦 Example data & pretrained weights available on [Hugging Face](https://huggingface.co/datasets/Yuehavingfun/ink3d-example-data).
- **[2026/06/15]** 🏗️ Dataset preparation: batch GLB rendering to custom camera-trajectory videos (rgb, albedo, normal, position, depth, mask, metalness/roughness) — see [`Render/scripts/batch_render_albedo.py`](Render/scripts/batch_render_albedo.py) and [`batch_render_mr.py`](Render/scripts/batch_render_mr.py).
- **[2026/06/15]** 🔥🔥🔥 Initial release. See [`Render/`](Render/README.md) for condition video rendering, [`OrbitVideoGen/`](OrbitVideoGen/README.md) for video generation training & inference, and [`TextureOptimizer/`](TextureOptimizer/README.md) for texture baking.

---

## Overview

**Ink3D** aims to generate complex textures on arbitrary 3D meshes using video generative models. Given a 3D mesh and a reference image, the system:

1. **Renders** geometry condition videos (position, normal) from horizontal and vertical camera orbits
2. **Generates** textured appearance videos using a fine-tuned 14B video diffusion model with geometry control
3. **Bakes** the generated multi-view videos into a voxelized texture grid via optimization

```
  GLB Mesh  ──→  Render  ──→  Video Gen  ──→  Texture Baking  ──→  .vxz + .glb
  (geometry)     (H/V)        (WAN 14B)        (priority mode)     (PBR asset)
```

## Quick Start: End-to-End Demo

Walk through the full pipeline. Pre-computed outputs are provided for every step — skip any stage using the provided data.

### Step 0: Environment

```bash
# Render + OrbitVideoGen
conda create -n ink3d python=3.10 -y && conda activate ink3d
pip install torch torchvision torchaudio diffusers accelerate transformers
pip install imageio[ffmpeg] opencv-python-headless scipy numpy Pillow pandas

# TextureOptimizer
conda create -n trellis2 python=3.10 -y && conda activate trellis2
pip install o_voxel numpy opencv-python Pillow imageio trimesh utils3d gco-wrapper

# One-time: Blender & RMBG-2.0
wget https://download.blender.org/release/Blender4.5/blender-4.5.1-linux-x64.tar.xz -P /tmp/
tar -xf /tmp/blender-4.5.1-linux-x64.tar.xz -C /tmp/
export BLENDER_PATH=/tmp/blender-4.5.1-linux-x64/blender
```

### Step 1: Download Example Data

```bash
pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Yuehavingfun/ink3d-example-data', repo_type='dataset',
                  allow_patterns='034/*', local_dir='./example_data')
"
# Downloads: mesh.glb, h120/, v120/, h_034_ref.mp4, v_034_ref.mp4
```

### Step 2: Render Condition Videos

*Skip if using pre-rendered from Step 1.*

```bash
conda create -n bpy40 python=3.10 -y && conda activate bpy40
# bpy 4.0.0: download from Hugging Face (see Render/README.md)
pip install -e Render/

python3 Render/render.py --input_file ./example_data/034/mesh.glb \
    --output_dir ./example_data/034 --model_name 034 \
    --orbit horizontal --num_cameras 120
python3 Render/render.py --input_file ./example_data/034/mesh.glb \
    --output_dir ./example_data/034 --model_name 034 \
    --orbit vertical --num_cameras 120
# Output: example_data/034/h120/*.mp4, v120/*.mp4
```

### Step 3: Generate Textured Videos

*Skip if using pre-generated from Step 1.*

```bash
conda activate ink3d
cd OrbitVideoGen && export PYTHONPATH="$(pwd):${PYTHONPATH}"

python tests/test_single_h.py \
    --ref_image ../example_data/034/ref.png \
    --video_dir ../example_data/034/h120 \
    --models_base /path/to/local_models \
    --model_ckpt_high ./weights/high_noise.safetensors \
    --model_ckpt_low ./weights/low_noise.safetensors \
    --output ../example_data/034/h_034_gen.mp4

python tests/test_single_v_hv.py \
    --ref_image ../example_data/034/ref.png \
    --video_dir ../example_data/034/v120 \
    --models_base /path/to/local_models \
    --model_ckpt_high ./weights/high_noise.safetensors \
    --model_ckpt_low ./weights/low_noise.safetensors \
    --output ../example_data/034/v_034_gen.mp4
```

### Step 4: Bake PBR Texture

```bash
conda activate trellis2
python3 TextureOptimizer/voxelize.py ./example_data/034/mesh.glb \
    --video ./example_data/034/h_034_ref.mp4 \
    --video_v ./example_data/034/v_034_ref.mp4 \
    --video_num_cols 4 --video_col 2 --video_v_num_cols 4 --video_v_col 2 \
    --priority_mode --depth_eps 5e-4 \
    --output_vxz ./example_data/034/034.vxz --resolution 1024
# Output: 034.vxz, 034.pickle, 034.mp4
```

### Step 5: PBR Render

```bash
python3 TextureOptimizer/render_vxz.py \
    --vxz ./example_data/034/034.vxz --mesh ./example_data/034/034.pickle \
    -o ./example_data/034/034_pbr.mp4 \
    --roughness 0.15 --metallic 0.3 --turntable --shaded_only
# Output: 034_pbr.mp4
```

| Step | Input | Output | Skip? |
|------|-------|--------|-------|
| Render | mesh.glb | h120/, v120/ | Use pre-rendered from HF |
| Video Gen | h120/, v120/, ref.png | h_034_ref.mp4, v_034_ref.mp4 | Use pre-generated from HF |
| Bake | mesh.glb + generated mp4 | 034.vxz, 034.pickle | — |
| PBR | 034.vxz + 034.pickle | 034_pbr.mp4 | — |

All intermediate outputs available on [Hugging Face](https://huggingface.co/datasets/Yuehavingfun/ink3d-example-data/tree/main/034).

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
