# Ink3D — 3D Texture Generation Pipeline

End-to-end pipeline: 3D mesh → multi-view rendered videos → AI-generated texture videos → PBR texture baking.

```
┌─────────┐    ┌──────────────┐    ┌──────────────────┐
│  GLB    │───→│   Render     │───→│  OrbitVideoGen   │
│  mesh   │    │  H/V videos  │    │  AI inference    │
└─────────┘    └──────────────┘    └──────────────────┘
                                            │
                    ┌───────────────────────┘
                    ▼
              ┌──────────────┐
              │TextureOptimizer│
              │ PBR baking    │
              └──────────────┘
                    │
                    ▼
              .vxz + .mp4 + .glb
```

## Quick Start (Spider Example)

### Step 1: Render condition videos

```bash
conda activate bpy40
python3 Render/render.py \
    --input_file OrbitVideoGen/tests/example_data/spider/mesh.glb \
    --output_dir OrbitVideoGen/tests/example_data \
    --orbit horizontal --num_cameras 120 --model_name spider
python3 Render/render.py \
    --input_file OrbitVideoGen/tests/example_data/spider/mesh.glb \
    --output_dir OrbitVideoGen/tests/example_data \
    --orbit vertical --num_cameras 120 --model_name spider --flip_x
# Output: spider/h120/{position,normal,albedo,rgb}.mp4 (120 frames each)
#         spider/v120/{position,normal,albedo,rgb}.mp4
```

### Step 2: Generate AI videos

```bash
cd OrbitVideoGen
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# H track inference
python tests/test_single_h.py \
    --ref_image tests/example_data/spider/G000.png \
    --video_dir tests/example_data/spider/h120 \
    --model_ckpt_high /path/to/high_noise_lora.safetensors \
    --model_ckpt_low /path/to/low_noise_lora.safetensors \
    --output output/spider/h_spider.mp4

# V track inference
python tests/test_single_v_hv.py \
    --ref_image tests/example_data/spider/G000.png \
    --video_dir tests/example_data/spider/v120 \
    --model_ckpt_high /path/to/high_noise_lora.safetensors \
    --model_ckpt_low /path/to/low_noise_lora.safetensors \
    --output output/spider/v_spider.mp4
# Output: 4-panel videos [ref | condition | generated | albedo], 121 frames each
```

### Step 3: PBR Texture Baking

```bash
export BLENDER_PATH=/tmp/blender-4.5.1/blender-4.5.1-linux-x64/blender
python3 TextureOptimizer/voxelize.py \
    OrbitVideoGen/tests/example_data/spider/mesh.glb \
    --video output/spider/h_spider.mp4 \
    --video_v output/spider/v_spider.mp4 \
    --video_num_cols 4 --video_col 2 \
    --video_v_num_cols 4 --video_v_col 2 \
    --priority_mode --depth_eps 5e-4 \
    --output_vxz output/spider/spider.vxz \
    --resolution 1024

# PBR render (optional, requires trellis2 module)
TRELLIS_ROOT=/path/to/TRELLIS.2 PYTHONPATH=$TRELLIS_ROOT:$PYTHONPATH python3 TextureOptimizer/render_vxz.py \
    --vxz output/spider/spider.vxz \
    --mesh output/spider/spider.pickle \
    -o output/spider/spider_pbr.mp4 \
    --roughness 0.15 --metallic 0.3 --turntable --shaded_only
# Output: spider.vxz (voxel PBR grid), spider.pickle, spider_pbr.mp4
```

## Modules

| Module | Description | Docs |
|--------|-------------|------|
| `Render/` | Blender-based H/V orbit rendering | `Render/README.md` |
| `OrbitVideoGen/` | WAN 14B video generation + training | `OrbitVideoGen/TRAINING_CONFIG.md` |
| `TextureOptimizer/` | Multi-view → PBR texture baking | `TextureOptimizer/README.md` |

## Trained Models

14B WAN model with LoRA fine-tuning on H+V orbit data. Training configs and experiments tracked in `OrbitVideoGen/TRAINING_CONFIG.md`.

## Results

| Case | H coverage | V coverage | Total (before BFS) |
|------|-----------|-----------|---------------------|
| spider | 93.1% | — | 99.9% (H+V priority) |
| 034 (cup) | 61.5% | 96.2% | 98.6% (H120+V60) |
