# OrbitVideoGen

Multi-view video generation with WAN 14B + LoRA. Given a 3D mesh's rendered condition videos (position, normal, albedo) and a reference image, generates textured appearance videos from orbital viewpoints.

```
Condition Videos (H/V orbit) + Reference Image → WAN 14B + LoRA → Generated Video (4-panel MP4)
```

## Directory

```
OrbitVideoGen/
├── tests/
│   ├── test_single_h.py         # H-track inference
│   ├── test_single_v_hv.py      # V-track inference
│   └── bg_remover.py            # RMBG-2.0 background removal
├── scripts/
│   ├── train_*.sh               # Training launch scripts (4 modes × high/low)
│   └── generate_*.py            # CSV generation utilities
├── examples/                     # Training code
│   └── wanvideo/model_training/
├── diffsynth/                    # WAN pipeline, LoRA, VAE, trainers
└── README.md
```

## Environment

```bash
conda create -n orbitgen python=3.10
conda activate orbitgen

# Install PyTorch (CUDA 12.x)
pip install torch torchvision torchaudio

# Install dependencies
pip install diffusers accelerate transformers imageio[ffmpeg] opencv-python-headless \
    scipy numpy Pillow pandas

# Download base models (WAN 14B, T5, VAE) — paths configured in training scripts
# Download LoRA weights from Hugging Face or train your own
```

## Training Modes

Four data augmentation strategies for training the model with horizontal (H) and vertical (V) orbit data.

### Mode 1: H-only

Standard training using only horizontal orbit videos.

| Property | Value |
|----------|-------|
| Resolution | 512×512 |
| Frames | 121 |
| CSV | `position2albedo_all_321.csv` |
| Training script | `train_14b_ref_normal_revise_light_aug_61_revise2_121h0322_v2.py` |

```bash
# Training
bash scripts/train_h_only_high.sh    # High noise [900, 1000]
bash scripts/train_h_only_low.sh     # Low noise [0, 900]

# Inference
python tests/test_single_h.py \
    --ref_image ./example_data/034/ref.png \
    --video_dir ./example_data/034/h120 \
    --models_base /path/to/models \
    --model_ckpt_high /path/to/high.safetensors \
    --model_ckpt_low /path/to/low.safetensors \
    --output output_h.mp4
```

### Mode 2: H/V Random Sampling

H and V entries randomly interleaved in training CSV. Each batch draws from either H or V data.

| Property | Value |
|----------|-------|
| Resolution | 512×512 |
| Frames | 121 |
| CSV | `position2albedo_all_321_hv.csv` |
| Training script | Same as Mode 1 |

```bash
# Generate merged CSV (if needed)
python scripts/generate_hv_csv.py

# Training
bash scripts/train_hv_random_high.sh
bash scripts/train_hv_random_low.sh

# Inference (same as Mode 1 — separate H and V tests)
python tests/test_single_h.py ... --output output_h.mp4
python tests/test_single_v_hv.py ... --output output_v.mp4
```

### Mode 3: H/V Spatial Concat

H and V frames concatenated spatially (side-by-side) into a single 1024×512 frame. Each frame contains both viewpoints simultaneously.

| Property | Value |
|----------|-------|
| Resolution | 1024×512 (H|V) |
| Frames | 61 (stride-2 from 121-frame source) |
| CSV | `position2albedo_all_321hv.csv` |
| Training script | `train_14b_ref_normal_revise_light_aug_61_revise2_imageref0322.py` |

```
Frame layout:  [ H 512×512 | V 512×512 ]
                ←    1024    →
```

```bash
# Training
bash scripts/train_hv_spatial_high.sh
bash scripts/train_hv_spatial_low.sh

# Inference (use same test scripts, model expects 1024x512 input)
python tests/test_single_h.py \
    --width 1024 --height 512 \
    --video_dir ./example_data/034/h120 \
    ...
```

### Mode 4: H/V Temporal Interleave

H and V frames interleaved in time: 60 H frames (stride-2) + 60 V frames (stride-2) + H[0] loop closure = 121 frames.

| Property | Value |
|----------|-------|
| Resolution | 512×512 |
| Frames | 60 H + 60 V + 1 = 121 |
| CSV | `position2albedo_all_321_hv_paired.csv` |
| Training script | `train_14b_hv_interleave.py` |

```
Timeline: [H0, H2, H4, ..., H118] [V0, V2, V4, ..., V118] [H0]
            ← 60 H frames →       ← 60 V frames →       loop
```

```bash
# Generate paired CSV
python scripts/generate_hv_paired_csv.py

# Training
bash scripts/train_hv_interleave_high.sh
bash scripts/train_hv_interleave_low.sh

# Inference (same as Mode 1)
python tests/test_single_h.py ... --output output_h.mp4
python tests/test_single_v_hv.py ... --output output_v.mp4
```

## Mode Comparison

| Mode | Resolution | Frames | H+V in one sample? | Training code |
|------|-----------|--------|---------------------|---------------|
| 1: H-only | 512×512 | 121 | No | `train_14b_*_v2.py` |
| 2: HV random | 512×512 | 121 | No (alternating) | Same as Mode 1 |
| 3: Spatial concat | 1024×512 | 121 | Yes (side-by-side) | `train_14b_*_imageref0322.py` |
| 4: Temporal interleave | 512×512 | 121 | Yes (back-to-back) | `train_14b_hv_interleave.py` |

## CSV Data Format

### Mode 1, 2 (single-video)
```csv
video,control_video,reference_image,prompt
objaverse_60k_120h/videos_curve/000/xxx/albedo.mp4,objaverse_60k_120h/.../position.mp4,...
```

### Mode 3, 4 (paired H+V)
```csv
h_video,h_control_video,v_video,v_control_video,reference_image,prompt
objaverse_60k_120h/.../albedo.mp4,.../position.mp4,objaverse_60k_120v/.../albedo.mp4,.../position_flip.mp4,...
```

Generate with:
```bash
python scripts/generate_hv_csv.py           # Mode 2: H+V merged (shuffled)
python scripts/generate_hv_paired_csv.py    # Mode 3/4: paired H+V per object
```

## Background Removal

```bash
# RMBG-2.0 (recommended, matches training distribution)
# Download weights to /path/to/RMBG-2.0
python tests/test_single_h.py --ref_image img.png ...   # auto-runs bg_remover
python tests/test_single_h.py --ref_image img.png --no_bg_remove ...  # skip if ref already has black bg
```
