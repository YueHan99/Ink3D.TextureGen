# OrbitVideoGen

Video generation with WAN 14B + LoRA for multi-view textured appearance.

```
Condition Videos (H+V) + Reference Image → WAN 14B → Generated Videos (4-panel: ref|condition|generated|albedo)
```

## Directory

```
OrbitVideoGen/
├── tests/
│   ├── test_single_h.py         # H-track inference
│   ├── test_single_v_hv.py      # V-track inference
│   ├── bg_remover.py            # RMBG-2.0 background removal
│   └── batch_hv_infer.sh        # Batch H+V inference
├── scripts/
│   ├── generate_hv_csv.py       # Merge H+V CSV for training
│   ├── generate_hv_paired_csv.py # Create paired H+V CSV (interleave)
│   ├── run_hv_interleave_high.sh # High-noise interleave training
│   └── run_hv_interleave_low.sh  # Low-noise interleave training
├── examples/                     # Training scripts
├── diffsynth/                    # WAN pipeline, LoRA, VAE, trainers
└── TRAINING_CONFIG.md           # Training experiment log
```

## Quick Start

### Inference

```bash
cd OrbitVideoGen
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# H track
python tests/test_single_h.py \
    --ref_image /path/to/ref.png \
    --video_dir /path/to/h120 \
    --models_base /path/to/models \
    --model_ckpt_high /path/to/high_noise_lora.safetensors \
    --model_ckpt_low /path/to/low_noise_lora.safetensors \
    --output output_h.mp4

# V track
python tests/test_single_v_hv.py \
    --ref_image /path/to/ref.png \
    --video_dir /path/to/v120 \
    --models_base /path/to/models \
    --model_ckpt_high /path/to/high_noise_lora.safetensors \
    --model_ckpt_low /path/to/low_noise_lora.safetensors \
    --output output_v.mp4
# Output: 4-panel MP4 [ref | condition | generated | albedo], 121 frames
```

### Background Removal

```bash
# Uses RMBG-2.0 (download weights first)
python tests/bg_remover.py
# Add --no_bg_remove to skip if ref images already have proper background
```

## Training

See `TRAINING_CONFIG.md` for experiment configurations.

### Standard H+V Training

```bash
# Original training script (not included, see TRAINING_CONFIG.md for configs)
accelerate launch examples/wanvideo/model_training/train_14b_*.py \
  --data_file_keys "video,control_video,reference_image" \
  --dataset_metadata_path position2albedo_all_321_hv.csv \
  --height 512 --width 512 --num_frames 121 \
  --lora_rank 32 --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --extra_inputs "control_video,reference_image"
```

### Interleaved H+V Training

H frames (every-other) + V frames (every-other) concatenated:

```bash
bash scripts/run_hv_interleave_low.sh   # Low noise [0, 900]
bash scripts/run_hv_interleave_high.sh  # High noise [900, 1000]
```

## Options

| Argument | Description |
|----------|-------------|
| `--models_base` | Base path for pretrained models (WAN 14B, T5, VAE) |
| `--model_ckpt_high` | High-noise LoRA checkpoint |
| `--model_ckpt_low` | Low-noise LoRA checkpoint |
| `--ref_image` | Reference image for appearance conditioning |
| `--video_dir` | Directory with condition videos (position, normal, albedo) |
| `--h_video` | H-track output for first-frame latent replacement (V track only) |
| `--no_bg_remove` | Skip RMBG-2.0 background removal |
