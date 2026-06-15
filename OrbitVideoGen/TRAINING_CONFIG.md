# OrbitVideoGen v2 Training Config Reference

## Experiment Pairs

### 1. Original experiments (train_v2_orbit.yaml)
Start: 2026-06-10

| Job | Noise | Timestep | Dataset | Rows |
|-----|-------|----------|---------|------|
| `v2_train_low_noise` | LOW (0-900) | boundary [0.358, 1] | `position2albedo_all_321_aeth.csv` | 4782 |
| `v2_train_high_noise` | HIGH (900-1000) | boundary [0, 0.358] | `position2albedo_all_321.csv` | 23617 |

### 2. Swap experiments (train_v2_orbit_swap.yaml)
Start: 2026-06-10

Same as above but **datasets swapped** to isolate dataset size vs noise level effects:

| Job | Noise | Dataset |
|-----|-------|---------|
| `v2_train_low_noise_swap` | LOW | `position2albedo_all_321.csv` (large) |
| `v2_train_high_noise_swap` | HIGH | `position2albedo_all_321_aeth.csv` (small) |

### 3. H+V joint training (train_v2_orbit_hv.yaml)
Start: 2026-06-10

| Job | Noise | Timestep | Dataset | Rows |
|-----|-------|----------|---------|------|
| `v2_hv_train_low_noise` | LOW (0-900) | boundary [0.358, 1] | `position2albedo_all_321_hv.csv` | 47234 |
| `v2_hv_train_high_noise` | HIGH (900-1000) | boundary [0, 0.358] | `position2albedo_all_321_hv.csv` | 47234 |

- `_hv.csv` = H data (`objaverse_60k_120h/`, 23617 rows) + V data (`objaverse_60k_120v/`, 23617 rows) merged & shuffled (seed=42)
- V data uses `position_flip.mp4` for control_video

### 4. Resume experiments (train_v2_orbit_resume.yaml)
Start: 2026-06-10

Continuation from interrupted original + swap runs:

| Job | Script | Resume from | Dataset |
|-----|--------|-------------|---------|
| `v2_resume_high_noise` | `run_remote_h_high_v2_resume.sh` | step-2250 | `321.csv` |
| `v2_resume_high_noise_swap` | `run_remote_h_high_v2_swap_resume.sh` | step-2100 | `321_aeth.csv` |
| `v2_resume_low_noise` | `run_remote_h_low_v2_resume.sh` | step-2000 | `321_aeth.csv` |
| `v2_resume_low_noise_swap` | `run_remote_h_low_v2_swap_resume.sh` | step-1850 | `321.csv` |

**What "swap" means**: The datasets between high_noise and low_noise models are exchanged.
- Original: high→321.csv, low→321_aeth.csv
- Swap: high→321_aeth.csv, low→321.csv
Purpose: isolate whether performance difference comes from noise level or dataset size.

## Model paths
- High noise: `.../Wan2.2-Fun-A14B-Control/Wan2.2-Fun-A14B-Control/high_noise_model/diffusion_pytorch_model.safetensors`
- Low noise: `.../Wan2.2-Fun-A14B-Control/Wan2.2-Fun-A14B-Control/low_noise_model/diffusion_pytorch_model.safetensors`
- T5: `.../Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth`
- VAE: `.../Wan2.1-T2V-1.3B/Wan2.1_VAE.pth`

## Training config (all experiments)
- Resolution: 512x512
- Frames: 121
- LoRA rank: 32, target: q,k,v,o,ffn.0,ffn.2
- LR: 1e-4, epochs: 5, save_steps: 50
- Extra inputs: control_video, reference_image
- Nodes: 2, GPUS: 4, SKU: 80G4-H100, Cluster: msrresrchbasicvc

## Dataset CSVs
| CSV | Rows | Description |
|-----|------|-------------|
| `position2albedo_all_321.csv` | 23617 | H track only |
| `position2albedo_all_321_aeth.csv` | 4782 | H track, aeth subset |
| `position2albedo_all_321_hv.csv` | 47234 | H+V merged & shuffled |

### 5. HV Resume experiments (train_v2_orbit_hv_resume.yaml)
Start: 2026-06-10

| Job | Script | Resume from | Dataset |
|-----|--------|-------------|---------|
| `v2_hv_resume_low_noise` | `run_remote_hv_low_v2_resume.sh` | step-2000 | `321_hv.csv` |
| `v2_hv_resume_high_noise` | `run_remote_hv_high_v2_resume.sh` | step-1700 | `321_hv.csv` |

## Entry points
- Shell scripts: `/home/v-hanyue/blobmnt/Ink3D/OrbitVideoGen/run_remote_h_*_v2*.sh`
- Training code: `examples/wanvideo/model_training/train_14b_ref_normal_revise_light_aug_61_revise2_121h0322_v2.py`
- AMLT YAMLs: `/home/v-hanyue/example/train_v2_orbit*.yaml`

## Inference

### Test scripts
| Script | Track | Weights |
|--------|-------|---------|
| `tests/test_single_h.py` | H (horizontal) | HV ckpts |
| `tests/test_single_v_hv.py` | V (vertical) | HV ckpts |
| `tests/test_single_hv.py` | HV joint | Old DiffSynth ckpts |
| `tests/test_single_v.py` | V | Old ckpts |
| `tests/test_single_v_as_h.py` | V via H model | Old ckpts |

### Local HV weights (azcopy from blob)
```
/home/v-hanyue/local_models/lora_ckpt/hv_high_noise_step-1700.safetensors
/home/v-hanyue/local_models/lora_ckpt/hv_low_noise_step-2000.safetensors
```
Source: `https://msraiegmultimedia.blob.core.windows.net/v-hanyue/Ink3D/OrbitVideoGen/models/train/v2_*_noise_lora_hv_imageref0322_imgseq/step-*.safetensors`

### H+V two-step inference workflow
```
# Step 1: H track inference → produces horizontal orbit video
cd /home/v-hanyue/workspace/Ink3D/OrbitVideoGen
export PYTHONPATH="$(pwd):${PYTHONPATH}"
python tests/test_single_h.py \
    --ref_image tests/example_data/<case>/ref.png \
    --video_dir tests/example_data/<case>/h120 \
    --output output_h.mp4

# Step 2: V track inference → uses H frame 0 for first-frame latent replacement
python tests/test_single_v_hv.py \
    --ref_image tests/example_data/<case>/ref.png \
    --video_dir tests/example_data/<case>/v120 \
    --h_video output_h.mp4 \
    --output output_v.mp4
```
- `--h_video`: extracts frame 0 from H output, VAE-encodes it, injects into V latent frame 0 at every denoise step
- Ensures H and V results share identical first frame appearance

### Example data (see FRAME_COUNTS.md)
- 120-frame (numbered): use ref.png, h120/, v120/
- 121-frame (named): use G000.png, h120/, v120/
- Ref image: numbered cases use `ref.png`, named cases use `G000.png`
