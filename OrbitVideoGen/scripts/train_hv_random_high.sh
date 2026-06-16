#!/bin/bash
# Mode 2: H/V random sampling training (HIGH noise, 512x512, 121 frames)
# Data: H and V entries interleaved randomly in CSV
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):${PYTHONPATH}"

accelerate launch examples/wanvideo/model_training/train_14b_ref_normal_revise_light_aug_61_revise2_121h0322_v2.py \
  --data_file_keys "video,control_video,reference_image" \
  --dataset_base_path "${DATASET_BASE:-/mnt/v-hanyue}" \
  --dataset_metadata_path position2albedo_all_321_hv.csv \
  --height 512 --width 512 --num_frames 121 \
  --dataset_repeat 1 \
  --model_paths '["/${MODELS_BASE:-/mnt/v-hanyue/local_models}/PAI/Wan2.2-Fun-A14B-Control/Wan2.2-Fun-A14B-Control/high_noise_model/diffusion_pytorch_model.safetensors", "/${MODELS_BASE:-/mnt/v-hanyue/local_models}/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth", "/${MODELS_BASE:-/mnt/v-hanyue/local_models}/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"]' \
  --learning_rate 1e-4 --num_epochs 5 --save_steps 50 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/hv_random_high" \
  --lora_base_model "dit" --lora_target_modules "q,k,v,o,ffn.0,ffn.2" --lora_rank 32 \
  --extra_inputs "control_video,reference_image" \
  --max_timestep_boundary 0.358 --min_timestep_boundary 0
