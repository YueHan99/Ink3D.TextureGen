#!/bin/bash
# Remote training: H+V interleaved (HIGH noise, 512x512)
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd):${PYTHONPATH}"

accelerate launch examples/wanvideo/model_training/train_14b_hv_interleave.py \
  --dataset_base_path "${DATASET_BASE:-/mnt/v-hanyue}" \
  --dataset_metadata_path Ink3D/OrbitVideoGen/position2albedo_all_321_hv_paired.csv \
  --data_file_keys "h_video,h_control_video,v_video,v_control_video" \
  --height 512 \
  --width 512 \
  --dataset_repeat 1 \
  --model_paths '["/${MODELS_BASE:-/mnt/v-hanyue/local_models}/PAI/Wan2.2-Fun-A14B-Control/Wan2.2-Fun-A14B-Control/high_noise_model/diffusion_pytorch_model.safetensors", "/${MODELS_BASE:-/mnt/v-hanyue/local_models}/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth", "/${MODELS_BASE:-/mnt/v-hanyue/local_models}/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"]' \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --save_steps 50 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/v2_high_noise_lora_hv_interleave" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "reference_image" \
  --max_timestep_boundary 0.358 \
  --min_timestep_boundary 0
# boundary corresponds to timesteps [900, 1000]
