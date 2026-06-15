#!/bin/bash
# Local training script - uses local model paths and local data mount
# For other users: use the default .sh which auto-downloads models via ModelScope

# Ensure local diffsynth package takes priority over editable install
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd):${PYTHONPATH}"

accelerate launch examples/wanvideo/model_training/train_14b_ref_normal_revise_light_aug_61_revise2_imageref0322.py \
  --data_file_keys "video,control_video,reference_image" \
  --dataset_base_path '/home/v-hanyue/blobmnt/objaverse_60k_120hv/videos_curve' \
  --dataset_metadata_path position2albedo_all_321hv.csv \
  --height 512 \
  --width 1024 \
  --num_frames 121 \
  --dataset_repeat 1 \
  --model_paths '["/home/v-hanyue/local_models/PAI/Wan2.2-Fun-A14B-Control/Wan2.2-Fun-A14B-Control/high_noise_model/diffusion_pytorch_model.safetensors", "/home/v-hanyue/local_models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth", "/home/v-hanyue/local_models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"]' \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --save_steps 50 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-Fun-A14B-Control_high_noise_lora_ref_position_normal_drop_revise512_1001_121hv_light_aug_61_512_imageref0322" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "control_video,reference_image" \
  --max_timestep_boundary 0.358 \
  --min_timestep_boundary 0
# boundary corresponds to timesteps [900, 1000]
