accelerate launch examples/wanvideo/model_training/train_14b_ref_normal_revise_light_aug_61_revise2_imageref0322.py \
  --data_file_keys "video,control_video,reference_image" \
  --dataset_base_path "${DATASET_BASE:-/mnt/v-hanyue/objaverse_60k_120hv/videos_curve}" \
  --dataset_metadata_path position2albedo_all_321hv.csv \
  --height 512 \
  --width 1024 \
  --num_frames 121 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "PAI/Wan2.2-Fun-A14B-Control:low_noise_model/diffusion_pytorch_model*.safetensors,PAI/Wan2.2-Fun-A14B-Control:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.2-Fun-A14B-Control:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --save_steps 50 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-Fun-A14B-Control_low_noise_lora_ref_position_normal_drop_revise512_1001_121hv_light_aug_61_512_boundrevise_imageref0322" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "control_video,reference_image" \
  --max_timestep_boundary 1 \
  --min_timestep_boundary 0.358
# boundary corresponds to timesteps [900, 1000]
  # --lora_checkpoint "./models/train/Wan2.2-Fun-A14B-Control_low_noise_lora_ref_position_normal_drop_revise512_1001_121hv_light_aug_61_512_boundrevise_imageref0316/step-1950.safetensors" \
