"""
Single-sample inference: V-track via H-model (rotate v120 videos 90 CW)

Usage:
  export PYTHONPATH="/home/v-hanyue/workspace/Ink3D/OrbitVideoGen:${PYTHONPATH}"
  python tests/test_single_v_as_h.py --ref_image /path/to/ref.png --video_dir /path/to/v120/ [--output output_v.mp4]

  video_dir should contain: position.mp4, normal.mp4, albedo.mp4 (vertical orbit)
  All condition videos and ref image are rotated 90 CW before inference,
  then the generated video is rotated 90 CCW back to vertical orientation.
"""
import torch
import os, sys, argparse
from PIL import Image
from diffsynth import save_video, VideoData, load_state_dict
from diffsynth.pipelines.wan_video_new_14b_ref_drop_normal import WanVideoPipeline, ModelConfig


def rgba_to_rgb(rgba_image, background_color=(255, 255, 255)):
    if rgba_image.mode != 'RGBA':
        return rgba_image.convert('RGB')
    bg = Image.new('RGB', rgba_image.size, background_color)
    alpha = rgba_image.split()[-1]
    bg.paste(rgba_image.convert('RGB'), mask=alpha)
    return bg


def rotate_cw(img):
    """Rotate PIL image 90 degrees clockwise."""
    return img.transpose(Image.ROTATE_270)


def rotate_ccw(img):
    """Rotate PIL image 90 degrees counter-clockwise."""
    return img.transpose(Image.ROTATE_90)


def main():
    parser = argparse.ArgumentParser(description="V-track inference via H-model (rotate 90 CW)")
    parser.add_argument("--ref_image", type=str, required=True, help="Path to reference image (png)")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing v120 videos (position.mp4, normal.mp4, albedo.mp4)")
    parser.add_argument("--output", type=str, default="./output_v_as_h.mp4", help="Output video path")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--cfg_scale", type=float, default=6.0)
    parser.add_argument("--prompt", type=str, default="This is a 3D model")
    parser.add_argument("--model_ckpt_high", type=str,
        default="/home/v-hanyue/local_models/lora_ckpt/high_noise_step-3500.safetensors")
    parser.add_argument("--model_ckpt_low", type=str,
        default="/home/v-hanyue/local_models/lora_ckpt/low_noise_step-3300.safetensors")
    parser.add_argument("--h_first_frame", type=str, default=None,
        help="Path to H-track first frame image (or H-track output video). "
             "If a video, extracts frame 0. Used as input_image so each denoise step replaces frame 0 with this image latent.")
    args = parser.parse_args()

    HEIGHT, WIDTH = args.height, args.width
    NUM_FRAMES = args.num_frames

    # --- Load model ---
    print("Loading models...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda:0",
        model_configs=[
            ModelConfig(path="/home/v-hanyue/local_models/PAI/Wan2.2-Fun-A14B-Control/Wan2.2-Fun-A14B-Control/high_noise_model/diffusion_pytorch_model.safetensors", offload_device="cpu"),
            ModelConfig(path="/home/v-hanyue/local_models/PAI/Wan2.2-Fun-A14B-Control/Wan2.2-Fun-A14B-Control/low_noise_model/diffusion_pytorch_model.safetensors", offload_device="cpu"),
            ModelConfig(path="/home/v-hanyue/local_models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(path="/home/v-hanyue/local_models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth", offload_device="cpu"),
        ],
    )

    # Inject patch embedding + LoRA
    state_dict1 = load_state_dict(args.model_ckpt_high)
    pipe.dit.patch_embedding.load_state_dict({
        'bias': state_dict1['patch_embedding.bias'],
        'weight': state_dict1['patch_embedding.weight']
    })
    state_dict2 = load_state_dict(args.model_ckpt_low)
    pipe.dit2.patch_embedding.load_state_dict({
        'bias': state_dict2['patch_embedding.bias'],
        'weight': state_dict2['patch_embedding.weight']
    })
    pipe.load_lora(pipe.dit, args.model_ckpt_high, alpha=1)
    pipe.load_lora(pipe.dit2, args.model_ckpt_low, alpha=1)
    pipe.enable_vram_management()
    print("Model loaded.")

    # --- Load data & rotate 90 CW ---
    def load_frames_rotated(video_path):
        vd = VideoData(video_path, height=HEIGHT, width=WIDTH)
        count = min(NUM_FRAMES, len(vd))
        frames = [vd[i] for i in range(count)] + [vd[0]]
        return [rotate_cw(f) for f in frames]

    print("Loading & rotating v120 condition videos 90 CW...")
    condition_video_frames = load_frames_rotated(os.path.join(args.video_dir, "position.mp4"))
    normal_video_frames = load_frames_rotated(os.path.join(args.video_dir, "normal.mp4"))
    albedo_video_frames = load_frames_rotated(os.path.join(args.video_dir, "albedo.mp4"))

    min_len = min(len(condition_video_frames), len(normal_video_frames), len(albedo_video_frames))

    reference_image = Image.open(args.ref_image)
    reference_image = rgba_to_rgb(reference_image)
    reference_image = reference_image.resize((WIDTH, HEIGHT))
    reference_image_rotated = rotate_cw(reference_image)

    # --- Load H-track first frame (optional, for first-frame latent replacement) ---
    h_first_frame_rotated = None
    if args.h_first_frame:
        h_path = args.h_first_frame
        if h_path.endswith(('.mp4', '.avi', '.mov', '.webm')):
            # Extract frame 0 from video
            vd = VideoData(h_path, height=HEIGHT, width=WIDTH)
            h_first_frame = vd[0]
        else:
            h_first_frame = Image.open(h_path)
            h_first_frame = rgba_to_rgb(h_first_frame)
            h_first_frame = h_first_frame.resize((WIDTH, HEIGHT))
        h_first_frame_rotated = rotate_cw(h_first_frame)
        print(f"Will replace V frame 0 with H first frame: {h_path}")

    # --- Inference (model sees "horizontal" orbit) ---
    print("Generating (H-model on rotated V-track)...")
    pipe.dit.fuse_vae_embedding_in_latents = True
    generated_video = pipe(
        prompt=args.prompt,
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        reference_image=reference_image_rotated,
        input_image=h_first_frame_rotated,
        control_video=condition_video_frames[:min_len],
        normal=normal_video_frames[:min_len],
        num_frames=min_len,
        height=HEIGHT,
        width=WIDTH,
        seed=args.seed,
        tiled=True,
        cfg_scale=args.cfg_scale,
    )

    # --- Rotate generated video back 90 CCW ---
    print("Rotating generated video 90 CCW back to vertical...")
    generated_video_back = [rotate_ccw(f) for f in generated_video]

    # --- Save: 4-panel [ref | condition | generated | albedo] (all in original vertical orientation) ---
    combined_frames = []
    for i in range(len(generated_video_back)):
        # Rotate condition/albedo back to vertical for comparison
        cond_orig = rotate_ccw(condition_video_frames[i])
        albedo_orig = rotate_ccw(albedo_video_frames[i])

        combined = Image.new('RGB', (WIDTH * 4, HEIGHT))
        combined.paste(reference_image.convert('RGB'), (0, 0))
        combined.paste(cond_orig.convert('RGB'), (WIDTH, 0))
        combined.paste(generated_video_back[i].convert('RGB'), (WIDTH * 2, 0))
        combined.paste(albedo_orig.convert('RGB'), (WIDTH * 3, 0))
        combined_frames.append(combined)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_video(combined_frames, args.output, fps=15, quality=5)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
