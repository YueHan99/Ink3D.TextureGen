"""
Single-sample inference script (single-track h, 768x768)

Usage:
  cd OrbitVideoGen && export PYTHONPATH="$(pwd):${PYTHONPATH}"
  python tests/test_single_h.py --ref_image /path/to/ref.png --video_dir /path/to/video_dir/ [--output output.mp4]

  video_dir should contain: position.mp4, normal.mp4, albedo.mp4, rgb.mp4
"""
import torch
import os, sys, argparse
from PIL import Image
from diffsynth import save_video, VideoData, load_state_dict
from diffsynth.pipelines.wan_video_new_14b_ref_drop_normal import WanVideoPipeline, ModelConfig


def rgba_to_rgb(rgba_image, background_color=(0, 0, 0)):
    if rgba_image.mode != 'RGBA':
        return rgba_image.convert('RGB')
    bg = Image.new('RGB', rgba_image.size, background_color)
    alpha = rgba_image.split()[-1]
    bg.paste(rgba_image.convert('RGB'), mask=alpha)
    return bg


def main():
    parser = argparse.ArgumentParser(description="Single-sample inference (h, 768x768)")
    parser.add_argument("--ref_image", type=str, required=True, help="Path to reference image (png)")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing position.mp4, normal.mp4, albedo.mp4, rgb.mp4")
    parser.add_argument("--output", type=str, default="./output_h.mp4", help="Output video path")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--cfg_scale", type=float, default=6.0)
    parser.add_argument("--prompt", type=str, default="This is a 3D model")
    parser.add_argument("--models_base", type=str, required=True,
        help="Base path for pretrained models (PAI/..., Wan-AI/...)")
    parser.add_argument("--model_ckpt_high", type=str, required=True,
        help="Path to high-noise LoRA checkpoint")
    parser.add_argument("--model_ckpt_low", type=str, required=True,
        help="Path to low-noise LoRA checkpoint")
    parser.add_argument("--no_bg_remove", action="store_true", help="Skip background removal")
    args = parser.parse_args()

    HEIGHT, WIDTH = args.height, args.width
    NUM_FRAMES = args.num_frames

    # --- Load model ---
    print("Loading models...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda:0",
        model_configs=[
            ModelConfig(path=f"{args.models_base}/PAI/Wan2.2-Fun-A14B-Control/Wan2.2-Fun-A14B-Control/high_noise_model/diffusion_pytorch_model.safetensors", offload_device="cpu"),
            ModelConfig(path=f"{args.models_base}/PAI/Wan2.2-Fun-A14B-Control/Wan2.2-Fun-A14B-Control/low_noise_model/diffusion_pytorch_model.safetensors", offload_device="cpu"),
            ModelConfig(path=f"{args.models_base}/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(path=f"{args.models_base}/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth", offload_device="cpu"),
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

    # --- Load data ---
    def load_frames(video_path):
        vd = VideoData(video_path, height=HEIGHT, width=WIDTH)
        count = min(NUM_FRAMES, len(vd))
        frames = [vd[i] for i in range(count)]
        if len(frames) < NUM_FRAMES:
            frames.append(vd[0])
        return frames

    condition_video_frames = load_frames(os.path.join(args.video_dir, "position.mp4"))
    normal_video_frames = load_frames(os.path.join(args.video_dir, "normal.mp4"))
    albedo_video_frames = load_frames(os.path.join(args.video_dir, "albedo.mp4"))

    min_len = min(len(condition_video_frames), len(normal_video_frames), len(albedo_video_frames))

    reference_image = Image.open(args.ref_image).convert("RGBA")
    if not args.no_bg_remove:
        from bg_remover import remove_background, rgba_to_rgb_black
        reference_image = remove_background(reference_image)
        reference_image = rgba_to_rgb_black(reference_image)
    else:
        reference_image = rgba_to_rgb(reference_image)
    reference_image = reference_image.resize((WIDTH, HEIGHT))

    # --- Inference ---
    print("Generating...")
    generated_video = pipe(
        prompt=args.prompt,
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        reference_image=reference_image,
        control_video=condition_video_frames[:min_len],
        normal=normal_video_frames[:min_len],
        num_frames=min_len,
        height=HEIGHT,
        width=WIDTH,
        seed=args.seed,
        tiled=True,
        cfg_scale=args.cfg_scale,
    )

    # --- Save: 4-panel [ref | condition | generated | albedo] ---
    combined_frames = []
    for i in range(len(generated_video)):
        combined = Image.new('RGB', (WIDTH * 4, HEIGHT))
        combined.paste(reference_image.convert('RGB'), (0, 0))
        combined.paste(condition_video_frames[i].convert('RGB'), (WIDTH, 0))
        combined.paste(generated_video[i].convert('RGB'), (WIDTH * 2, 0))
        combined.paste(albedo_video_frames[i].convert('RGB'), (WIDTH * 3, 0))
        combined_frames.append(combined)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_video(combined_frames, args.output, fps=15, quality=5)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
