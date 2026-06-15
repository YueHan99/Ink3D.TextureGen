"""
Batch inference: v120 via H-model (load model once, run all cases)
"""
import torch
import os, sys, random
from PIL import Image
from diffsynth import save_video, VideoData, load_state_dict
from diffsynth.pipelines.wan_video_new_14b_ref_drop_normal import WanVideoPipeline, ModelConfig

HEIGHT, WIDTH, NUM_FRAMES = 768, 768, 121
SEED = 3
DATA_ROOT = "/home/v-hanyue/workspace/Ink3D/OrbitVideoGen/tests/example_data"
OUTDIR = "/home/v-hanyue/workspace/Ink3D/OrbitVideoGen/outputs_v_as_h"
os.makedirs(OUTDIR, exist_ok=True)

def rgba_to_rgb(img, bg=(255,255,255)):
    if img.mode != 'RGBA': return img.convert('RGB')
    b = Image.new('RGB', img.size, bg)
    b.paste(img.convert('RGB'), mask=img.split()[-1])
    return b

def rotate_cw(img):  return img.transpose(Image.ROTATE_270)
def rotate_ccw(img): return img.transpose(Image.ROTATE_90)

# Collect valid cases
cases = []
for name in sorted(os.listdir(DATA_ROOT)):
    d = os.path.join(DATA_ROOT, name)
    v120 = os.path.join(d, "v120")
    ref = os.path.join(d, "ref.png")
    if os.path.isfile(ref) and os.path.isdir(v120) \
       and os.path.isfile(os.path.join(v120, "position.mp4")) \
       and os.path.isfile(os.path.join(v120, "normal.mp4")) \
       and os.path.isfile(os.path.join(v120, "albedo.mp4")):
        cases.append(name)

# Shuffle
random.seed(42)
random.shuffle(cases)
print(f"Found {len(cases)} cases: {cases}")

# Load model once
print("Loading models...")
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16, device="cuda:0",
    model_configs=[
        ModelConfig(path="/home/v-hanyue/local_models/PAI/Wan2.2-Fun-A14B-Control/Wan2.2-Fun-A14B-Control/high_noise_model/diffusion_pytorch_model.safetensors", offload_device="cpu"),
        ModelConfig(path="/home/v-hanyue/local_models/PAI/Wan2.2-Fun-A14B-Control/Wan2.2-Fun-A14B-Control/low_noise_model/diffusion_pytorch_model.safetensors", offload_device="cpu"),
        ModelConfig(path="/home/v-hanyue/local_models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(path="/home/v-hanyue/local_models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth", offload_device="cpu"),
    ],
)
ckpt_h = "/home/v-hanyue/local_models/lora_ckpt/high_noise_step-3500.safetensors"
ckpt_l = "/home/v-hanyue/local_models/lora_ckpt/low_noise_step-3300.safetensors"
sd1 = load_state_dict(ckpt_h)
pipe.dit.patch_embedding.load_state_dict({'bias': sd1['patch_embedding.bias'], 'weight': sd1['patch_embedding.weight']})
sd2 = load_state_dict(ckpt_l)
pipe.dit2.patch_embedding.load_state_dict({'bias': sd2['patch_embedding.bias'], 'weight': sd2['patch_embedding.weight']})
pipe.load_lora(pipe.dit, ckpt_h, alpha=1)
pipe.load_lora(pipe.dit2, ckpt_l, alpha=1)
pipe.enable_vram_management()
print("Model loaded.")

# Run all cases
for idx, name in enumerate(cases):
    out_path = os.path.join(OUTDIR, f"output_v_as_h_{name}.mp4")
    if os.path.exists(out_path):
        print(f"[{idx+1}/{len(cases)}] SKIP {name} (exists)")
        continue

    print(f"\n[{idx+1}/{len(cases)}] Processing {name}...")
    try:
        v120 = os.path.join(DATA_ROOT, name, "v120")
        def load_rot(vp):
            vd = VideoData(vp, height=HEIGHT, width=WIDTH)
            count = min(NUM_FRAMES, len(vd))
            return [rotate_cw(vd[i]) for i in range(count)] + [rotate_cw(vd[0])]

        cond = load_rot(os.path.join(v120, "position.mp4"))
        norm = load_rot(os.path.join(v120, "normal.mp4"))
        alb  = load_rot(os.path.join(v120, "albedo.mp4"))
        min_len = min(len(cond), len(norm), len(alb))

        ref = rgba_to_rgb(Image.open(os.path.join(DATA_ROOT, name, "ref.png")))
        ref = ref.resize((WIDTH, HEIGHT))
        ref_rot = rotate_cw(ref)

        gen = pipe(
            prompt="This is a 3D model",
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            reference_image=ref_rot,
            control_video=cond[:min_len],
            normal=norm[:min_len],
            num_frames=min_len,
            height=HEIGHT, width=WIDTH,
            seed=SEED + idx,
            tiled=True, cfg_scale=6.0,
        )

        gen_back = [rotate_ccw(f) for f in gen]
        combined = []
        for i in range(len(gen_back)):
            c_orig = rotate_ccw(cond[i])
            a_orig = rotate_ccw(alb[i])
            frame = Image.new('RGB', (WIDTH*4, HEIGHT))
            frame.paste(ref.convert('RGB'), (0,0))
            frame.paste(c_orig.convert('RGB'), (WIDTH,0))
            frame.paste(gen_back[i].convert('RGB'), (WIDTH*2,0))
            frame.paste(a_orig.convert('RGB'), (WIDTH*3,0))
            combined.append(frame)

        save_video(combined, out_path, fps=15, quality=5)
        print(f"  Saved: {out_path}")
    except Exception as e:
        print(f"  FAILED {name}: {e}")
        import traceback; traceback.print_exc()
        continue

print(f"\nALL DONE. Outputs in {OUTDIR}")
