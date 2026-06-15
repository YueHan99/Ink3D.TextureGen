import torch
import os, sys
import random
import re
import subprocess
import gc
from glob import glob
from PIL import Image
from diffsynth import save_video, VideoData, load_state_dict
from diffsynth.pipelines.wan_video_new_14b_ref_drop_normal import WanVideoPipeline, ModelConfig

# === 配置参数 ===
RANDOM_SEED = 3
HEIGHT = 768
WIDTH = 768
NUM_FRAMES = 121
FPS = 15
QUALITY = 5
bake = False
skip = False

# 获取命令行起始索引
start = int(sys.argv[1]) if len(sys.argv) > 1 else 0

OUTPUT_DIR = f"./outputs0324_h120_3k_cfg6/" 
os.makedirs(OUTPUT_DIR, exist_ok=True)

def rgba_to_rgb(rgba_image, background_color=(255, 255, 255)):
    if rgba_image.mode != 'RGBA':
        return rgba_image.convert('RGB')
    bg = Image.new('RGB', rgba_image.size, background_color)
    alpha = rgba_image.split()[-1]
    bg.paste(rgba_image.convert('RGB'), mask=alpha)
    return bg

# === 路径配置（请根据实际路径修改）===
REF_IMAGE_ROOT = "/home/v-hanyue/blobmnt/gemini/"
VIDEO_ROOT = "/home/v-hanyue/blobmnt/gemini/trellis2/120hh/videos_curve/glb"
FIXED_PROMPT = "This is a 3D model"

# === 固定随机种子 ===
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
print(f"🎲 已设置随机种子: {RANDOM_SEED}")

# === 模型初始化 ===
print("🚀 正在加载模型...")
pipe = None
if not skip:
    try:
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

        model_ckpt1 = "/home/v-hanyue/blobmnt/workspace/DiffSynth-Studio/models/train/Wan2.2-Fun-A14B-Control_high_noise_lora_ref_position_normal_drop_revise512_1001_121hv_light_aug_61_512_h_imageref0322/step-3500.safetensors"
        model_ckpt2 = "/home/v-hanyue/blobmnt/workspace/DiffSynth-Studio/models/train/Wan2.2-Fun-A14B-Control_low_noise_lora_ref_position_normal_drop_revise512_1001_121hv_light_aug_61_512_boundrevise_512_h_imageref0322/step-3300.safetensors"
        
        # 检查权重文件
        for ckpt in [model_ckpt1, model_ckpt2]:
            if not os.path.exists(ckpt):
                raise FileNotFoundError(f"权重文件不存在: {ckpt}")
        
        print(f"⚙️ 正在注入 Patch Embedding 权重...")
        state_dict = load_state_dict(model_ckpt1)
        pipe.dit.patch_embedding.load_state_dict({
            'bias': state_dict['patch_embedding.bias'], 
            'weight': state_dict['patch_embedding.weight']
        })
        
        state_dict2 = load_state_dict(model_ckpt2)
        pipe.dit2.patch_embedding.load_state_dict({
            'bias': state_dict2['patch_embedding.bias'], 
            'weight': state_dict2['patch_embedding.weight']
        })

        print("🧩 正在加载 LoRA...")
        pipe.load_lora(pipe.dit, model_ckpt1, alpha=1)
        pipe.load_lora(pipe.dit2, model_ckpt2, alpha=1)
        
        pipe.enable_vram_management()
        print("✅ 模型加载完成。")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# === 辅助函数：从ref文件名提取数字编号 ===
def extract_id_from_ref_filename(filename):
    """匹配格式: 001.png, 1.png, 0001.png 等"""
    match = re.search(r'^(\d+)\.png$', filename, re.IGNORECASE)
    if match:
        return match.group(1)
    # 备用：匹配文件名末尾的数字
    match = re.search(r'(\d+)\.png$', filename, re.IGNORECASE)
    return match.group(1) if match else None

# === 构建样本列表：遍历ref_image目录 ===
all_samples = []
ref_files = sorted([f for f in glob(os.path.join(REF_IMAGE_ROOT, "*.png")) if not os.path.basename(f).startswith('.')])
print(f"🔍 找到 {len(ref_files)} 个候选图片，开始验证配套视频...")

for ref_abs in ref_files:
    filename = os.path.basename(ref_abs)
    file_id = extract_id_from_ref_filename(filename)
    
    if file_id is None:
        print(f"⚠️ 无法提取编号，跳过: {filename}")
        continue
    
    # 构建视频目录: {VIDEO_ROOT}/{file_id}/
    video_dir = os.path.join(VIDEO_ROOT, file_id)
    
    if not os.path.isdir(video_dir):
        continue
    
    # 构建各视频路径
    paths = {
        "position": os.path.join(video_dir, "position.mp4"),
        "normal": os.path.join(video_dir, "normal.mp4"),
        "albedo": os.path.join(video_dir, "albedo.mp4"),
        "rgb": os.path.join(video_dir, "rgb.mp4"),
    }
    
    # 检查文件完整性
    missing = [k for k, v in paths.items() if not os.path.exists(v)]
    if missing:
        # 可选：打印缺失详情
        # print(f"⚠️ 跳过 {file_id}: 缺失 {missing}")
        continue
    
    all_samples.append({
        "ref_image": ref_abs,
        "hash_id": file_id,
        "alpha_path": paths["albedo"],
        "rgb_path": paths["rgb"],
        "depth_path": paths["position"],
        "normal_path": paths["normal"],
        "prompt": FIXED_PROMPT
    })

print(f"✅ 有效样本: {len(all_samples)} / {len(ref_files)}")

if len(all_samples) == 0:
    print("❌ 没有发现任何有效样本，请检查路径配置。")
    sys.exit(0)

# 切片处理（支持断点续跑）
all_samples = all_samples[start:]
sample_size = len(all_samples)
print(f"📋 本次将处理 {sample_size} 个样本 (起始索引: {start})")

# === 批量推理 ===
for idx, row in enumerate(all_samples):
    REF = row["ref_image"]
    INPUT_VIDEO_PATH = row["rgb_path"]
    alpha_VIDEO_PATH = row["alpha_path"]
    CONDITION_VIDEO_PATH = row["depth_path"]
    NORMAL_VIDEO_PATH = row["normal_path"]
    prompt = row["prompt"]
    file_id = row["hash_id"]
    
    suff = os.path.basename(REF).split('.')[0]
    output_name = f"{file_id}_{suff}.mp4"
    OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, output_name)
    
    # 跳过已完成的
    if os.path.exists(OUTPUT_VIDEO_PATH):
        print(f"[{idx+1}/{sample_size}] ⏭️  已存在，跳过: {output_name}")
        continue
        
    print(f"\n[{idx+1}/{sample_size}] 🎬 正在处理 ID: {file_id}")
    
    try:
        # 加载视频帧
        def load_frames_safe(video_path, total_frames=NUM_FRAMES):
            vd = VideoData(video_path, height=HEIGHT, width=WIDTH)
            count = min(total_frames, len(vd))
            return [vd[i] for i in range(count)] + [vd[0]]

        input_video_frames = load_frames_safe(INPUT_VIDEO_PATH)
        alpha_video_frames = load_frames_safe(alpha_VIDEO_PATH)
        condition_video_frames = load_frames_safe(CONDITION_VIDEO_PATH)
        normal_video_frames = load_frames_safe(NORMAL_VIDEO_PATH)

        # 确保帧数一致
        min_len = min(len(input_video_frames), len(alpha_video_frames), 
                      len(condition_video_frames), len(normal_video_frames))
        if min_len < 121:
            raise ValueError(f"视频帧数过少: {min_len}")
        
        input_video_frames = input_video_frames[:min_len]
        alpha_video_frames = alpha_video_frames[:min_len]
        condition_video_frames = condition_video_frames[:min_len]
        normal_video_frames = normal_video_frames[:min_len]

        # 处理参考图 (768x768, 不拼接)
        reference_image = Image.open(REF)
        reference_image = rgba_to_rgb(reference_image)
        reference_image = reference_image.resize((WIDTH, HEIGHT))

        # 生成视频
        print("   🤖 生成中...")
        generated_video = pipe(
            prompt=prompt,
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            reference_image=reference_image,
            control_video=condition_video_frames,
            normal=normal_video_frames,
            num_frames=min_len,  # 动态匹配实际帧数
            height=HEIGHT,
            width=WIDTH,
            seed=RANDOM_SEED + idx,
            tiled=True,
            cfg_scale=6.0
        )
        
        # 四面板拼接: [ref | condition | generated | alpha]
        combined_frames = []
        for i in range(len(generated_video)):
            combined = Image.new('RGB', (WIDTH * 4, HEIGHT))
            combined.paste(reference_image.convert('RGB'), (0, 0))
            combined.paste(condition_video_frames[i].convert('RGB'), (WIDTH, 0))
            combined.paste(generated_video[i].convert('RGB'), (WIDTH * 2, 0))
            combined.paste(alpha_video_frames[i].convert('RGB'), (WIDTH * 3, 0))
            combined_frames.append(combined)

        # 保存视频
        print("   💾 保存中...")
        save_video(combined_frames, OUTPUT_VIDEO_PATH, fps=FPS, quality=QUALITY)
        print(f"   ✅ 成功: {OUTPUT_VIDEO_PATH}")
        
        # Texture Baking (可选)
        if bake:
            print("   🔥 开始烘焙...")
            cmd = [
                "python", "tex_bake_gen_hv_vsr_arg.py",
                "--mr_video_path", OUTPUT_VIDEO_PATH,
                "--video_path", OUTPUT_VIDEO_PATH,
                "--output_dir", OUTPUT_DIR + "hv",
                "--resolution", "2048",
                "--sha256", file_id,
                "--usr_gt_mr", "--use_custom"
            ]
            subprocess.run(cmd, check=True)

    except Exception as e:
        print(f"   ❌ 处理失败 {file_id}: {e}")
        # import traceback
        # traceback.print_exc()
        continue
    # finally:
    #     # === 关键：清理显存 ===
    #     for var in [input_video_frames, alpha_video_frames, condition_video_frames, 
    #                 normal_video_frames, generated_video, combined_frames, reference_image]:
    #         if var is not None:
    #             del var
    #     gc.collect()
    #     torch.cuda.empty_cache()

print(f"\n🎉 全部完成！输出目录: {OUTPUT_DIR}")