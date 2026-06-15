import torch
import os, sys
import random
import re
import subprocess
from glob import glob
from PIL import Image
from diffsynth import save_video, VideoData, load_state_dict
from diffsynth.pipelines.wan_video_new_14b_ref_drop_normal import WanVideoPipeline, ModelConfig

# === 配置参数 ===
RANDOM_SEED = 42
SAMPLE_COUNT = 10

# === 统一分辨率配置 ===
HEIGHT = 768
WIDTH = 768 * 2
NUM_FRAMES = 121
FPS = 15
QUALITY = 5
bake = False
skip = False

# 获取命令行起始索引 (可选)
start = int(sys.argv[1]) if len(sys.argv) > 1 else 0

OUTPUT_DIR = f"./outputs0324_hv120_3600_revise/" 
os.makedirs(OUTPUT_DIR, exist_ok=True)

def rgba_to_rgb(rgba_image, background_color=(255, 255, 255)):
    rgba = rgba_image.convert('RGBA')
    bg = Image.new('RGB', rgba.size, background_color)
    alpha = rgba.split()[-1]
    bg.paste(rgba.convert('RGB'), mask=alpha)
    return bg

# === 路径配置 ===
REF_IMAGE_ROOT = "/home/v-hanyue/blobmnt/gemini/"
VIDEO_ROOT = "/home/v-hanyue/blobmnt/gemini/trellis2/120hhvv/videos_curve/glb"
FIXED_PROMPT = "This is a 3D model"


# === 固定随机种子 ===
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
print(f"已设置随机种子: {RANDOM_SEED}")

# === 模型初始化 ===
print("正在加载模型...")
if not skip:
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

    model_ckpt1 = "/home/v-hanyue/blobmnt/workspace/DiffSynth-Studio/models/train/Wan2.2-Fun-A14B-Control_high_noise_lora_ref_position_normal_drop_revise512_1001_121hv_light_aug_61_512_imageref0322/step-3600.safetensors"
    model_ckpt2 = "/home/v-hanyue/blobmnt/workspace/DiffSynth-Studio/models/train/Wan2.2-Fun-A14B-Control_low_noise_lora_ref_position_normal_drop_revise512_1001_121hv_light_aug_61_512_boundrevise_imageref0322/step-3600.safetensors"
    
    modify_patch_embedding = True
    if modify_patch_embedding:
        print(f"正在加载DIT1模型权重: {model_ckpt1}")
        state_dict = load_state_dict(model_ckpt1)
        patch_embedding_state_dict1 = {
            'bias': state_dict['patch_embedding.bias'], 
            'weight': state_dict['patch_embedding.weight']
        }           
        pipe.dit.patch_embedding.load_state_dict(patch_embedding_state_dict1)
        
        print(f"正在加载DIT2模型权重: {model_ckpt2}")
        state_dict2 = load_state_dict(model_ckpt2)
        patch_embedding_state_dict2 = {
            'bias': state_dict2['patch_embedding.bias'], 
            'weight': state_dict2['patch_embedding.weight']
        }           
        pipe.dit2.patch_embedding.load_state_dict(patch_embedding_state_dict2)

    pipe.load_lora(pipe.dit, model_ckpt1, alpha=1)
    pipe.load_lora(pipe.dit2, model_ckpt2, alpha=1)
    pipe.enable_vram_management()
    print("模型加载完成。")
else:
    pipe = None

# === 辅助函数：从ref文件名提取编号（如 "001.png" → "001"）===
def extract_id_from_ref_filename(filename):
    """匹配格式: 001.png, 002.png 等纯数字编号"""
    match = re.search(r'^(\d+)\.png$', filename, re.IGNORECASE)
    return match.group(1) if match else None

# === 构建样本列表：遍历ref_image目录 ===
all_samples = []
ref_files = sorted(glob(os.path.join(REF_IMAGE_ROOT, "*.png")))
print(f"找到 {len(ref_files)} 个ref_image文件，开始验证配套视频...")

for ref_abs in ref_files:
    filename = os.path.basename(ref_abs)
    file_id = extract_id_from_ref_filename(filename)  # 如 "001"
    
    if file_id is None:
        print(f"⚠️ 无法提取编号，跳过: {filename}")
        continue
    
    # 构建视频目录: {VIDEO_ROOT}/{file_id}/  (注意: 不是 {hash}_1024)
    video_dir = os.path.join(VIDEO_ROOT, file_id)
    
    # 构建各视频路径
    paths = {
        "position": os.path.join(video_dir, "position.mp4"),
        "normal": os.path.join(video_dir, "normal.mp4"),
        "albedo": os.path.join(video_dir, "albedo.mp4"),
        "rgb": os.path.join(video_dir, "rgb.mp4"),
    }
    
    # 检查文件完整性
    missing = [f"{k}: {v}" for k, v in paths.items() if not os.path.exists(v)]
    if missing:
        print(f"⚠️ 文件缺失，跳过 {file_id}:")
        for m in missing:
            print(f"   - {m}")
        continue
    
    all_samples.append({
        "ref_image": ref_abs,
        "hash_id": file_id,           # 用编号作为唯一标识
        "alpha_path": paths["albedo"],
        "rgb_path": paths["rgb"],
        "depth_path": paths["position"],
        "normal_path": paths["normal"],
        "prompt": FIXED_PROMPT
    })

print(f"✅ 有效样本: {len(all_samples)} / {len(ref_files)}")
all_samples = all_samples[start:]
sample_size = len(all_samples)

# === 批量推理 ===
for idx, row in enumerate(all_samples):
    REF = row["ref_image"]
    INPUT_VIDEO_PATH = row["rgb_path"]
    alpha_VIDEO_PATH = row["alpha_path"]
    CONDITION_VIDEO_PATH = row["depth_path"]
    NORMAL_VIDEO_PATH = row["normal_path"]
    prompt = row["prompt"]
    hash_id = row["hash_id"]
    sha256 = hash_id
    suff = os.path.basename(REF).split('.')[0]
    output_name = f"{hash_id}_{suff}.mp4"
    OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, output_name)
    os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)
    
    if os.path.exists(OUTPUT_VIDEO_PATH):
        print(f"[{idx+1}/{sample_size}] 已存在，跳过: {output_name}")
        continue
        
    print(f"\n[{idx+1}/{sample_size}] 正在处理 {hash_id}:")
    print(f"  输出: {OUTPUT_VIDEO_PATH}")

    # 加载视频帧 (隔帧采样)
    def load_frames(video_path, total_frames=NUM_FRAMES, step=2):
        vd = VideoData(video_path, height=HEIGHT, width=WIDTH)
        return [vd[i] for i in range(0, min(total_frames, len(vd)), step)] + [vd[0]]
    
    input_video_frames = load_frames(INPUT_VIDEO_PATH)
    alpha_video_frames = load_frames(alpha_VIDEO_PATH)
    condition_video_frames = load_frames(CONDITION_VIDEO_PATH)
    normal_video_frames = load_frames(NORMAL_VIDEO_PATH)

    # 处理参考图: RGBA→RGB + 左右拼接
    reference_image = Image.open(REF).convert('RGBA')
    reference_image = rgba_to_rgb(reference_image)
    reference_image = reference_image.resize((WIDTH//2, HEIGHT))
    concatenated = Image.new('RGB', (WIDTH, HEIGHT))
    concatenated.paste(reference_image, (0, 0))
    concatenated.paste(reference_image, (WIDTH//2, 0))
    reference_image = concatenated

    # 生成视频
    generated_video = pipe(
        prompt=prompt,
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        reference_image=reference_image,
        control_video=condition_video_frames,
        normal=normal_video_frames,
        num_frames=61,
        height=HEIGHT,
        width=WIDTH,
        seed=RANDOM_SEED + idx,
        tiled=True,
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

    save_video(combined_frames, OUTPUT_VIDEO_PATH, fps=FPS, quality=QUALITY)
    print(f"✅ 已保存: {OUTPUT_VIDEO_PATH}")
    
    # 可选: texture baking
    if bake:
        cmd = [
            "python", "tex_bake_gen_hvflip_vsr_arg_mnt5.py",
            "--mr_video_path", OUTPUT_VIDEO_PATH,
            "--video_path", OUTPUT_VIDEO_PATH,
            "--output_dir", OUTPUT_DIR,
            "--resolution", "2048",
            "--sha256", sha256,
            "--usr_gt_mr", "--use_custom"
        ]
        subprocess.run(cmd, check=True)

print(f"\n🎉 全部完成！输出目录: {OUTPUT_DIR}")