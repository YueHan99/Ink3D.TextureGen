"""
测试 v2 VideoDataset 完整数据加载的可视化。
对一个样本的所有通道做 snapshot: video(render), control_video(position), normal, albedo, ref_image。

用法:
    python test_dataset_snapshot.py --sample_dir <images_curve/000-xxx/uid> --output_dir <output_dir>
"""
import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
from diffsynth.trainers.utils_ref_normal_albedo_light_aug_61_revise_imageref_h0322_v2 import augment_image


def load_image_sequence(directory, prefix, start_idx=1, max_frames=120):
    frames = []
    for i in range(start_idx, start_idx + max_frames):
        path = os.path.join(directory, f'{prefix}_{i:04d}.png')
        if not os.path.exists(path):
            break
        frames.append(Image.open(path))
    return frames


def add_label(image, text, position=(5, 5)):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    bbox = draw.textbbox(position, text, font=font)
    draw.rectangle(bbox, fill=(0, 0, 0))
    draw.text(position, text, fill=(255, 255, 255), font=font)
    return image


def rgba_on_black(image):
    if image.mode == 'RGBA':
        bg = Image.new('RGB', image.size, (0, 0, 0))
        bg.paste(image, (0, 0), image.split()[3])
        return bg
    return image.convert('RGB')


def extract_alpha_vis(image):
    """将 alpha 通道可视化为灰度图（白=不透明, 黑=透明）"""
    if image.mode == 'RGBA':
        alpha = image.split()[3]
        return alpha.convert('RGB')
    else:
        # 没有 alpha，返回全白
        return Image.new('RGB', image.size, (255, 255, 255))


def make_grid(frames, indices, thumb_size=256, cols=8):
    """从 frames 列表中选取 indices 帧，拼成 grid"""
    n = len(indices)
    rows = (n + cols - 1) // cols
    canvas = Image.new('RGB', (cols * thumb_size, rows * thumb_size), (32, 32, 32))
    for i, idx in enumerate(indices):
        r, c = i // cols, i % cols
        frame = frames[idx]
        thumb = rgba_on_black(frame).resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        thumb = add_label(thumb, f"#{idx}")
        canvas.paste(thumb, (c * thumb_size, r * thumb_size))
    return canvas


def make_grid_custom(frames, indices, thumb_size=256, cols=8, transform=None, label_prefix=""):
    """通用 grid，支持自定义 transform"""
    n = len(indices)
    rows = (n + cols - 1) // cols
    canvas = Image.new('RGB', (cols * thumb_size, rows * thumb_size), (32, 32, 32))
    for i, idx in enumerate(indices):
        r, c = i // cols, i % cols
        frame = frames[idx]
        if transform:
            frame = transform(frame)
        else:
            frame = rgba_on_black(frame)
        thumb = frame.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        thumb = add_label(thumb, f"{label_prefix}#{idx}")
        canvas.paste(thumb, (c * thumb_size, r * thumb_size))
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", type=str, required=True,
                        help="Path to sample image dir, e.g. images_curve/000-001/uid")
    parser.add_argument("--output_dir", type=str,
                        default="/home/v-hanyue/workspace/Ink3D/test_dataset_snapshot")
    parser.add_argument("--thumb_size", type=int, default=256)
    parser.add_argument("--cols", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sample_dir = args.sample_dir
    thumb = args.thumb_size
    cols = args.cols

    # 选取要展示的帧索引: 均匀抽 16 帧
    sample_indices = list(range(0, 120, 120 // 16))[:16]

    # ========== 1. render (对应原 rgb.mp4 / video) ==========
    print("Loading render...")
    render_frames = load_image_sequence(sample_dir, 'render')
    print(f"  {len(render_frames)} frames, mode={render_frames[0].mode}")

    grid = make_grid(render_frames, sample_indices, thumb, cols)
    grid = add_label(grid, "RENDER (video)", (5, 5))
    grid.save(os.path.join(args.output_dir, "01_render.png"))
    print(f"  Saved 01_render.png")

    # ========== 2. render alpha 通道 ==========
    grid_alpha = make_grid_custom(render_frames, sample_indices, thumb, cols,
                                   transform=extract_alpha_vis, label_prefix="a")
    grid_alpha = add_label(grid_alpha, "RENDER ALPHA", (5, 5))
    grid_alpha.save(os.path.join(args.output_dir, "02_render_alpha.png"))
    print(f"  Saved 02_render_alpha.png")

    # ========== 3. position (对应 control_video) ==========
    print("Loading position...")
    position_frames = load_image_sequence(sample_dir, 'position')
    print(f"  {len(position_frames)} frames, mode={position_frames[0].mode}")

    grid = make_grid(position_frames, sample_indices, thumb, cols)
    grid = add_label(grid, "POSITION (control_video)", (5, 5))
    grid.save(os.path.join(args.output_dir, "03_position.png"))
    print(f"  Saved 03_position.png")

    # ========== 4. normal ==========
    print("Loading normal...")
    normal_frames = load_image_sequence(sample_dir, 'normal')
    print(f"  {len(normal_frames)} frames, mode={normal_frames[0].mode}")

    grid = make_grid(normal_frames, sample_indices, thumb, cols)
    grid = add_label(grid, "NORMAL", (5, 5))
    grid.save(os.path.join(args.output_dir, "04_normal.png"))
    print(f"  Saved 04_normal.png")

    # ========== 5. albedo ==========
    print("Loading albedo...")
    albedo_frames = load_image_sequence(sample_dir, 'albedo')
    print(f"  {len(albedo_frames)} frames, mode={albedo_frames[0].mode}")

    grid = make_grid(albedo_frames, sample_indices, thumb, cols)
    grid = add_label(grid, "ALBEDO", (5, 5))
    grid.save(os.path.join(args.output_dir, "05_albedo.png"))
    print(f"  Saved 05_albedo.png")

    # ========== 6. ref image augment ==========
    print("Testing ref image augment...")
    # 用 render 帧模拟 ref image
    ref_augmented = []
    import random
    for i in range(8):
        idx = random.randint(0, len(render_frames) - 1)
        orig = render_frames[idx].copy()
        aug = augment_image(orig)
        ref_augmented.append((idx, orig, aug))

    # 拼 2 行: 上面原图, 下面 augment
    canvas = Image.new('RGB', (8 * thumb, 2 * thumb), (32, 32, 32))
    for i, (idx, orig, aug) in enumerate(ref_augmented):
        orig_t = rgba_on_black(orig).resize((thumb, thumb), Image.Resampling.LANCZOS)
        aug_t = aug.resize((thumb, thumb), Image.Resampling.LANCZOS)
        orig_t = add_label(orig_t, f"orig #{idx}")
        aug_t = add_label(aug_t, f"aug #{idx}")
        canvas.paste(orig_t, (i * thumb, 0))
        canvas.paste(aug_t, (i * thumb, thumb))
    canvas = add_label(canvas, "REF IMAGE: top=original, bottom=augmented", (5, 5))
    canvas.save(os.path.join(args.output_dir, "06_ref_augment.png"))
    print(f"  Saved 06_ref_augment.png")

    # ========== 7. 综合 snapshot: 同一帧的所有通道并排 ==========
    print("Making per-frame multi-channel snapshot...")
    snapshot_indices = [0, 15, 30, 60, 90, 119]
    n_channels = 5  # render, alpha, position, normal, albedo
    canvas = Image.new('RGB', (n_channels * thumb, len(snapshot_indices) * thumb), (32, 32, 32))

    for row, idx in enumerate(snapshot_indices):
        # render
        t = rgba_on_black(render_frames[idx]).resize((thumb, thumb), Image.Resampling.LANCZOS)
        t = add_label(t, f"render #{idx}")
        canvas.paste(t, (0, row * thumb))

        # alpha
        t = extract_alpha_vis(render_frames[idx]).resize((thumb, thumb), Image.Resampling.LANCZOS)
        t = add_label(t, f"alpha #{idx}")
        canvas.paste(t, (thumb, row * thumb))

        # position
        t = rgba_on_black(position_frames[idx]).resize((thumb, thumb), Image.Resampling.LANCZOS)
        t = add_label(t, f"pos #{idx}")
        canvas.paste(t, (2 * thumb, row * thumb))

        # normal
        t = rgba_on_black(normal_frames[idx]).resize((thumb, thumb), Image.Resampling.LANCZOS)
        t = add_label(t, f"nrm #{idx}")
        canvas.paste(t, (3 * thumb, row * thumb))

        # albedo
        t = rgba_on_black(albedo_frames[idx]).resize((thumb, thumb), Image.Resampling.LANCZOS)
        t = add_label(t, f"alb #{idx}")
        canvas.paste(t, (4 * thumb, row * thumb))

    canvas = add_label(canvas, "render | alpha | position | normal | albedo", (5, 5))
    canvas.save(os.path.join(args.output_dir, "07_multichannel_snapshot.png"))
    print(f"  Saved 07_multichannel_snapshot.png")

    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
