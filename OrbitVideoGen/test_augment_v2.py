"""
测试 v2 augment_image 和 frontal frame sampling 的可视化脚本。
使用图片序列（RGBA PNG）而非视频。

用法:
    python test_augment_v2.py --sample_dir <images_curve/000-000/uid> --output_dir <output_dir> [--num_samples 20]

输出:
    output_dir/
        augment_00.png    # 左=原图(RGBA on black bg), 右=augment后
        ...
        sampling_histogram.png   # 帧采样分布直方图
        sampled_frames.png       # 被采样到的帧拼接
"""
import os
import sys
import argparse
import random
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
from diffsynth.trainers.utils_ref_normal_albedo_light_aug_61_revise_imageref_h0322_v2 import augment_image


def load_image_sequence(directory, prefix, start_idx=1, max_frames=120):
    """加载 prefix_NNNN.png 图片序列"""
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
    """RGBA -> RGB on black background"""
    if image.mode == 'RGBA':
        bg = Image.new('RGB', image.size, (0, 0, 0))
        bg.paste(image, (0, 0), image.split()[3])
        return bg
    return image.convert('RGB')


def test_augment(frames, output_dir, num_samples=20):
    """对随机帧做 augment，保存 before/after 对比图"""
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_samples):
        idx = random.randint(0, len(frames) - 1)
        original = frames[idx].copy()

        augmented = augment_image(original)

        # 拼接: 原图(on black) | augmented
        orig_rgb = rgba_on_black(original)
        w, h = orig_rgb.size
        canvas = Image.new('RGB', (w * 2 + 4, h), (128, 128, 128))

        orig_labeled = add_label(orig_rgb.copy(), f"Original (frame {idx})")
        aug_labeled = add_label(augmented.copy(), "Augmented")

        canvas.paste(orig_labeled, (0, 0))
        canvas.paste(aug_labeled, (w + 4, 0))

        save_path = os.path.join(output_dir, f"augment_{i:02d}.png")
        canvas.save(save_path)
        print(f"Saved {save_path}")


def test_sampling_distribution(n_total, output_dir, n_trials=5000):
    """测试 frontal frame sampling 的分布，画直方图"""
    os.makedirs(output_dir, exist_ok=True)

    n_front = 30
    front_indices = list(range(min(n_front, n_total)))
    back_indices = list(range(max(0, n_total - n_front), n_total))
    candidates = sorted(set(front_indices + back_indices))
    n_ref = len(candidates)

    counts = np.zeros(n_ref, dtype=int)
    for _ in range(n_trials):
        weights_ref = []
        for i in range(n_ref):
            dist_to_end = min(i, n_ref - 1 - i)
            angle = (dist_to_end / max(n_ref // 2, 1)) * (math.pi / 2)
            w = math.cos(angle) + 0.1
            weights_ref.append(w)
        total_w = sum(weights_ref)
        probs_ref = [w / total_w for w in weights_ref]
        chosen = np.random.choice(n_ref, p=probs_ref)
        counts[chosen] += 1

    bar_w = 8
    max_h = 300
    margin = 40
    img_w = n_ref * bar_w + margin * 2
    img_h = max_h + margin * 2 + 20
    canvas = Image.new('RGB', (img_w, img_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    max_count = max(counts) if max(counts) > 0 else 1
    for i, c in enumerate(counts):
        bar_h = int(c / max_count * max_h)
        x0 = margin + i * bar_w
        y0 = margin + max_h - bar_h
        x1 = x0 + bar_w - 1
        y1 = margin + max_h
        dist = min(i, n_ref - 1 - i)
        ratio = dist / max(n_ref // 2, 1)
        r = int(200 * ratio)
        g = int(200 * (1 - ratio))
        draw.rectangle([x0, y0, x1, y1], fill=(r, g, 50))

    font = ImageFont.load_default()
    draw.text((margin, 5), f"Frontal Frame Sampling (n_total={n_total}, {n_trials} trials)", fill=(0, 0, 0), font=font)
    draw.text((margin, margin + max_h + 5), "<- front(0)    candidate index    back(end) ->", fill=(0, 0, 0), font=font)

    mid = n_ref // 2
    draw.text((margin + mid * bar_w - 20, margin + max_h + 18), "side(90deg)", fill=(200, 0, 0), font=font)
    draw.text((margin, margin + max_h + 18), "front(0deg)", fill=(0, 150, 0), font=font)
    draw.text((margin + (n_ref - 5) * bar_w, margin + max_h + 18), "front(0deg)", fill=(0, 150, 0), font=font)

    save_path = os.path.join(output_dir, "sampling_histogram.png")
    canvas.save(save_path)
    print(f"Saved {save_path}")

    probs_pct = counts / n_trials * 100
    print(f"\nSampling distribution (top-5 most sampled):")
    top5 = np.argsort(-counts)[:5]
    for idx in top5:
        real_frame = candidates[idx]
        print(f"  candidate[{idx}] -> frame {real_frame}: {probs_pct[idx]:.1f}%")
    print(f"\nFront 5 total: {probs_pct[:5].sum():.1f}%")
    print(f"Back 5 total: {probs_pct[-5:].sum():.1f}%")
    print(f"Middle 10 total: {probs_pct[n_ref//2-5:n_ref//2+5].sum():.1f}%")


def test_sampled_frames_visual(frames, output_dir, n_samples=16):
    """可视化实际被采样到的帧"""
    os.makedirs(output_dir, exist_ok=True)

    n_total = len(frames)
    ref_data = frames[:30] + frames[-30:]
    n_ref = len(ref_data)

    sampled = []
    for _ in range(n_samples):
        weights_ref = []
        for i in range(n_ref):
            dist_to_end = min(i, n_ref - 1 - i)
            angle = (dist_to_end / max(n_ref // 2, 1)) * (math.pi / 2)
            w = math.cos(angle) + 0.1
            weights_ref.append(w)
        total_w = sum(weights_ref)
        probs_ref = [w / total_w for w in weights_ref]
        chosen = np.random.choice(n_ref, p=probs_ref)
        sampled.append((chosen, ref_data[chosen]))

    cols = 4
    rows = (n_samples + cols - 1) // cols
    thumb_size = 256
    canvas = Image.new('RGB', (cols * thumb_size, rows * thumb_size), (64, 64, 64))

    for i, (idx, frame) in enumerate(sampled):
        r, c = i // cols, i % cols
        thumb = rgba_on_black(frame).resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)

        if idx < 30:
            label = f"front[{idx}]"
        else:
            real_frame = n_total - 30 + (idx - 30)
            label = f"back[{real_frame}]"

        thumb = add_label(thumb, label)
        canvas.paste(thumb, (c * thumb_size, r * thumb_size))

    save_path = os.path.join(output_dir, "sampled_frames.png")
    canvas.save(save_path)
    print(f"Saved {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", type=str, required=True,
                        help="Path to sample image dir, e.g. images_curve/000-000/uid")
    parser.add_argument("--output_dir", type=str,
                        default="/home/v-hanyue/workspace/Ink3D/test_augment_v2",
                        help="Output directory")
    parser.add_argument("--num_samples", type=int, default=20)
    args = parser.parse_args()

    print(f"Loading render images from: {args.sample_dir}")
    frames = load_image_sequence(args.sample_dir, 'render', start_idx=1)
    print(f"Loaded {len(frames)} frames, size={frames[0].size}, mode={frames[0].mode}")

    print(f"\n=== Test 1: augment_image before/after ===")
    test_augment(frames, args.output_dir, args.num_samples)

    print(f"\n=== Test 2: sampling distribution histogram ===")
    test_sampling_distribution(len(frames), args.output_dir)

    print(f"\n=== Test 3: sampled frames visualization ===")
    test_sampled_frames_visual(frames, args.output_dir)

    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
