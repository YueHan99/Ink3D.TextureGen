"""
测试 v2 ref image 三源采样逻辑的可视化。
对指定样本目录，检查三个源的可用性，多次采样并保存原图+augment结果。

用法:
    python test_ref_sources.py --sample_dir <videos_curve/000-xxx/uid>

    sample_dir 是 videos_curve 下的样本目录，例如:
    /home/v-hanyue/blobmnt/objaverse_60k_120h000/videos_curve/000-000/00d1cb5aa82745228a3b764c97f867de
"""
import os
import sys
import math
import random
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from diffsynth.trainers.utils_ref_normal_albedo_light_aug_61_revise_imageref_h0322_v2 import augment_image


def rgba_on_black(image):
    if image.mode == 'RGBA':
        bg = Image.new('RGB', image.size, (0, 0, 0))
        bg.paste(image, (0, 0), image.split()[3])
        return bg
    return image.convert('RGB')


def add_label(image, text, position=(5, 5)):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    bbox = draw.textbbox(position, text, font=font)
    draw.rectangle(bbox, fill=(0, 0, 0))
    draw.text(position, text, fill=(255, 255, 255), font=font)
    return image


def cos_weighted_sample(candidate_indices):
    n = len(candidate_indices)
    if n <= 1:
        return candidate_indices[0]
    weights = []
    for i in range(n):
        dist_to_end = min(i, n - 1 - i)
        angle = (dist_to_end / max(n // 2, 1)) * (math.pi / 2)
        w = math.cos(angle) + 0.1
        weights.append(w)
    total_w = sum(weights)
    probs = [w / total_w for w in weights]
    chosen_pos = np.random.choice(n, p=probs)
    return candidate_indices[chosen_pos]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", type=str, required=True,
                        help="videos_curve sample dir, e.g. .../videos_curve/000-000/uid")
    parser.add_argument("--output_dir", type=str,
                        default="/home/v-hanyue/workspace/Ink3D/test_ref_sources")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples for source distribution stats")
    parser.add_argument("--thumb_size", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sample_dir = args.sample_dir
    thumb = args.thumb_size

    # ========== 1. 检查三个源的可用性 ==========
    ref_dir_A = sample_dir.replace('videos_curve/', 'images_curve/')
    ref_dir_B = sample_dir.replace('objaverse_60k_120h/videos_curve/', 'objavers_ref_random2/')
    ref_dir_C = sample_dir.replace(
        'objaverse_60k_120h/videos_curve/',
        'objaverse_60k_1001_customh_3pointlight/images_curve/'
    )

    has_A = os.path.exists(os.path.join(ref_dir_A, 'render_0001.png'))
    has_B = os.path.exists(os.path.join(ref_dir_B, '000.png'))
    has_C = os.path.exists(os.path.join(ref_dir_C, 'render_0000.png'))

    print("=" * 60)
    print("Ref image source availability:")
    print(f"  A (120h images_curve, 60%): {ref_dir_A}")
    print(f"    exists: {has_A}")
    print(f"  B (ref_random2, 20%):       {ref_dir_B}")
    print(f"    exists: {has_B}")
    print(f"  C (3pointlight, 20%):       {ref_dir_C}")
    print(f"    exists: {has_C}")
    print("=" * 60)

    # Count frames per source
    total_A = 0
    if has_A:
        for i in range(1, 121):
            if os.path.exists(os.path.join(ref_dir_A, f'render_{i:04d}.png')):
                total_A += 1
            else:
                break
        print(f"  Source A: {total_A} frames (1-indexed)")

    total_B = 0
    if has_B:
        for i in range(8):
            if os.path.exists(os.path.join(ref_dir_B, f'{i:03d}.png')):
                total_B += 1
        print(f"  Source B: {total_B} frames")

    total_C = 0
    if has_C:
        for i in range(30):
            if os.path.exists(os.path.join(ref_dir_C, f'render_{i:04d}.png')):
                total_C += 1
            else:
                break
        print(f"  Source C: {total_C} frames (0-indexed)")

    # ========== 2. 模拟采样分布 ==========
    sources_list = []
    if has_A: sources_list.append(('A', 0.6))
    if has_B: sources_list.append(('B', 0.2))
    if has_C: sources_list.append(('C', 0.2))

    if len(sources_list) == 0:
        print("ERROR: No ref source available!")
        return

    src_names = [s[0] for s in sources_list]
    src_weights = np.array([s[1] for s in sources_list])
    src_weights = src_weights / src_weights.sum()

    print(f"\nNormalized weights: {dict(zip(src_names, src_weights))}")

    # 采样统计
    source_counter = Counter()
    frame_counter_A = Counter()
    frame_counter_C = Counter()

    for _ in range(args.num_samples):
        chosen = src_names[np.random.choice(len(src_names), p=src_weights)]
        source_counter[chosen] += 1

        if chosen == 'A' and total_A > 0:
            front = list(range(1, min(31, total_A + 1)))
            back = list(range(max(1, total_A - 29), total_A + 1))
            cands = front + [i for i in back if i not in front]
            idx = cos_weighted_sample(cands)
            frame_counter_A[idx] += 1
        elif chosen == 'C' and total_C > 0:
            cands = list(range(total_C))
            idx = cos_weighted_sample(cands)
            frame_counter_C[idx] += 1

    print(f"\nSource distribution ({args.num_samples} samples):")
    for name in ['A', 'B', 'C']:
        cnt = source_counter.get(name, 0)
        print(f"  {name}: {cnt} ({cnt/args.num_samples*100:.1f}%)")

    # ========== 3. 可视化: 每个源展示几帧原图 + augment ==========
    print("\nGenerating visualization...")
    rows = []  # list of (label, original_img, augmented_img)

    # Source A: 展示6帧
    if has_A and total_A > 0:
        front = list(range(1, min(31, total_A + 1)))
        back = list(range(max(1, total_A - 29), total_A + 1))
        cands = front + [i for i in back if i not in front]
        for _ in range(6):
            idx = cos_weighted_sample(cands)
            path = os.path.join(ref_dir_A, f'render_{idx:04d}.png')
            img = Image.open(path)
            aug = augment_image(img.copy())
            rows.append((f"A #{idx}", img, aug))

    # Source B: 展示最多6帧
    if has_B:
        for i in range(min(6, total_B)):
            path = os.path.join(ref_dir_B, f'{i:03d}.png')
            img = Image.open(path)
            aug = augment_image(img.copy())
            rows.append((f"B #{i}", img, aug))

    # Source C: 展示6帧
    if has_C and total_C > 0:
        cands = list(range(total_C))
        for _ in range(6):
            idx = cos_weighted_sample(cands)
            path = os.path.join(ref_dir_C, f'render_{idx:04d}.png')
            img = Image.open(path)
            aug = augment_image(img.copy())
            rows.append((f"C #{idx}", img, aug))

    if len(rows) == 0:
        print("No images to visualize!")
        return

    # 拼图: 每行 = label | original | augmented
    label_w = 100
    cols = 2  # original + augmented
    canvas_w = label_w + cols * thumb
    canvas_h = len(rows) * thumb
    canvas = Image.new('RGB', (canvas_w, canvas_h), (32, 32, 32))

    for r, (label, orig, aug) in enumerate(rows):
        y = r * thumb
        # label area
        add_label(canvas, label, (5, y + thumb // 2 - 6))

        # original
        orig_rgb = rgba_on_black(orig).resize((thumb, thumb), Image.Resampling.LANCZOS)
        orig_rgb = add_label(orig_rgb, "orig")
        canvas.paste(orig_rgb, (label_w, y))

        # augmented
        aug_rgb = aug.convert('RGB') if aug.mode != 'RGB' else aug
        aug_rgb = aug_rgb.resize((thumb, thumb), Image.Resampling.LANCZOS)
        aug_rgb = add_label(aug_rgb, "aug")
        canvas.paste(aug_rgb, (label_w + thumb, y))

    out_path = os.path.join(args.output_dir, "ref_sources_vis.png")
    canvas.save(out_path)
    print(f"Saved: {out_path}")

    # ========== 4. 帧索引分布直方图 (text-based) ==========
    if frame_counter_A:
        print(f"\nSource A frame sampling distribution (top 20):")
        for idx, cnt in frame_counter_A.most_common(20):
            bar = "#" * cnt
            print(f"  frame {idx:3d}: {cnt:3d} {bar}")

    if frame_counter_C:
        print(f"\nSource C frame sampling distribution:")
        for idx in sorted(frame_counter_C.keys()):
            cnt = frame_counter_C[idx]
            bar = "#" * cnt
            print(f"  frame {idx:3d}: {cnt:3d} {bar}")

    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
