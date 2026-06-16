#!/usr/bin/env python3
"""
Batch rendering for albedo/normal/position/depth channels.
Wraps render.py with multi-threading, skip/resume.

Output per GLB: rgb.mp4, albedo.mp4, normal.mp4, position.mp4, depth.mp4, mask.mp4

Usage:
  python batch_render_albedo.py --input_json glb_list.json \
      --base_glb_path /path/to/glbs \
      --output_dir_h ./h_output --output_dir_v ./v_output \
      --num_camera 120 --threads 4
"""
import json, os, subprocess, argparse, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RENDER_SCRIPT = os.path.join(SCRIPT_DIR, "render.py")

VIDEO_TYPES = ["rgb", "albedo", "normal", "position", "depth", "mask"]
print_lock = threading.Lock()


def load_glb_paths(json_file):
    with open(json_file) as f:
        return json.load(f)


def get_dir_and_id(relative_path):
    p = Path(relative_path)
    return p.parent.name, p.stem


def check_processed(relative_path, output_dir):
    dir_name, glb_id = get_dir_and_id(relative_path)
    base = os.path.join(output_dir, "videos_curve", dir_name, glb_id)
    return all(os.path.exists(os.path.join(base, f"{vt}.mp4")) for vt in VIDEO_TYPES)


def render_orbit(glb_path, relative_path, output_dir, orbit, args):
    dir_name, glb_id = get_dir_and_id(relative_path)
    cmd = [
        "python3", RENDER_SCRIPT,
        "--input_file", os.path.join(args.base_glb_path, relative_path),
        "--output_dir", output_dir,
        "--orbit", orbit,
        "--num_cameras", str(args.num_camera),
        "--model_name", f"{dir_name}/{glb_id}",
        "--engine", "CYCLES_GPU",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=72000)
        if r.returncode != 0:
            with print_lock:
                print(f"\n[ERROR] {orbit} render failed for {relative_path}: {r.stderr[:200]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        with print_lock:
            print(f"\n[ERROR] {orbit} render timeout for {relative_path}")
        return False


def process_single(relative_path, args):
    with print_lock:
        print(f"[START] {relative_path}")

    if not check_processed(relative_path, args.output_dir_h):
        if not render_orbit(None, relative_path, args.output_dir_h, "horizontal", args):
            return False

    if not check_processed(relative_path, args.output_dir_v):
        if not render_orbit(None, relative_path, args.output_dir_v, "vertical", args):
            return False

    with print_lock:
        print(f"[DONE] {relative_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Batch albedo/normal/position rendering")
    parser.add_argument("--input_json", default="selected_glb_paths.json")
    parser.add_argument("--base_glb_path", required=True)
    parser.add_argument("--output_dir_h", required=True)
    parser.add_argument("--output_dir_v", required=True)
    parser.add_argument("--num_camera", type=int, default=120)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--skip", type=int, default=0)
    args = parser.parse_args()

    glb_paths = load_glb_paths(args.input_json)
    print(f"Loaded {len(glb_paths)} GLBs from {args.input_json}")

    if args.skip > 0:
        glb_paths = glb_paths[args.skip:]
        print(f"Skipped {args.skip}, {len(glb_paths)} remaining")

    if args.threads > 1:
        with ThreadPoolExecutor(max_workers=args.threads) as ex:
            futures = [ex.submit(process_single, p, args) for p in glb_paths]
            for f in tqdm(as_completed(futures), total=len(futures)):
                f.result()
    else:
        for p in tqdm(glb_paths):
            process_single(p, args)

    print("Done.")


if __name__ == "__main__":
    main()
