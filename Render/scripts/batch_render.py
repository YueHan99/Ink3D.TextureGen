#!/usr/bin/env python3
"""
Batch GLB rendering for dataset preparation.
Wraps render.py with multi-threading, skip/resume, and H/V/HV output.

Usage:
  python batch_render.py --input_json glb_list.json --base_glb_path /path/to/glbs \
      --output_dir_h ./h_output --output_dir_v ./v_output \
      --mode albedo --num_camera 120 --threads 4
"""
import json, os, subprocess, argparse, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

RENDER_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "render.py")

VIDEO_TYPES_ALBEDO = ["rgb", "albedo", "normal", "position", "depth", "mask"]
VIDEO_TYPES_MR = ["mr"]

print_lock = threading.Lock()


def load_glb_paths(json_file):
    with open(json_file) as f:
        return json.load(f)


def get_full_glb_path(base, rel):
    return os.path.join(base, rel)


def get_dir_and_id(relative_path):
    p = Path(relative_path)
    return p.parent.name, p.stem


def check_processed(relative_path, output_dir):
    """Check if all video types exist for this GLB."""
    dir_name, glb_id = get_dir_and_id(relative_path)
    base = os.path.join(output_dir, "videos_curve", dir_name, glb_id)
    for vt in args.video_types:
        if not os.path.exists(os.path.join(base, f"{vt}.mp4")):
            return False
    return True


def render_orbit(glb_path, relative_path, output_dir, orbit, args):
    """Run render.py for H or V orbit."""
    dir_name, glb_id = get_dir_and_id(relative_path)
    output_model_dir = os.path.join(output_dir, "videos_curve", dir_name)
    cmd = [
        "python3", RENDER_SCRIPT,
        "--input_file", glb_path,
        "--output_dir", output_dir,
        "--orbit", orbit,
        "--num_cameras", str(args.num_camera),
        "--model_name", f"{dir_name}/{glb_id}",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=72000)
        if r.returncode != 0:
            with print_lock:
                print(f"[ERROR] {orbit} render failed {relative_path}: {r.stderr[:200]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        with print_lock:
            print(f"[ERROR] {orbit} render timeout {relative_path}")
        return False
    except Exception as e:
        with print_lock:
            print(f"[ERROR] {orbit} render error {relative_path}: {e}")
        return False


def process_single(relative_path, args):
    glb_path = get_full_glb_path(args.base_glb_path, relative_path)
    with print_lock:
        print(f"[START] {relative_path}")

    # H orbit
    if not check_processed(relative_path, args.output_dir_h):
        if not render_orbit(glb_path, relative_path, args.output_dir_h, "horizontal", args):
            return False

    # V orbit
    if not check_processed(relative_path, args.output_dir_v):
        if not render_orbit(glb_path, relative_path, args.output_dir_v, "vertical", args):
            return False

    with print_lock:
        print(f"[DONE] {relative_path}")
    return True


def main():
    global args
    parser = argparse.ArgumentParser(description="Batch GLB rendering")
    parser.add_argument("--input_json", default="selected_glb_paths.json")
    parser.add_argument("--base_glb_path", required=True)
    parser.add_argument("--output_dir_h", required=True)
    parser.add_argument("--output_dir_v", required=True)
    parser.add_argument("--mode", choices=["albedo", "mr"], default="albedo")
    parser.add_argument("--num_camera", type=int, default=120)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--skip", type=int, default=0)
    args = parser.parse_args()

    args.video_types = VIDEO_TYPES_ALBEDO if args.mode == "albedo" else VIDEO_TYPES_MR

    glb_paths = load_glb_paths(args.input_json)
    print(f"Loaded {len(glb_paths)} GLBs")

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
