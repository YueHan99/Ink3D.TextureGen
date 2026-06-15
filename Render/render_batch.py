#!/usr/bin/env python3
"""
Ink3D Batch Render — Batch rendering of GLB files with H/V orbits.

Scans a directory for GLB files and renders each one using render.py.

Usage:
    python render_batch.py --input_dir /path/to/glbs --output_dir ./output --orbit horizontal
    python render_batch.py --input_dir /path/to/glbs --orbit vertical --threads 2
"""

import glob
import os
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

print_lock = threading.Lock()


def find_glb_files(input_dir):
    """Find all .glb files in the input directory."""
    patterns = [
        os.path.join(input_dir, "*.glb"),
        os.path.join(input_dir, "**", "*.glb"),
    ]
    paths = set()
    for pattern in patterns:
        paths.update(glob.glob(pattern, recursive=True))
    return sorted(paths)


def render_single(glb_path, args):
    """Render a single GLB file."""
    glb_id = Path(glb_path).stem
    cmd = [
        "python3", os.path.join(os.path.dirname(os.path.abspath(__file__)), "render.py"),
        "--input_file", glb_path,
        "--output_dir", args.output_dir,
        "--orbit", args.orbit,
        "--num_cameras", str(args.num_cameras),
        "--engine", args.engine,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=97200)
        if result.returncode != 0:
            with print_lock:
                print(f"[ERROR] {glb_id}: {result.stderr[:300]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        with print_lock:
            print(f"[ERROR] Timeout: {glb_id}")
        return False
    except Exception as e:
        with print_lock:
            print(f"[ERROR] {glb_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Ink3D Batch Render")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing GLB files")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--orbit", type=str, choices=["horizontal", "vertical"],
                        default="horizontal", help="Camera orbit type")
    parser.add_argument("--num_cameras", type=int, default=120, help="Number of cameras")
    parser.add_argument("--engine", type=str, default="CYCLES_GPU",
                        choices=["CYCLES_GPU", "CYCLES_CPU", "BLENDER_EEVEE"])
    parser.add_argument("--skip", type=int, default=0, help="Skip first N files")
    parser.add_argument("--threads", type=int, default=1, help="Number of parallel threads")
    args = parser.parse_args()

    print(f"Scanning: {args.input_dir}")
    glb_paths = find_glb_files(args.input_dir)
    print(f"Found {len(glb_paths)} GLB files")

    if args.skip > 0:
        glb_paths = glb_paths[args.skip:]
        print(f"Skipped {args.skip}, remaining: {len(glb_paths)}")

    if not glb_paths:
        print("No GLB files to process.")
        return

    if args.threads > 1:
        print(f"Using {args.threads} threads")
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = {executor.submit(render_single, p, args): p for p in glb_paths}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Rendering"):
                future.result()
    else:
        for path in tqdm(glb_paths, desc="Rendering"):
            render_single(path, args)

    print("\n===== All tasks complete =====")


if __name__ == "__main__":
    main()
