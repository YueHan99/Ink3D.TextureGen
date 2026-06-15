#!/usr/bin/env python3
"""
Generate paired H+V CSV for interleaved training.
Each row: paired H and V data for the same object.
H: objaverse_60k_120h/videos_curve/<uuid>/<file>
V: objaverse_60k_120v/videos_curve/<uuid>/<file> (position.mp4 → position_flip.mp4)
"""
import pandas as pd
import os

h_csv = "position2albedo_all_321.csv"
h_df = pd.read_csv(h_csv)

# Build dict: uuid → H row data
h_map = {}
for _, row in h_df.iterrows():
    uuid = "/".join(row["video"].split("/")[:2])  # e.g. 000-000/abc123
    h_map[uuid] = {
        "albedo": "objaverse_60k_120h/videos_curve/" + row["video"],
        "control": "objaverse_60k_120h/videos_curve/" + row["control_video"],
        "ref": row["reference_image"] if "reference_image" in row else "",
        "prompt": row["prompt"] if "prompt" in row else "",
    }

# Build V map
v_entries = []
for uuid, h_data in h_map.items():
    v_entries.append({
        "h_video": h_data["albedo"],
        "h_control_video": h_data["control"],
        "v_video": "objaverse_60k_120v/videos_curve/" + uuid + "/albedo.mp4",
        "v_control_video": "objaverse_60k_120v/videos_curve/" + uuid + "/position_flip.mp4",
        "reference_image": h_data["ref"],
        "prompt": h_data["prompt"],
    })

paired = pd.DataFrame(v_entries)
out_path = "position2albedo_all_321_hv_paired.csv"
paired.to_csv(out_path, index=False)
print(f"Paired {len(paired)} rows → {out_path}")
print(paired.head(2).to_string())
