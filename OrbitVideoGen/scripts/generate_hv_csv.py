#!/usr/bin/env python3
"""
Generate merged H+V CSV for training.
- H rows: prefix paths with objaverse_60k_120h/videos_curve/
- V rows: prefix paths with objaverse_60k_120v/videos_curve/, position.mp4 -> position_flip.mp4
- base_path should be set to your dataset base path
"""
import pandas as pd

# H csv
h_csv = "position2albedo_all_321.csv"
h_df = pd.read_csv(h_csv)
h_df["video"] = "objaverse_60k_120h/videos_curve/" + h_df["video"]
h_df["control_video"] = "objaverse_60k_120h/videos_curve/" + h_df["control_video"]

# V csv: same structure, but position.mp4 -> position_flip.mp4
v_df = pd.read_csv(h_csv)
v_df["video"] = "objaverse_60k_120v/videos_curve/" + v_df["video"]
v_df["control_video"] = "objaverse_60k_120v/videos_curve/" + v_df["control_video"].str.replace("position.mp4", "position_flip.mp4")

# Merge
merged = pd.concat([h_df, v_df], ignore_index=True)
out_path = "position2albedo_all_321_hv.csv"
merged.to_csv(out_path, index=False)
print(f"H: {len(h_df)}, V: {len(v_df)}, Merged: {len(merged)} -> {out_path}")
print(merged.head(2).to_string())
print("...")
print(merged.tail(2).to_string())
