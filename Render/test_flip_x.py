#!/usr/bin/env python3
"""
Test: render single frame with tangent=[1,0,0] vs tangent=[-1,0,0] on meridian_270.
If tangent flip produces left-right mirror, the two images will differ in X direction.
"""
import os, sys
import numpy as np
from math import pi
from scipy.spatial.transform import Rotation as R

# Simulate camera matrices (no Blender needed)
center = np.array([0., 0., 0.])
radius = 1.5
num_cameras = 120
start_angle_offset = pi / 2

angles = np.linspace(0, 2 * pi, num_cameras, endpoint=False) + start_angle_offset

# Pick frame 30
angle = angles[30]
azimuth_rad = 3 * pi / 2  # meridian_270
elevation_rad = angle - pi / 2

x = center[0] + radius * np.cos(elevation_rad) * np.cos(azimuth_rad)
y = center[1] + radius * np.cos(elevation_rad) * np.sin(azimuth_rad)
z = center[2] + radius * np.sin(elevation_rad)
cam_pos = np.array([x, y, z])
direction = center - cam_pos
direction = direction / np.linalg.norm(direction)

for label, tangent_init in [("tangent +X", np.array([1., 0., 0.])), ("tangent -X", np.array([-1., 0., 0.]))]:
    tangent = tangent_init
    tangent = tangent - np.dot(tangent, direction) * direction
    tangent = tangent / (np.linalg.norm(tangent) + 1e-8)
    right = np.cross(tangent, direction)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(direction, right)
    rot_matrix = np.column_stack([right, up, -direction])
    rot_90 = R.from_rotvec(-pi / 2 * direction).as_matrix()
    rot_matrix = rot_90 @ rot_matrix
    print(f"\n{label}:")
    print(f"  right (image X):     {rot_matrix[:, 0]}")
    print(f"  up (image Y):        {rot_matrix[:, 1]}")
    print(f"  look (-Z direction): {rot_matrix[:, 2]}")
    print(f"  cam_pos:             {cam_pos}")
