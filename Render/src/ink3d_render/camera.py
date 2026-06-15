"""Camera trajectory functions for orbit rendering."""

import math
from typing import List, Tuple, Optional
import numpy as np
from mathutils import Euler, Matrix, Vector
from math import pi, sin, cos
import bpy


def camera_rotation_matrix_world_up(cam_pos, center, up_vector=Vector((0, 0, 1))):
    """Build a stable camera rotation matrix using world Z as up vector."""
    forward = (center - cam_pos).normalized()
    right = forward.cross(up_vector).normalized()
    if right.length < 1e-6:
        right = Vector((1, 0, 0))
    up = right.cross(forward).normalized()

    rot_mat = Matrix().to_3x3()
    rot_mat.col[0] = right
    rot_mat.col[1] = up
    rot_mat.col[2] = -forward

    return np.array(rot_mat)


def build_transformation_mat(translation, rotation) -> np.ndarray:
    """Build a 4x4 transformation matrix from translation and rotation."""
    translation = np.array(translation)
    rotation = np.array(rotation)

    mat = np.eye(4)
    if translation.shape[0] == 3:
        mat[:3, 3] = translation
    else:
        raise RuntimeError(f"Translation has invalid shape: {translation.shape}")
    if rotation.shape == (3, 3):
        mat[:3, :3] = rotation
    elif rotation.shape[0] == 3:
        mat[:3, :3] = np.array(Euler(rotation).to_matrix())
    else:
        raise RuntimeError(f"Rotation has invalid shape: {rotation.shape}")

    return mat


def get_camera_positions_on_sphere(
    center: Tuple[float, float, float],
    radius: float,
    elevations: List[float],
    num_camera_per_layer: Optional[int] = None,
    azimuth_offset: Optional[float] = 0.0,
    azimuths: Optional[List[float]] = None,
) -> Tuple[List, List, List, List]:
    """Get camera positions on a sphere (horizontal orbit).

    Places cameras at specified elevation angles around a sphere.

    Args:
        center: (x,y,z) coordinates of the sphere center
        radius: Radius of the sphere
        elevations: List of elevation angles in degrees
        num_camera_per_layer: Number of cameras per elevation layer
        azimuth_offset: Azimuth offset in degrees
        azimuths: Optional explicit azimuth angles

    Returns:
        Tuple of (points, mats, elevation_t, azimuth_t)
    """
    points, mats, elevation_t, azimuth_t = [], [], [], []

    elevation_deg = elevations
    elevation = np.deg2rad(elevation_deg)

    if num_camera_per_layer is not None and azimuths is None:
        azimuth_deg = np.linspace(0, 360, num_camera_per_layer + 1)[:-1]
        azimuth_deg = azimuth_deg % 360
        if azimuth_offset is not None:
            azimuth_deg += azimuth_offset
    else:
        azimuth_deg = azimuths
    azimuth = np.deg2rad(azimuth_deg)

    for _phi in elevation:
        for theta in azimuth:
            phi = 0.5 * math.pi - _phi
            elevation_t.append(_phi)
            azimuth_t.append(theta)

            r = radius
            x = center[0] + r * math.sin(phi) * math.cos(theta)
            y = center[1] + r * math.sin(phi) * math.sin(theta)
            z = center[2] + r * math.cos(phi)
            cam_pos = Vector((x, y, z))
            points.append(cam_pos)

            center = Vector(center)
            rotation_euler = (center - cam_pos).to_track_quat("-Z", "Y").to_euler()
            cam_matrix = build_transformation_mat(cam_pos, rotation_euler)
            mats.append(cam_matrix)

    return points, mats, elevation_t, azimuth_t


def get_vertical_camera_path_360(
    center=(0, 0, 0),
    radius=1.5,
    num_cameras=9,
    full_circle=True,
    trajectory_type='meridian_270',
    start_angle_offset=0,
    flip_x=False,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[float]]:
    """Generate a 360° vertical camera trajectory along a meridian.

    Args:
        center: Sphere center
        radius: Orbit radius
        num_cameras: Number of camera positions
        full_circle: Whether to complete a full 360° loop
        trajectory_type: Type of vertical trajectory (e.g. 'meridian_270')
        start_angle_offset: Starting angle offset in radians

    Returns:
        Tuple of (points, mats, elevation_t, azimuth_t)
    """
    from scipy.spatial.transform import Rotation as R

    center = np.array(center)
    points, mats = [], []
    elevation_t, azimuth_t = [], []

    if full_circle:
        angles = np.linspace(0, 2 * pi, num_cameras, endpoint=False) + start_angle_offset
    else:
        angles = np.linspace(-pi / 2, pi / 2, num_cameras) + start_angle_offset

    for angle in angles:
        if trajectory_type.startswith('meridian'):
            azimuth_map = {
                'meridian_0': 0,
                'meridian_90': pi / 2,
                'meridian_180': pi,
                'meridian_270': 3 * pi / 2
            }
            azimuth_rad = azimuth_map[trajectory_type]
            elevation_rad = angle - pi / 2
            x = center[0] + radius * np.cos(elevation_rad) * np.cos(azimuth_rad)
            y = center[1] + radius * np.cos(elevation_rad) * np.sin(azimuth_rad)
            z = center[2] + radius * np.sin(elevation_rad)
            elevation = elevation_rad
            azimuth = azimuth_rad
        else:
            raise ValueError(f"Unknown trajectory_type: {trajectory_type}")

        cam_pos = np.array([x, y, z])
        points.append(cam_pos)
        elevation_t.append(elevation)
        azimuth_t.append(azimuth)

        direction = center - cam_pos
        direction = direction / np.linalg.norm(direction)

        tangent = np.array([1, 0, 0])
        tangent = tangent - np.dot(tangent, direction) * direction
        tangent = tangent / (np.linalg.norm(tangent) + 1e-8)

        right = np.cross(tangent, direction)
        right = right / (np.linalg.norm(right) + 1e-8)

        up = np.cross(direction, right)

        rot_matrix = np.column_stack([right, up, -direction])

        rot_90 = R.from_rotvec(-pi / 2 * direction).as_matrix()
        rot_matrix = rot_90 @ rot_matrix

        if flip_x:
            rot_matrix[:, 0] = -rot_matrix[:, 0]

        mat = np.eye(4)
        mat[:3, :3] = rot_matrix
        mat[:3, 3] = cam_pos
        mats.append(mat)

    return points, mats, elevation_t, azimuth_t
