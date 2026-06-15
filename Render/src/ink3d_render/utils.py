"""Utility functions for image processing and conversion."""

import numpy as np
from PIL import Image
from typing import List, Tuple
import imageio


def get_local2world_mat(blender_obj) -> np.ndarray:
    """Returns the 4x4 local2world matrix of a Blender object."""
    obj = blender_obj
    matrix_world = obj.matrix_basis
    while obj.parent is not None:
        matrix_world = obj.parent.matrix_basis @ obj.matrix_parent_inverse @ matrix_world
        obj = obj.parent
    return np.array(matrix_world)


def rgba_to_rgb(rgba_image, bg_color=[255, 255, 255]):
    """Composite RGBA image onto a solid background color."""
    background = np.array(bg_color)
    foreground = rgba_image[..., :3].astype(float)
    background = background.astype(float)
    alpha = rgba_image[..., 3:].astype(float) / 255
    rgb_image = alpha * foreground + (1 - alpha) * background
    return rgb_image.astype(np.uint8)


def get_keyframes(obj_list):
    """Extract keyframe positions from animation data."""
    keyframes = []
    for obj in obj_list:
        anim = obj.animation_data
        if anim is not None and anim.action is not None:
            for fcu in anim.action.fcurves:
                for keyframe in fcu.keyframe_points:
                    x, y = keyframe.co
                    if x not in keyframes:
                        keyframes.append(int(x))
    return keyframes


def load_image(file_path: str, num_channels: int = 3) -> np.ndarray:
    """Load an image file and return as numpy array."""
    file_ending = file_path[file_path.rfind(".") + 1:].lower()
    if file_ending in ["exr", "png", "webp"]:
        return imageio.imread(file_path)[:, :, :num_channels]
    elif file_ending in ["jpg"]:
        import cv2
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    else:
        raise NotImplementedError(f"Cannot load file with ending: {file_ending}")


def convert_normal_to_webp(src: str, dst: str, src_render: str):
    """Convert normal EXR to PNG/WebP with alpha from render image."""
    data = load_image(src, 4)
    normal_map = data[:, :, :3] * 255
    try:
        alpha_channel = load_image(src_render, 4)[:, :, 3]
        for i in range(alpha_channel.shape[0]):
            for j in range(alpha_channel.shape[1]):
                alpha_channel[i][j] = 255 if alpha_channel[i][j] > 0 else 0
        normal_map = np.concatenate((normal_map, alpha_channel[:, :, np.newaxis]), axis=2)
    except:
        pass

    save_type = dst.split(".")[-1]
    Image.fromarray(normal_map.astype(np.uint8)).save(dst, save_type, quality=100)


def convert_depth_to_webp(src: List[str], dst: List[str]) -> Tuple[float, float]:
    """Convert depth EXR images to PNG with global normalization.

    Returns:
        (min_depth, scale) used for normalization
    """
    depth_images = []
    valid_masks = []
    min_depth = float("inf")
    max_depth = float("-inf")

    for path in src:
        depth = imageio.imread(path)
        mask = np.ones_like(depth, dtype=float)
        mask[depth > 1000.0] = 0.0
        depth[~(mask > 0.5)] = 0.0

        valid_depths = depth[mask > 0.5]
        if len(valid_depths) > 0:
            min_depth = min(min_depth, valid_depths.min())
            max_depth = max(max_depth, valid_depths.max())

        depth_images.append(depth)
        valid_masks.append(mask)

    scale = 255.0 / (max_depth - min_depth) if max_depth > min_depth else 1.0

    for depth, mask, output_path in zip(depth_images, valid_masks, dst):
        normalized_depth = (depth - min_depth) * scale
        normalized_depth[~(mask > 0.5)] = 0.0
        depth_uint8 = normalized_depth.astype(np.uint8)
        imageio.imwrite(output_path, depth_uint8)

    return min_depth, scale


PRESET_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
]
