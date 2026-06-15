"""Blender camera add utility."""

import bpy
from mathutils import Matrix
from typing import Literal


def add_camera(
    cam2world_matrix: Matrix,
    camera_type: Literal["PERSP", "ORTHO"] = "PERSP",
    camera_sensor_width: int = 32,
    camera_lens: int = 35,
    ortho_scale: int = 1.1,
    add_frame: bool = False,
):
    """Add a camera keyframe at the current frame end.

    Args:
        cam2world_matrix: 4x4 camera-to-world transformation matrix
        camera_type: Perspective or orthographic
        camera_sensor_width: Sensor width in mm
        camera_lens: Focal length in mm
        ortho_scale: Orthographic scale
        add_frame: Whether to advance frame counter
    """
    if not isinstance(cam2world_matrix, Matrix):
        cam2world_matrix = Matrix(cam2world_matrix)
    if bpy.context.scene.camera is None:
        bpy.ops.object.camera_add(location=(0, 0, 0))
        for obj in bpy.data.objects:
            if obj.type == "CAMERA":
                bpy.context.scene.camera = obj

    cam_ob = bpy.context.scene.camera
    cam_ob.data.type = camera_type
    cam_ob.data.sensor_width = camera_sensor_width
    if camera_type == "PERSP":
        cam_ob.data.lens = camera_lens
    elif camera_type == "ORTHO":
        cam_ob.data.ortho_scale = ortho_scale
    cam_ob.matrix_world = cam2world_matrix

    frame = bpy.context.scene.frame_end
    cam_ob.keyframe_insert(data_path="location", frame=frame)
    cam_ob.keyframe_insert(data_path="rotation_euler", frame=frame)
    cam_ob.data.keyframe_insert(data_path="type", frame=frame)
    cam_ob.data.keyframe_insert(data_path="sensor_width", frame=frame)

    if camera_type == "ORTHO":
        cam_ob.data.keyframe_insert(data_path="ortho_scale", frame=frame)
    elif camera_type == "PERSP":
        cam_ob.data.keyframe_insert(data_path="lens", frame=frame)

    if add_frame:
        bpy.context.scene.frame_end += 1

    return cam_ob
