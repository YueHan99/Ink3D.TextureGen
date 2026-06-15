"""Compositor output setup for multi-pass rendering (bpy 4.x/5.x compatible)."""

import math
import os
from typing import List, Optional, Literal, Tuple, Dict

import bpy
import numpy as np


# ─── bpy 4.x / 5.x compatibility helpers ───

def _new_output_file_node(tree, output_dir):
    """Create CompositorNodeOutputFile with bpy 4.x/5.x compat."""
    node = tree.nodes.new("CompositorNodeOutputFile")
    if hasattr(node, 'base_path'):
        node.base_path = output_dir
    else:
        node.directory = output_dir
    if hasattr(node, 'file_output_items') and len(node.file_output_items) == 0:
        node.file_output_items.new("RGBA", "Image")
    return node


def _set_output_item_path(node, idx, value):
    """Set output item path/name compatible with bpy 4.x/5.x."""
    if hasattr(node, 'file_slots'):
        node.file_slots.values()[idx].path = value
    else:
        node.file_output_items[idx].name = value


def _set_output_item_use_node_format(node, idx, value):
    """Set use_node_format/override_node_format compatible with bpy 4.x/5.x."""
    if hasattr(node, 'file_slots'):
        node.file_slots[idx].use_node_format = value
    else:
        node.file_output_items[idx].override_node_format = not value


def _exr_format():
    """Return correct EXR format enum for current bpy version."""
    if bpy.app.version >= (5, 0, 0):
        return "OPEN_EXR_MULTILAYER"
    return "OPEN_EXR"


def _ensure_compositor_node_tree():
    """Ensure scene compositor node_tree is available (bpy 4.0+ compat)."""
    scene = bpy.context.scene
    scene.render.use_compositing = True
    scene.use_nodes = True

    if hasattr(scene, 'node_tree') and scene.node_tree is not None:
        return scene.node_tree

    if hasattr(scene, 'compositing_node_group'):
        ng = scene.compositing_node_group
        if ng is None:
            ng = bpy.data.node_groups.new('Compositing', 'CompositorNodeTree')
            scene.compositing_node_group = ng
        return ng

    raise RuntimeError("Cannot initialize scene compositor node_tree")


# ─── Output functions ───

def enable_color_output(
    width: int,
    height: int,
    output_dir: Optional[str] = "",
    file_prefix: str = "render_",
    file_format: Literal["WEBP", "PNG"] = "WEBP",
    mode: Literal["IMAGE", "VIDEO"] = "IMAGE",
    **kwargs,
):
    film_transparent = kwargs.get("film_transparent", True)
    fps = kwargs.get("fps", 24)

    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = film_transparent
    scene.render.image_settings.quality = 100

    if mode == "IMAGE":
        scene.render.image_settings.file_format = file_format
        scene.render.image_settings.color_mode = "RGBA"
    elif mode == "VIDEO":
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.fps = fps
    scene.render.filepath = os.path.join(output_dir, file_prefix)


def make_normal_to_rgb_node_group(node_tree, editor_type="Compositor"):
    """Create a node group that converts normal vectors to RGB colors."""
    link = lambda from_socket, to_socket: node_tree.links.new(from_socket, to_socket)
    sep_color_node = node_tree.nodes.new(f"{editor_type}NodeSeparateColor")

    def create_normal_to_rgb_map_node():
        node = node_tree.nodes.new(f"{editor_type}NodeMapRange")
        if editor_type == "Shader":
            node.clamp = True
        elif editor_type == "Compositor":
            node.use_clamp = True
        node.inputs["From Min"].default_value = -1.0
        node.inputs["From Max"].default_value = 1.0
        node.inputs["To Min"].default_value = 0.0
        node.inputs["To Max"].default_value = 1.0
        return node

    if editor_type == "Shader":
        map_range_node_output_socket_name = "Result"
        converter_io_node_socket_name = "Color"
    elif editor_type == "Compositor":
        map_range_node_output_socket_name = "Value"
        converter_io_node_socket_name = "Image"

    map_range_nodes = {k: create_normal_to_rgb_map_node() for k in ["R", "G", "B"]}
    comb_color_node = node_tree.nodes.new(f"{editor_type}NodeCombineColor")

    link(sep_color_node.outputs["Red"], map_range_nodes["R"].inputs["Value"])
    link(sep_color_node.outputs["Green"], map_range_nodes["G"].inputs["Value"])
    link(sep_color_node.outputs["Blue"], map_range_nodes["B"].inputs["Value"])

    link(map_range_nodes["R"].outputs[map_range_node_output_socket_name], comb_color_node.inputs["Red"])
    link(map_range_nodes["G"].outputs[map_range_node_output_socket_name], comb_color_node.inputs["Green"])
    link(map_range_nodes["B"].outputs[map_range_node_output_socket_name], comb_color_node.inputs["Blue"])

    return (
        sep_color_node.inputs[converter_io_node_socket_name],
        comb_color_node.outputs[converter_io_node_socket_name],
    )


def set_file_output_non_color(node):
    """Set file output node to use non-color data management."""
    if int(bpy.app.version_string[0]) >= 4:
        node.format.color_management = "OVERRIDE"
        node.format.view_settings.view_transform = "Raw"
    else:
        node.format.color_management = "OVERRIDE"
        node.format.display_settings.display_device = "None"


def enable_normals_output(
    output_dir: Optional[str] = "",
    file_prefix: str = "normal_",
    use_rgb_conversion: bool = True,
    file_format: Literal["OPEN_EXR", "WEBP", "PNG"] = "PNG",
):
    """Enable normal map output."""
    tree = _ensure_compositor_node_tree()

    if "Render Layers" not in tree.nodes:
        rl = tree.nodes.new("CompositorNodeRLayers")
    else:
        rl = tree.nodes["Render Layers"]
    bpy.context.view_layer.use_pass_normal = True

    normal_file_output = _new_output_file_node(tree, output_dir)
    normal_file_output.location.x = 400
    _set_output_item_path(normal_file_output, 0, file_prefix)

    if use_rgb_conversion and file_format in ["WEBP", "PNG"]:
        normal_trans_input_socket, normal_trans_output_socket = (
            make_normal_to_rgb_node_group(tree, editor_type="Compositor")
        )

        set_normal_alpha_node = tree.nodes.new("CompositorNodeSetAlpha")
        set_normal_alpha_node.mode = "REPLACE_ALPHA"

        tree.links.new(rl.outputs["Normal"], normal_trans_input_socket)
        tree.links.new(normal_trans_output_socket, set_normal_alpha_node.inputs["Image"])
        tree.links.new(rl.outputs["Alpha"], set_normal_alpha_node.inputs["Alpha"])
        tree.links.new(set_normal_alpha_node.outputs["Image"], normal_file_output.inputs[0])

        if file_format == "WEBP":
            normal_file_output.format.file_format = "WEBP"
            normal_file_output.format.quality = 100
            normal_file_output.format.color_depth = "8"
        elif file_format == "PNG":
            normal_file_output.format.file_format = "PNG"
            normal_file_output.format.color_depth = "16"

        set_file_output_non_color(normal_file_output)
    else:
        tree.links.new(rl.outputs["Normal"], normal_file_output.inputs[0])
        if file_format == "OPEN_EXR":
            normal_file_output.format.file_format = _exr_format()
            normal_file_output.format.color_mode = "RGBA"
            normal_file_output.format.color_depth = "32"


def enable_depth_output(output_dir: Optional[str] = "", file_prefix: str = "depth_"):
    """Enable depth map output (EXR format)."""
    tree = _ensure_compositor_node_tree()
    links = tree.links

    if "Render Layers" not in tree.nodes:
        rl = tree.nodes.new("CompositorNodeRLayers")
    else:
        rl = tree.nodes["Render Layers"]
    bpy.context.view_layer.use_pass_z = True

    depth_output = _new_output_file_node(tree, output_dir)
    depth_output.name = "DepthOutput"
    depth_output.format.file_format = _exr_format()
    depth_output.format.color_depth = "32"
    _set_output_item_path(depth_output, 0, file_prefix)

    links.new(rl.outputs["Depth"], depth_output.inputs[0])


def enable_albedo_output(output_dir: Optional[str] = "", file_prefix: str = "albedo_"):
    """Enable albedo (diffuse color) output."""
    tree = _ensure_compositor_node_tree()

    if "Render Layers" not in tree.nodes:
        rl = tree.nodes.new("CompositorNodeRLayers")
    else:
        rl = tree.nodes["Render Layers"]
    bpy.context.view_layer.use_pass_diffuse_color = True

    alpha_albedo = tree.nodes.new(type="CompositorNodeSetAlpha")
    tree.links.new(rl.outputs["DiffCol"], alpha_albedo.inputs["Image"])
    tree.links.new(rl.outputs["Alpha"], alpha_albedo.inputs["Alpha"])

    albedo_file_output = _new_output_file_node(tree, output_dir)
    _set_output_item_use_node_format(albedo_file_output, 0, True)
    albedo_file_output.format.file_format = "PNG"
    albedo_file_output.format.color_mode = "RGBA"
    albedo_file_output.format.color_depth = "16"
    _set_output_item_path(albedo_file_output, 0, file_prefix)

    tree.links.new(alpha_albedo.outputs["Image"], albedo_file_output.inputs[0])


def enable_position_output(
    output_dir: Optional[str] = "",
    file_prefix: str = "position_",
    space: Literal["WORLD", "VIEW"] = "WORLD",
    file_format: Literal["OPEN_EXR", "PNG"] = "OPEN_EXR"
):
    """Enable position map output (world or view space)."""
    tree = _ensure_compositor_node_tree()
    links = tree.links
    scene = bpy.context.scene

    if "Render Layers" not in tree.nodes:
        rl = tree.nodes.new("CompositorNodeRLayers")
    else:
        rl = tree.nodes["Render Layers"]

    aov_name = f"Position_{space}"

    for obj in scene.objects:
        if obj.type != 'MESH':
            continue
        if not obj.data.materials:
            mat = bpy.data.materials.new(name="DefaultMaterial")
            mat.use_nodes = True
            obj.data.materials.append(mat)
        for material in obj.data.materials:
            if not material:
                continue
            if not material.use_nodes:
                material.use_nodes = True
            node_tree = material.node_tree
            nodes = node_tree.nodes

            if aov_name in [n.name for n in nodes if n.type == 'OUTPUT_AOV']:
                continue

            geo_node = nodes.new("ShaderNodeNewGeometry")
            pos_socket = geo_node.outputs["Position"]

            aov_output = nodes.new("ShaderNodeOutputAOV")
            aov_output.name = aov_name
            node_tree.links.new(pos_socket, aov_output.inputs[0])

    view_layer = scene.view_layers["ViewLayer"]
    if aov_name not in [a.name for a in view_layer.aovs]:
        bpy.ops.scene.view_layer_add_aov()
        new_aov = view_layer.active_aov
        new_aov.name = aov_name
        new_aov.type = 'COLOR'

    pos_file_output = _new_output_file_node(tree, output_dir)
    _set_output_item_use_node_format(pos_file_output, 0, True)
    _set_output_item_path(pos_file_output, 0, file_prefix)

    if file_format == "OPEN_EXR":
        pos_file_output.format.file_format = _exr_format()
        pos_file_output.format.color_mode = "RGB"
        pos_file_output.format.color_depth = "32"
    elif file_format == "PNG":
        pos_file_output.format.file_format = "PNG"
        pos_file_output.format.color_mode = "RGB"
        pos_file_output.format.color_depth = "16"

    set_file_output_non_color(pos_file_output)
    links.new(rl.outputs[aov_name], pos_file_output.inputs[0])
