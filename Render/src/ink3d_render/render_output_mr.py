"""PBR metallic/roughness AOV output with Raw color transform.

Separate from render_output_a because metallic/roughness requires
view_transform='Raw' and gamma=1 to preserve physical values.
"""

import bpy
from typing import Optional, Literal

from .render_output import (
    _new_output_file_node,
    _set_output_item_path,
    _set_output_item_use_node_format,
    _ensure_compositor_node_tree,
)


def enable_pbr_output(output_dir, attr_name, color_mode="BW", file_prefix: str = ""):
    """Enable PBR AOV output for metallic/roughness with Raw color transform.

    Uses view_transform='Raw' and gamma=1 so physical material values
    are written directly without Blender color management interference.
    """
    scene = bpy.context.scene
    scene.render.film_transparent = True
    scene.render.image_settings.quality = 100
    scene.display_settings.display_device = 'sRGB'
    scene.view_settings.view_transform = 'Raw'
    scene.view_settings.look = 'None'
    scene.view_settings.gamma = 1

    if file_prefix == "":
        file_prefix = attr_name.lower().replace(" ", "-") + "_"

    for material in bpy.data.materials:
        if not material.use_nodes:
            continue
        node_tree = material.node_tree
        if not node_tree:
            continue
        nodes = node_tree.nodes

        if "Principled BSDF" not in nodes:
            continue

        principled_node = nodes["Principled BSDF"]
        if attr_name not in principled_node.inputs:
            print(f"Warning: '{attr_name}' not found in Principled BSDF for material '{material.name}'")
            continue

        attr_input = principled_node.inputs[attr_name]

        if attr_input.is_linked:
            linked_socket = attr_input.links[0].from_socket
            aov_output = nodes.new("ShaderNodeOutputAOV")
            aov_output.name = attr_name
            node_tree.links.new(linked_socket, aov_output.inputs[0])
        else:
            fixed_value = attr_input.default_value
            if isinstance(fixed_value, float):
                value_node = nodes.new("ShaderNodeValue")
                value_node.outputs[0].default_value = fixed_value
            else:
                value_node = nodes.new("ShaderNodeRGB")
                value_node.outputs[0].default_value = fixed_value

            aov_output = nodes.new("ShaderNodeOutputAOV")
            aov_output.name = attr_name
            node_tree.links.new(value_node.outputs[0], aov_output.inputs[0])

    tree = _ensure_compositor_node_tree()
    links = tree.links
    if "Render Layers" not in tree.nodes:
        rl = tree.nodes.new("CompositorNodeRLayers")
    else:
        rl = tree.nodes["Render Layers"]

    pbr_file_output = _new_output_file_node(tree, output_dir)
    _set_output_item_use_node_format(pbr_file_output, 0, True)
    pbr_file_output.format.file_format = "PNG"
    pbr_file_output.format.color_mode = color_mode
    pbr_file_output.format.color_depth = "16"
    _set_output_item_path(pbr_file_output, 0, file_prefix)

    bpy.ops.scene.view_layer_add_aov()
    bpy.context.scene.view_layers["ViewLayer"].active_aov.name = attr_name

    pbr_alpha = tree.nodes.new(type="CompositorNodeSetAlpha")
    tree.links.new(rl.outputs[attr_name], pbr_alpha.inputs["Image"])
    tree.links.new(rl.outputs["Alpha"], pbr_alpha.inputs["Alpha"])

    links.new(pbr_alpha.outputs["Image"], pbr_file_output.inputs[0])
