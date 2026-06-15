"""Scene management utilities for Blender rendering."""

import bpy
import math
import numpy as np
import mathutils
from mathutils import Vector
from typing import Optional, Literal
from .utils import get_keyframes


class SceneManager:
    @property
    def objects(self):
        return bpy.context.scene.objects

    @property
    def scene_meshes(self):
        return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]

    @property
    def scene_armatures(self):
        return [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]

    @property
    def root_objects(self):
        for obj in bpy.context.scene.objects.values():
            if not obj.parent:
                yield obj

    @property
    def num_frames(self):
        return bpy.context.scene.frame_end + 1

    def get_scene_bbox(self, single_obj=None, ignore_matrix=False):
        bbox_min = (math.inf,) * 3
        bbox_max = (-math.inf,) * 3

        meshes = self.scene_meshes if single_obj is None else [single_obj]
        if len(meshes) == 0:
            raise RuntimeError("No objects in scene to compute bounding box for")

        for obj in meshes:
            for coord in obj.bound_box:
                coord = Vector(coord)
                if not ignore_matrix:
                    coord = obj.matrix_world @ coord
                bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
                bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))

        return Vector(bbox_min), Vector(bbox_max)

    def normalize_scene(
        self,
        normalize_range: float = 1.0,
        range_type: Literal["CUBE", "SPHERE"] = "CUBE",
    ):
        bbox_min, bbox_max = self.get_scene_bbox()

        if range_type == "CUBE":
            scale = normalize_range / max(bbox_max - bbox_min)
        elif range_type == "SPHERE":
            scale = normalize_range / (bbox_max - bbox_min).length
        else:
            raise ValueError(f"Invalid range_type: {range_type}")

        offset = -(bbox_min + bbox_max) / 2

        for obj in self.root_objects:
            obj.matrix_world.translation += offset
            original_translation = obj.matrix_world.translation.copy()
            obj.matrix_world.translation = original_translation * scale
            obj.scale = obj.scale * scale
            bpy.context.view_layer.update()

        bpy.ops.object.select_all(action="DESELECT")

    def render(self):
        bpy.context.scene.render.use_compositing = True
        bpy.context.scene.use_nodes = True

        # bpy 4.0+: use compositing_node_group instead of node_tree
        if hasattr(bpy.context.scene, 'node_tree') and bpy.context.scene.node_tree is not None:
            tree = bpy.context.scene.node_tree
        elif hasattr(bpy.context.scene, 'compositing_node_group'):
            tree = bpy.context.scene.compositing_node_group
            if tree is None:
                tree = bpy.data.node_groups.new('Compositing', 'CompositorNodeTree')
                bpy.context.scene.compositing_node_group = tree
        else:
            tree = None

        if tree is not None and "Render Layers" not in tree.nodes:
            tree.nodes.new("CompositorNodeRLayers")

        bpy.ops.render.render(animation=True, write_still=True)

    def smooth(self):
        for obj in self.scene_meshes:
            if hasattr(obj.data, 'use_auto_smooth'):
                obj.data.use_auto_smooth = True
                obj.data.auto_smooth_angle = np.deg2rad(30)

    def set_materials_opaque(self) -> None:
        """Set all materials to opaque blend mode."""
        for material in bpy.data.materials:
            if not material.use_nodes:
                continue
            material.blend_method = "OPAQUE"

    def set_material_transparency(self, show_transparent_back: bool) -> None:
        """Set transparency settings for materials with blend mode 'BLEND'."""
        for material in bpy.data.materials:
            if not material.use_nodes:
                continue
            if material.blend_method == "BLEND":
                material.show_transparent_back = show_transparent_back

    def clear(
        self,
        clear_objects: Optional[bool] = True,
        clear_nodes: Optional[bool] = True,
        reset_keyframes: Optional[bool] = True,
    ):
        if clear_objects:
            objects = [x for x in bpy.data.objects]
            for obj in objects:
                bpy.data.objects.remove(obj, do_unlink=True)

        if clear_nodes:
            scene = bpy.context.scene
            if hasattr(scene, 'use_nodes'):
                scene.use_nodes = True
            node_tree = None
            if hasattr(scene, 'node_tree') and scene.node_tree is not None:
                node_tree = scene.node_tree
            elif hasattr(scene, 'compositing_node_group') and scene.compositing_node_group is not None:
                node_tree = scene.compositing_node_group
            if node_tree is not None:
                for node in list(node_tree.nodes):
                    node_tree.nodes.remove(node)

        if reset_keyframes:
            bpy.context.scene.frame_start = 0
            bpy.context.scene.frame_end = 0
            for a in bpy.data.actions:
                bpy.data.actions.remove(a)

    def gc(self):
        for _ in range(10):
            bpy.ops.outliner.orphans_purge()
