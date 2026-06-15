"""GLB/OBJ model importer for Blender."""

import bpy
import os


def load_file(path: str):
    """Import a 3D model file into Blender.

    Supports: .glb (glTF), .obj, .fbx, .ply
    Also handles OBJ files disguised as .glb (detected via magic bytes).

    Args:
        path: Path to the 3D model file.
    """
    if path.endswith(".glb"):
        with open(path, "rb") as f:
            magic = f.read(4)
        if magic == b"glTF":
            bpy.ops.import_scene.gltf(filepath=path)
        else:
            # OBJ disguised as .glb
            obj_link = os.path.splitext(path)[0] + ".obj"
            created_link = False
            try:
                if not os.path.exists(obj_link):
                    os.symlink(path, obj_link)
                    created_link = True
                bpy.ops.wm.obj_import(filepath=obj_link)
            finally:
                if created_link and os.path.islink(obj_link):
                    os.remove(obj_link)
    elif path.endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=path)
    elif path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=path)
    elif path.endswith(".ply"):
        bpy.ops.wm.ply_import(filepath=path)
    else:
        raise RuntimeError(f"Unsupported file format: {path}")
