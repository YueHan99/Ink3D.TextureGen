"""
Export vxz + mesh → GLB file.

Usage:
    conda run -n trellis2 python export_vxz_glb.py \
        --vxz path/to.vxz --mesh path/to.pickle -o output.glb

    conda run -n trellis2 python export_vxz_glb.py \
        --vxz path/to.vxz --mesh path/to.glb -o output.glb --texture_size 4096
"""
import os
import argparse
import pickle
import numpy as np
import torch
import trimesh
import o_voxel


def load_mesh_pickle(path):
    """Load pickle mesh (Blender Z-up) → vertices/faces, keep original coords."""
    with open(path, 'rb') as f:
        dump = pickle.load(f)
    objects = [obj for obj in dump['objects'] if obj['vertices'].size > 0 and obj['faces'].size > 0]
    if not objects:
        raise ValueError(f"No valid geometry in {path}")
    all_vertices = np.concatenate([obj['vertices'] for obj in objects], axis=0)
    all_faces = []
    offset = 0
    for obj in objects:
        all_faces.append(obj['faces'] + offset)
        offset += len(obj['vertices'])
    all_faces = np.concatenate(all_faces, axis=0)
    # Keep original Z-up coords (same space as vxz)
    return torch.tensor(all_vertices, dtype=torch.float32), torch.tensor(all_faces, dtype=torch.int32)


def load_mesh_file(path):
    """Load GLB/OBJ/PLY mesh via trimesh → vertices/faces."""
    mesh = trimesh.load(path, force='mesh')
    return (
        torch.tensor(np.array(mesh.vertices), dtype=torch.float32),
        torch.tensor(np.array(mesh.faces), dtype=torch.int32),
    )


def load_vxz(path, metallic=0.0, roughness=0.9):
    """Load vxz → coords (N,3), attrs (N,6) in [0,1]."""
    coords, attr = o_voxel.io.read_vxz(path, num_threads=4)
    if 'top6' in attr:
        attr['base_color'] = attr['top6'][:, :3]
        attr['roughness'] = torch.ones_like(attr['base_color'][:, :1]) * 255 * roughness
        attr['metallic'] = torch.ones_like(attr['base_color'][:, :1]) * 255 * metallic
        attr['alpha'] = torch.ones_like(attr['base_color'][:, :1]) * 255
    attrs = torch.cat([
        attr['base_color'], attr['metallic'], attr['roughness'], attr['alpha']
    ], dim=-1).float() / 255.0
    return coords, attrs


def main():
    parser = argparse.ArgumentParser(description="Export vxz + mesh → GLB")
    parser.add_argument("--vxz", type=str, required=True, help="Path to .vxz file")
    parser.add_argument("--mesh", type=str, required=True, help="Mesh file (.pickle / .glb / .obj / .ply)")
    parser.add_argument("-o", "--output", type=str, default="output.glb", help="Output GLB path")
    parser.add_argument("--voxel_resolution", type=int, default=1024, help="Voxel grid resolution")
    parser.add_argument("--texture_size", type=int, default=4096, help="UV texture resolution")
    parser.add_argument("--decimation_target", type=int, default=1000000, help="Target face count")
    parser.add_argument("--metallic", type=float, default=0.0, help="Metallic value (0-1)")
    parser.add_argument("--roughness", type=float, default=0.9, help="Roughness value (0-1)")
    args = parser.parse_args()

    # Load mesh
    ext = os.path.splitext(args.mesh)[1].lower()
    if ext == '.pickle':
        print(f"Loading pickle mesh: {args.mesh}")
        vertices, faces = load_mesh_pickle(args.mesh)
    elif ext in ('.glb', '.gltf', '.obj', '.ply', '.stl'):
        print(f"Loading mesh: {args.mesh}")
        vertices, faces = load_mesh_file(args.mesh)
    else:
        raise ValueError(f"Unsupported mesh format: {ext}")
    print(f"  Vertices: {vertices.shape[0]}, Faces: {faces.shape[0]}")

    # Load vxz
    print(f"Loading vxz: {args.vxz}")
    coords, attrs = load_vxz(args.vxz, metallic=args.metallic, roughness=args.roughness)
    print(f"  Voxels: {coords.shape[0]}, Attrs: {attrs.shape[1]}ch")

    # Export GLB
    print(f"Exporting GLB (texture_size={args.texture_size}, decimation={args.decimation_target})...")
    glb = o_voxel.postprocess.to_glb(
        vertices=vertices.cuda(),
        faces=faces.cuda(),
        attr_volume=attrs.cuda(),
        coords=coords.cuda(),
        attr_layout={
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        },
        voxel_size=1.0 / args.voxel_resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=args.decimation_target,
        texture_size=args.texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    glb.export(args.output, extension_webp=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
