#!/usr/bin/env python3
"""Generate voxels_delete.ply from mask GLB file."""

import argparse
import numpy as np
import trimesh
import torch
from pathlib import Path


def voxelize_mesh_simple(mesh, resolution=64):
    """Simple voxelization using trimesh.

    Args:
        mesh: trimesh mesh object
        resolution: voxel grid resolution

    Returns:
        coords: (N, 4) tensor with [batch, x, y, z] coordinates
    """
    # Normalize mesh to [-0.5, 0.5] range
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    scale = (bounds[1] - bounds[0]).max()

    mesh_normalized = mesh.copy()
    mesh_normalized.vertices = (mesh_normalized.vertices - center) / scale

    # Voxelize
    pitch = 1.0 / resolution
    voxel_grid = mesh_normalized.voxelized(pitch=pitch)

    # Get filled voxels
    filled = voxel_grid.matrix

    # Convert to coordinates
    coords_np = np.argwhere(filled)

    # Add batch dimension
    batch = np.zeros((coords_np.shape[0], 1), dtype=np.int32)
    coords_with_batch = np.concatenate([batch, coords_np], axis=1)

    return torch.from_numpy(coords_with_batch).int()


def coords_to_ply(coords, output_path, resolution=64):
    """Save coordinates as PLY file.

    Args:
        coords: (N, 4) tensor with [batch, x, y, z]
        output_path: output PLY path
        resolution: grid resolution
    """
    # Convert to world coordinates
    coords_xyz = coords[:, 1:].cpu().numpy().astype(np.float32)

    # Scale to [-0.5, 0.5] range
    coords_xyz = (coords_xyz / resolution) - 0.5

    # Create point cloud
    vertices = coords_xyz

    # Write PLY
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")

    print(f"Saved {len(vertices)} voxels to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate voxels_delete.ply from mask GLB")
    parser.add_argument("--mask-glb", required=True, help="Path to mask GLB file")
    parser.add_argument("--output", required=True, help="Output voxels_delete.ply path")
    parser.add_argument("--resolution", type=int, default=64, help="Voxel grid resolution")
    args = parser.parse_args()

    # Load mask mesh
    print(f"Loading mask: {args.mask_glb}")
    mesh = trimesh.load(args.mask_glb, force='mesh')

    if isinstance(mesh, trimesh.Scene):
        # Merge all geometries in scene
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
            for g in mesh.geometry.values()
        ])

    print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    # Voxelize
    print(f"Voxelizing at resolution {args.resolution}...")
    coords = voxelize_mesh_simple(mesh, resolution=args.resolution)
    print(f"Generated {coords.shape[0]} voxels")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coords_to_ply(coords, output_path, resolution=args.resolution)

    print("Done!")


if __name__ == "__main__":
    main()
