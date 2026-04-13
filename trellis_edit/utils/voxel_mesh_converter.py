"""Convert voxel coordinates to cubic mesh for visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import trimesh


def coords_to_cubic_mesh(
    coords: torch.Tensor,
    resolution: int = 64,
    voxel_size: Optional[float] = None,
    center_offset: float = -0.5,
) -> trimesh.Trimesh:
    """Convert voxel coordinates to cubic mesh.

    Each voxel is represented as a cube with 8 vertices and 12 triangles.

    Args:
        coords: Nx3 tensor of integer voxel coordinates
        resolution: Voxel grid resolution (default 64)
        voxel_size: Size of each voxel (default 1.0/resolution)
        center_offset: Offset to center the mesh (default -0.5)

    Returns:
        trimesh.Trimesh object with cubic voxel mesh
    """
    if voxel_size is None:
        voxel_size = 1.0 / resolution

    coords_np = coords.cpu().numpy() if isinstance(coords, torch.Tensor) else coords

    # Handle both [N, 3] and [N, 4] formats (with batch dimension)
    if coords_np.shape[1] == 4:
        # Remove batch dimension: [N, 4] -> [N, 3]
        coords_np = coords_np[:, 1:]
    elif coords_np.shape[1] != 3:
        raise ValueError(f"Expected coords shape [N, 3] or [N, 4], got {coords_np.shape}")

    vertices = []
    faces = []

    # Cube face indices (12 triangles for 6 faces), wound for outward normals
    cube_face_template = [
        # Front face (z=0)
        [0, 2, 1],
        [0, 3, 2],
        # Back face (z=1)
        [4, 5, 6],
        [4, 6, 7],
        # Left face (x=0)
        [0, 7, 3],
        [0, 4, 7],
        # Right face (x=1)
        [1, 6, 5],
        [1, 2, 6],
        # Bottom face (y=0)
        [0, 5, 4],
        [0, 1, 5],
        # Top face (y=1)
        [3, 6, 2],
        [3, 7, 6],
    ]

    for coord in coords_np:
        x, y, z = coord.astype(float) * voxel_size + center_offset

        # 8 vertices of the cube
        cube_vertices = [
            [x, y, z],
            [x + voxel_size, y, z],
            [x + voxel_size, y + voxel_size, z],
            [x, y + voxel_size, z],
            [x, y, z + voxel_size],
            [x + voxel_size, y, z + voxel_size],
            [x + voxel_size, y + voxel_size, z + voxel_size],
            [x, y + voxel_size, z + voxel_size],
        ]

        base_idx = len(vertices)
        vertices.extend(cube_vertices)

        # Add faces with offset
        for face in cube_face_template:
            faces.append([base_idx + face[0], base_idx + face[1], base_idx + face[2]])

    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)

    return trimesh.Trimesh(vertices=vertices, faces=faces)


def save_voxel_mesh(
    coords: torch.Tensor,
    output_path: Path,
    resolution: int = 64,
    voxel_size: Optional[float] = None,
) -> None:
    """Save voxel coordinates as cubic mesh in GLB format.

    Args:
        coords: Nx3 tensor of integer voxel coordinates
        output_path: Output file path (.glb)
        resolution: Voxel grid resolution (default 64)
        voxel_size: Size of each voxel (default 1.0/resolution)
    """
    mesh = coords_to_cubic_mesh(coords, resolution, voxel_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path))
    print(f"Saved voxel mesh to {output_path} ({len(coords)} voxels, {len(mesh.faces)} faces)")


def load_coords_from_file(coords_path: Path, device: torch.device) -> torch.Tensor:
    """Load voxel coordinates from file (.ply or .pt).

    Args:
        coords_path: Path to coords file
        device: Target torch device

    Returns:
        Nx3 tensor of integer coordinates
    """
    if coords_path.suffix == ".pt":
        coords = torch.load(coords_path, map_location=device)
    elif coords_path.suffix == ".ply":
        from trellis_edit.preprocess.asset_3d import ply_to_coords

        # Infer resolution from file metadata if available
        resolution = 64  # Default
        coords = ply_to_coords(coords_path, device, resolution)
    else:
        raise ValueError(f"Unsupported coords file format: {coords_path.suffix}")

    return coords


def save_coords_to_file(
    coords: Union[torch.Tensor, np.ndarray],
    output_path: Path,
    format: str = "ply",
    resolution: int = 64,
) -> None:
    """Save voxel coordinates to file.

    Args:
        coords: Nx3 tensor/array of integer coordinates [0, resolution-1]
        output_path: Output file path
        format: Output format ("ply" or "pt")
        resolution: Voxel grid resolution (for PLY metadata)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy array to torch tensor if needed
    if isinstance(coords, np.ndarray):
        coords = torch.from_numpy(coords)

    # Handle both [N, 3] and [N, 4] formats (with batch dimension)
    if coords.shape[1] == 4:
        # Remove batch dimension: [N, 4] -> [N, 3]
        coords = coords[:, 1:]
    elif coords.shape[1] != 3:
        raise ValueError(f"Expected coords shape [N, 3] or [N, 4], got {coords.shape}")

    if format == "pt":
        torch.save(coords.cpu(), output_path)
        print(f"Saved coords to {output_path} ({len(coords)} voxels)")
    elif format == "ply":
        # Convert coords to positions in [-0.5, 0.5] space
        positions = ((coords.float() + 0.5) / resolution) - 0.5
        positions_np = positions.cpu().numpy().astype(np.float32)

        # Save as PLY
        import utils3d
        utils3d.io.write_ply(str(output_path), positions_np)
        print(f"Saved coords to {output_path} ({len(coords)} voxels)")
