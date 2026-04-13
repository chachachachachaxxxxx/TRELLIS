"""Voxel filtering utilities for mask processing.

This module provides functions to filter voxels based on geometric intersection
with a target mesh. It supports multiple filtering strategies:
- Volume intersection: Most accurate, samples multiple points within each voxel
- Distance threshold: Good for boundary preservation, computationally efficient
- Corner sampling: Balanced approach, samples 8 corner points of each voxel
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
import trimesh
from pathlib import Path
from typing import Tuple, Optional

try:
    from pysdf import SDF
    HAS_PYSDF = True
except ImportError:
    HAS_PYSDF = False


def load_trimesh(model_path: str | Path) -> trimesh.Trimesh:
    """Load a mesh from file and ensure it's a single Trimesh object.

    Args:
        model_path: Path to mesh file (GLB, PLY, etc.)

    Returns:
        Trimesh object
    """
    scene = trimesh.load(str(model_path), force="mesh", process=False)
    if isinstance(scene, trimesh.Trimesh):
        return scene
    elif isinstance(scene, trimesh.scene.Scene):
        # Merge all geometries in scene
        mesh = trimesh.Trimesh()
        for obj in scene.geometry.values():
            mesh = trimesh.util.concatenate([mesh, obj])
        return mesh
    else:
        raise ValueError(f"Unknown mesh type at {model_path}")


def create_sdf(mesh: trimesh.Trimesh):
    """Create signed distance function from mesh.

    Args:
        mesh: Trimesh object

    Returns:
        SDF function that takes points and returns signed distances
    """
    if not HAS_PYSDF:
        raise ImportError(
            "pysdf is required for SDF-based voxel filtering. "
            "Install with: pip install pysdf"
        )
    return SDF(mesh.vertices, mesh.faces)


def check_voxels_with_volume_intersection(
    source_points: np.ndarray,
    sdf,
    voxel_size: float = 1 / 64,
    threshold: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Volume intersection-based voxel filtering.

    Samples 27 points (3x3x3 grid) within each voxel to estimate intersection
    with target geometry. Most accurate but slowest method.

    Args:
        source_points: Voxel center coordinates [N, 3]
        sdf: Signed distance function
        voxel_size: Size of each voxel cube
        threshold: Intersection threshold (0.0-1.0), percentage of voxel volume
                  that must be inside geometry

    Returns:
        inside_mask: Boolean array, True indicates voxel intersects with geometry
        overlap_ratios: Intersection ratio for each voxel (0.0-1.0)
    """
    inside_mask = np.zeros(len(source_points), dtype=bool)
    overlap_ratios = np.zeros(len(source_points))

    half_voxel = voxel_size / 2

    for i, center in enumerate(source_points):
        # Sample 3x3x3 grid within voxel
        samples = []
        for x in np.linspace(-half_voxel, half_voxel, 3):
            for y in np.linspace(-half_voxel, half_voxel, 3):
                for z in np.linspace(-half_voxel, half_voxel, 3):
                    sample_point = center + np.array([x, y, z])
                    samples.append(sample_point)

        samples = np.array(samples)
        distances = sdf(samples)
        inside_samples = distances > 0

        overlap_ratio = np.mean(inside_samples)
        overlap_ratios[i] = overlap_ratio

        if overlap_ratio >= threshold:
            inside_mask[i] = True

    return inside_mask, overlap_ratios


def check_voxels_with_distance_threshold(
    source_points: np.ndarray,
    sdf,
    voxel_size: float = 1 / 64,
    distance_threshold: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Distance threshold-based voxel filtering.

    Keeps voxels inside geometry and those close to surface. Good for
    preserving boundary details.

    Args:
        source_points: Voxel center coordinates [N, 3]
        sdf: Signed distance function
        voxel_size: Size of each voxel cube
        distance_threshold: Maximum distance from voxel center to geometry surface

    Returns:
        inside_mask: Boolean array, True indicates voxel intersects with geometry
        distances: Signed distance from each voxel center to geometry surface
    """
    distances = sdf(source_points)

    # Keep voxels inside geometry or close to surface
    outside_mask = distances <= 0
    near_surface_mask = np.abs(distances) <= distance_threshold
    inside_mask = (distances > 0) | (near_surface_mask & outside_mask)

    return inside_mask, distances


def check_voxels_with_corner_sampling(
    source_points: np.ndarray,
    sdf,
    voxel_size: float = 1 / 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Corner sampling-based voxel filtering.

    Samples 8 corner points of each voxel. Good balance between accuracy
    and computational cost.

    Args:
        source_points: Voxel center coordinates [N, 3]
        sdf: Signed distance function
        voxel_size: Size of each voxel cube

    Returns:
        inside_mask: Boolean array, True indicates voxel intersects with geometry
        corner_counts: Number of corner points inside geometry for each voxel (0-8)
    """
    inside_mask = np.zeros(len(source_points), dtype=bool)
    corner_counts = np.zeros(len(source_points), dtype=int)

    half_voxel = voxel_size / 2

    for i, center in enumerate(source_points):
        # Sample 8 corner points
        corners = []
        for x in [-half_voxel, half_voxel]:
            for y in [-half_voxel, half_voxel]:
                for z in [-half_voxel, half_voxel]:
                    corner = center + np.array([x, y, z])
                    corners.append(corner)

        corners = np.array(corners)
        distances = sdf(corners)
        inside_corners = distances > 0
        corner_count = np.sum(inside_corners)
        corner_counts[i] = corner_count

        if corner_count > 0:
            inside_mask[i] = True

    return inside_mask, corner_counts


def adaptive_voxel_filtering(
    source_points: np.ndarray,
    sdf,
    voxel_size: float = 1 / 64,
    method: str = "volume",
) -> Tuple[np.ndarray, np.ndarray]:
    """Adaptive voxel filtering with automatic parameter adjustment.

    Args:
        source_points: Voxel center coordinates [N, 3]
        sdf: Signed distance function
        voxel_size: Size of each voxel cube
        method: Filtering method ('volume', 'distance', 'corner')

    Returns:
        inside_mask: Boolean array, True indicates voxel intersects with geometry
        additional_info: Method-specific additional information
    """
    if method == "volume":
        # Adapt threshold based on voxel size
        if voxel_size >= 1 / 32:
            threshold = 0.05
        elif voxel_size >= 1 / 64:
            threshold = 0.1
        else:
            threshold = 0.2

        return check_voxels_with_volume_intersection(
            source_points, sdf, voxel_size, threshold
        )

    elif method == "distance":
        distance_threshold = voxel_size * 0.5
        return check_voxels_with_distance_threshold(
            source_points, sdf, voxel_size, distance_threshold
        )

    elif method == "corner":
        return check_voxels_with_corner_sampling(source_points, sdf, voxel_size)

    else:
        raise ValueError(f"Unknown method: {method}")


def process_voxels_with_improved_filtering(
    source_voxel_path: str | Path,
    mask_model_path: str | Path,
    output_path: str | Path,
    method: str = "volume",
    voxel_size: float = 1 / 64,
    inside: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Process voxels using improved filtering methods.

    Complete pipeline for filtering voxels based on geometric intersection
    with a target mesh.

    Args:
        source_voxel_path: Path to source voxel point cloud (.ply)
        mask_model_path: Path to mask geometry (.ply, .glb, etc.)
        output_path: Path to save filtered voxel point cloud (.ply)
        method: Filtering method ('volume', 'distance', 'corner')
        voxel_size: Size of voxels in the source data
        inside: If True, keep voxels inside geometry; if False, keep outside

    Returns:
        target_voxel_points: Filtered voxel coordinates
        additional_info: Method-specific additional information
    """
    # Load data
    mask_mesh = load_trimesh(mask_model_path)
    sdf = create_sdf(mask_mesh)

    source_pcd = o3d.io.read_point_cloud(str(source_voxel_path))
    source_points = np.asarray(source_pcd.points)

    # Apply filtering
    inside_mask, additional_info = adaptive_voxel_filtering(
        source_points, sdf, voxel_size, method
    )

    mask = inside_mask if inside else ~inside_mask
    target_voxel_points = source_points[mask]

    # Save results
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_voxel_points)
    o3d.io.write_point_cloud(str(output_path), target_pcd)

    print(f"Original voxel count: {len(source_points)}")
    print(f"Retained voxel count: {len(target_voxel_points)}")
    print(f"Filtered voxel count: {len(source_points) - len(target_voxel_points)}")

    if method == "volume":
        print(f"Average overlap ratio: {np.mean(additional_info):.3f}")
    elif method == "distance":
        print(f"Average distance: {np.mean(additional_info):.3f}")
    elif method == "corner":
        print(f"Average corner count: {np.mean(additional_info):.1f}")

    return target_voxel_points, additional_info
