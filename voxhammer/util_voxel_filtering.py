import numpy as np
import open3d as o3d

from voxhammer.util_pysdf import *


def check_voxels_with_volume_intersection(
    source_points, sdf, voxel_size=1 / 64, threshold=0.1
):
    """
    Volume intersection-based voxel filtering method

    This method samples multiple points within each voxel to estimate the intersection
    between the voxel volume and the target geometry. It provides the most accurate
    representation of voxel-geometry overlap.

    Principle:
    - For each voxel, sample 27 points in a 3x3x3 grid pattern within the voxel bounds
    - Calculate what percentage of these sampled points lie inside the target geometry
    - If the intersection ratio exceeds the threshold, mark the voxel as intersecting

    Advantages:
    - Most accurate intersection estimation among all methods
    - Provides quantitative overlap ratio for each voxel
    - Can be fine-tuned with different threshold values
    - Handles complex geometry shapes well

    Use cases:
    - When precise control over intersection tolerance is required
    - For applications where quantitative overlap information is needed
    - When dealing with complex or irregular geometry shapes
    - Research applications requiring detailed analysis

    Args:
        source_points: voxel center coordinates [N, 3]
        sdf: signed distance function of the target geometry
        voxel_size: size of each voxel cube
        threshold: intersection threshold (0.0-1.0), percentage of voxel volume that must be inside geometry

    Returns:
        inside_mask: boolean array, True indicates voxel intersects with geometry
        overlap_ratios: intersection ratio for each voxel (0.0-1.0)
    """
    inside_mask = np.zeros(len(source_points), dtype=bool)
    overlap_ratios = np.zeros(len(source_points))

    # Calculate voxel half-size
    half_voxel = voxel_size / 2

    for i, center in enumerate(source_points):
        # Sample multiple points within the voxel to estimate intersection
        num_samples = 27  # 3x3x3 grid sampling
        samples = []

        for x in np.linspace(-half_voxel, half_voxel, 3):
            for y in np.linspace(-half_voxel, half_voxel, 3):
                for z in np.linspace(-half_voxel, half_voxel, 3):
                    sample_point = center + np.array([x, y, z])
                    samples.append(sample_point)

        samples = np.array(samples)

        # Check if sampled points are inside the geometry
        distances = sdf(samples)
        inside_samples = distances > 0

        # Calculate intersection ratio
        overlap_ratio = np.mean(inside_samples)
        overlap_ratios[i] = overlap_ratio

        # Mark voxel as intersecting if ratio exceeds threshold
        if overlap_ratio >= threshold:
            inside_mask[i] = True

    return inside_mask, overlap_ratios


def check_voxels_with_distance_threshold(
    source_points, sdf, voxel_size=1 / 64, distance_threshold=0.05
):
    """
    Distance threshold-based voxel filtering method

    This method considers both voxels inside the geometry and those close to the
    geometry surface. It's particularly effective for preserving boundary details
    and handling cases where voxel centers are outside but the voxel volume
    intersects with the geometry.

    Principle:
    - Calculate signed distance from each voxel center to the geometry surface
    - Keep voxels that are inside the geometry (positive SDF)
    - Also keep voxels outside the geometry if their distance to surface is within threshold
    - The threshold is typically set to a fraction of voxel size

    Advantages:
    - Simple and computationally efficient
    - Preserves boundary details effectively
    - Handles thin geometry features well
    - Intuitive parameter tuning (distance threshold)

    Use cases:
    - When boundary preservation is important
    - For thin or fine geometric features
    - Applications requiring computational efficiency
    - When dealing with geometry that has small protrusions or details

    Args:
        source_points: voxel center coordinates [N, 3]
        sdf: signed distance function of the target geometry
        voxel_size: size of each voxel cube
        distance_threshold: maximum distance from voxel center to geometry surface to be considered intersecting

    Returns:
        inside_mask: boolean array, True indicates voxel should be filtered out
        distances: signed distance from each voxel center to geometry surface
    """
    distances = sdf(source_points)

    # For voxels outside geometry, also consider those close to the surface
    outside_mask = distances <= 0
    near_surface_mask = np.abs(distances) <= distance_threshold

    # Keep voxels inside geometry or close to surface
    inside_mask = (distances > 0) | (near_surface_mask & outside_mask)

    return inside_mask, distances


def check_voxels_with_corner_sampling(source_points, sdf, voxel_size=1 / 64):
    """
    Corner sampling-based voxel filtering method

    This method samples the 8 corner points of each voxel cube and checks if any
    of these corners lie inside the target geometry. It provides a good balance
    between accuracy and computational efficiency.

    Principle:
    - For each voxel, calculate the 8 corner points of the voxel cube
    - Check if any corner point is inside the target geometry using SDF
    - If at least one corner is inside, mark the voxel as intersecting
    - This ensures that any voxel containing part of the geometry is captured

    Advantages:
    - Good balance between accuracy and computational cost
    - Captures most intersection cases effectively
    - Simple and easy to understand
    - Provides corner count information for analysis

    Use cases:
    - General-purpose voxel filtering applications
    - When balanced performance is required
    - For applications where some false positives are acceptable
    - When computational resources are moderate

    Args:
        source_points: voxel center coordinates [N, 3]
        sdf: signed distance function of the target geometry
        voxel_size: size of each voxel cube

    Returns:
        inside_mask: boolean array, True indicates voxel intersects with geometry
        corner_counts: number of corner points inside geometry for each voxel (0-8)
    """
    inside_mask = np.zeros(len(source_points), dtype=bool)
    corner_counts = np.zeros(len(source_points), dtype=int)

    half_voxel = voxel_size / 2

    for i, center in enumerate(source_points):
        # Sample the 8 corner points of the voxel cube
        corners = []
        for x in [-half_voxel, half_voxel]:
            for y in [-half_voxel, half_voxel]:
                for z in [-half_voxel, half_voxel]:
                    corner = center + np.array([x, y, z])
                    corners.append(corner)

        corners = np.array(corners)

        # Check if corner points are inside the geometry
        distances = sdf(corners)
        inside_corners = distances > 0
        corner_count = np.sum(inside_corners)
        corner_counts[i] = corner_count

        # Mark voxel as intersecting if at least one corner is inside
        if corner_count > 0:
            inside_mask[i] = True

    return inside_mask, corner_counts


def adaptive_voxel_filtering(source_points, sdf, voxel_size=1 / 64, method="volume"):
    """
    Adaptive voxel filtering with automatic parameter adjustment

    This function automatically selects appropriate parameters based on voxel size
    and applies the chosen filtering method. It provides a convenient interface
    for different filtering strategies.

    Parameter adaptation:
    - Volume method: threshold scales with voxel size (larger voxels = lower threshold)
    - Distance method: threshold is proportional to voxel size
    - Corner method: no parameter adaptation needed

    Args:
        source_points: voxel center coordinates [N, 3]
        sdf: signed distance function of the target geometry
        voxel_size: size of each voxel cube
        method: filtering method ('volume', 'distance', 'corner')

    Returns:
        inside_mask: boolean array, True indicates voxel should be filtered out
        additional_info: additional information (overlap ratios, distances, corner counts)
    """
    if method == "volume":
        # Adapt threshold based on voxel size
        if voxel_size >= 1 / 32:
            threshold = 0.05  # Large voxels: more lenient threshold
        elif voxel_size >= 1 / 64:
            threshold = 0.1  # Medium voxels: moderate threshold
        else:
            threshold = 0.2  # Small voxels: stricter threshold

        return check_voxels_with_volume_intersection(
            source_points, sdf, voxel_size, threshold
        )

    elif method == "distance":
        # Adapt distance threshold based on voxel size
        distance_threshold = voxel_size * 0.5  # Half voxel size
        return check_voxels_with_distance_threshold(
            source_points, sdf, voxel_size, distance_threshold
        )

    elif method == "corner":
        return check_voxels_with_corner_sampling(source_points, sdf, voxel_size)

    else:
        raise ValueError(f"Unknown method: {method}")


def process_voxels_with_improved_filtering(
    source_voxel_path,
    mask_model_path,
    output_path,
    method="volume",
    voxel_size=1 / 64,
    inside=False,
):
    """
    Process voxels using improved filtering methods

    This function provides a complete pipeline for filtering voxels based on
    geometric intersection with a target mesh. It loads the source voxels and
    mask geometry, applies the chosen filtering method, and saves the results.

    The function supports three different filtering strategies:
    1. Volume intersection: Most accurate, suitable for precise applications
    2. Distance threshold: Good for boundary preservation, computationally efficient
    3. Corner sampling: Balanced approach, good for general use

    Args:
        source_voxel_path: path to source voxel point cloud file (.ply)
        mask_model_path: path to mask geometry file (.ply)
        output_path: path to save filtered voxel point cloud (.ply)
        method: filtering method ('volume', 'distance', 'corner')
        voxel_size: size of voxels in the source data

    Returns:
        target_voxel_points: filtered voxel coordinates
        additional_info: method-specific additional information
    """
    # Load data
    mask_mesh = load_trimesh(mask_model_path)
    sdf = load_and_create_sdf(mask_mesh)

    source_pcd = o3d.io.read_point_cloud(source_voxel_path)
    source_points = np.asarray(source_pcd.points)

    # Apply improved filtering method
    inside_mask, additional_info = adaptive_voxel_filtering(
        source_points, sdf, voxel_size, method
    )

    mask = inside_mask if inside else ~inside_mask

    # Keep voxels outside the geometry
    target_voxel_points = source_points[mask]

    # Save results
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_voxel_points)
    o3d.io.write_point_cloud(output_path, target_pcd)

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
