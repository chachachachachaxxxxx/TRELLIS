#!/usr/bin/env python3
"""Analyze mesh vertex overlap between source and edit GLB files."""

import argparse
from pathlib import Path
import numpy as np
import trimesh


def analyze_mesh_overlap(source_glb: Path, edit_glb: Path, tolerance: float = 1e-4):
    """Analyze vertex overlap between source and edit meshes.

    Args:
        source_glb: Path to source GLB file
        edit_glb: Path to edit GLB file
        tolerance: Distance tolerance for considering vertices as matching
    """

    print(f"Loading source mesh: {source_glb}")
    source_mesh = trimesh.load(source_glb, force='mesh')

    print(f"Loading edit mesh: {edit_glb}")
    edit_mesh = trimesh.load(edit_glb, force='mesh')

    # Get vertices
    src_verts = np.array(source_mesh.vertices)
    edit_verts = np.array(edit_mesh.vertices)

    print(f"\n=== Basic Statistics ===")
    print(f"Source vertices: {len(src_verts)}")
    print(f"Edit vertices: {len(edit_verts)}")
    print(f"Source faces: {len(source_mesh.faces)}")
    print(f"Edit faces: {len(edit_mesh.faces)}")

    # Compute bounding boxes
    src_bbox = src_verts.max(axis=0) - src_verts.min(axis=0)
    edit_bbox = edit_verts.max(axis=0) - edit_verts.min(axis=0)

    print(f"\n=== Bounding Box ===")
    print(f"Source: {src_bbox}")
    print(f"Edit: {edit_bbox}")

    # Find matching vertices using KDTree for efficiency
    from scipy.spatial import cKDTree

    print(f"\n=== Vertex Overlap Analysis (tolerance={tolerance}) ===")

    # Build KDTree for source vertices
    src_tree = cKDTree(src_verts)

    # For each edit vertex, find nearest source vertex
    distances, indices = src_tree.query(edit_verts, k=1)

    # Count matches within tolerance
    matches = distances < tolerance
    num_matches = matches.sum()

    print(f"Matching vertices: {num_matches} ({num_matches / len(edit_verts) * 100:.2f}% of edit)")
    print(f"Non-matching vertices (newly generated): {len(edit_verts) - num_matches} ({(len(edit_verts) - num_matches) / len(edit_verts) * 100:.2f}% of edit)")

    # Analyze distance distribution
    print(f"\n=== Distance Distribution ===")
    print(f"Min distance: {distances.min():.6f}")
    print(f"Max distance: {distances.max():.6f}")
    print(f"Mean distance: {distances.mean():.6f}")
    print(f"Median distance: {np.median(distances):.6f}")

    # Percentiles
    percentiles = [50, 75, 90, 95, 99]
    print(f"\nDistance percentiles:")
    for p in percentiles:
        val = np.percentile(distances, p)
        print(f"  {p}th: {val:.6f}")

    # Analyze which source vertices are matched
    unique_matched_src = np.unique(indices[matches])
    print(f"\n=== Source Vertex Usage ===")
    print(f"Source vertices matched: {len(unique_matched_src)} ({len(unique_matched_src) / len(src_verts) * 100:.2f}%)")
    print(f"Source vertices unused: {len(src_verts) - len(unique_matched_src)} ({(len(src_verts) - len(unique_matched_src)) / len(src_verts) * 100:.2f}%)")

    # Try different tolerance levels
    print(f"\n=== Sensitivity Analysis ===")
    for tol in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        matches_tol = (distances < tol).sum()
        print(f"Tolerance {tol:.0e}: {matches_tol} matches ({matches_tol / len(edit_verts) * 100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze mesh vertex overlap")
    parser.add_argument("--source", type=str, required=True, help="Source GLB file")
    parser.add_argument("--edit", type=str, required=True, help="Edit GLB file")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Distance tolerance")

    args = parser.parse_args()

    analyze_mesh_overlap(
        Path(args.source),
        Path(args.edit),
        args.tolerance
    )
