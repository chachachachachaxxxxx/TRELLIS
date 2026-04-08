#!/usr/bin/env python3
"""Analyze voxel overlap between source and edit latents (lightweight version)."""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image


def build_coord_hash(coords_3d):
    """Build hash map from 3D coords to indices."""
    hash_to_idx = {}
    for i, coord in enumerate(coords_3d):
        x, y, z = coord
        hash_str = f"{int(x)}_{int(y)}_{int(z)}"
        hash_to_idx[hash_str] = i
    return hash_to_idx


def analyze_overlap(source_npz: Path, edit_npz: Path, mask_path: Path = None):
    """Analyze voxel overlap between source and edit."""

    print("Loading source features...")
    source_data = np.load(source_npz)
    src_coords = source_data['coords']  # [N_src, 4] - (batch, x, y, z)
    src_coords_3d = src_coords[:, 1:]  # [N_src, 3]

    print("Loading edit features...")
    edit_data = np.load(edit_npz)
    edit_coords = edit_data['coords']  # [N_edit, 4]
    edit_coords_3d = edit_coords[:, 1:]  # [N_edit, 3]

    print(f"\nSource voxels: {len(src_coords_3d)}")
    print(f"Edit voxels: {len(edit_coords_3d)}")

    # Build hash maps
    src_hash_to_idx = build_coord_hash(src_coords_3d)
    edit_hash_to_idx = build_coord_hash(edit_coords_3d)

    # Find overlaps
    overlap_hashes = set(src_hash_to_idx.keys()) & set(edit_hash_to_idx.keys())

    print(f"\nOverlapping voxels: {len(overlap_hashes)}")
    print(f"Overlap ratio (vs source): {len(overlap_hashes) / len(src_coords_3d) * 100:.2f}%")
    print(f"Overlap ratio (vs edit): {len(overlap_hashes) / len(edit_coords_3d) * 100:.2f}%")

    # Calculate unique voxels
    src_only = len(src_hash_to_idx) - len(overlap_hashes)
    edit_only = len(edit_hash_to_idx) - len(overlap_hashes)

    print(f"\nSource-only voxels: {src_only} ({src_only / len(src_coords_3d) * 100:.2f}%)")
    print(f"Edit-only voxels (newly generated): {edit_only} ({edit_only / len(edit_coords_3d) * 100:.2f}%)")

    # Analyze mask coverage if provided
    if mask_path and mask_path.exists():
        mask_img = Image.open(mask_path).convert('L')
        mask_array = np.array(mask_img) / 255.0

        # Get resolution from source
        resolution = int(src_coords_3d[:, 0].max()) + 1
        mask_h, mask_w = mask_array.shape

        print(f"\n=== Mask Analysis ===")
        print(f"Mask size: {mask_w}x{mask_h}")
        print(f"Voxel resolution: {resolution}")
        print(f"Mask coverage (white pixels): {(mask_array > 0.5).sum() / mask_array.size * 100:.2f}%")

        # Analyze overlap by mask region
        masked_overlap = 0
        unmasked_overlap = 0

        for hash_str in overlap_hashes:
            src_idx = src_hash_to_idx[hash_str]
            x, y, _z = src_coords_3d[src_idx]

            # Project to 2D
            u = int((x / resolution) * mask_w)
            v = int((y / resolution) * mask_h)
            u = max(0, min(mask_w - 1, u))
            v = max(0, min(mask_h - 1, v))

            mask_value = mask_array[v, u]

            if mask_value > 0.5:
                masked_overlap += 1
            else:
                unmasked_overlap += 1

        print(f"\nOverlap in masked region (edit area): {masked_overlap} ({masked_overlap / len(overlap_hashes) * 100:.2f}%)")
        print(f"Overlap in unmasked region (preserve area): {unmasked_overlap} ({unmasked_overlap / len(overlap_hashes) * 100:.2f}%)")

        # Analyze edit-only voxels by mask
        edit_only_hashes = set(edit_hash_to_idx.keys()) - set(src_hash_to_idx.keys())
        edit_only_masked = 0
        edit_only_unmasked = 0

        for hash_str in edit_only_hashes:
            edit_idx = edit_hash_to_idx[hash_str]
            x, y, _z = edit_coords_3d[edit_idx]

            u = int((x / resolution) * mask_w)
            v = int((y / resolution) * mask_h)
            u = max(0, min(mask_w - 1, u))
            v = max(0, min(mask_h - 1, v))

            mask_value = mask_array[v, u]

            if mask_value > 0.5:
                edit_only_masked += 1
            else:
                edit_only_unmasked += 1

        print(f"\n=== Newly Generated Voxels (Edit-only) ===")
        print(f"In masked region: {edit_only_masked} ({edit_only_masked / len(edit_only_hashes) * 100:.2f}%)")
        print(f"In unmasked region: {edit_only_unmasked} ({edit_only_unmasked / len(edit_only_hashes) * 100:.2f}%)")

    # Analyze coordinate distribution
    print(f"\n=== Coordinate Ranges ===")
    print(f"Source:")
    print(f"  X: [{src_coords_3d[:, 0].min():.1f}, {src_coords_3d[:, 0].max():.1f}]")
    print(f"  Y: [{src_coords_3d[:, 1].min():.1f}, {src_coords_3d[:, 1].max():.1f}]")
    print(f"  Z: [{src_coords_3d[:, 2].min():.1f}, {src_coords_3d[:, 2].max():.1f}]")

    print(f"\nEdit:")
    print(f"  X: [{edit_coords_3d[:, 0].min():.1f}, {edit_coords_3d[:, 0].max():.1f}]")
    print(f"  Y: [{edit_coords_3d[:, 1].min():.1f}, {edit_coords_3d[:, 1].max():.1f}]")
    print(f"  Z: [{edit_coords_3d[:, 2].min():.1f}, {edit_coords_3d[:, 2].max():.1f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze voxel overlap")
    parser.add_argument("--source-npz", type=str, required=True, help="Source features.npz path")
    parser.add_argument("--edit-npz", type=str, required=True, help="Edit features.npz path")
    parser.add_argument("--mask", type=str, help="Mask image path (optional)")

    args = parser.parse_args()

    analyze_overlap(
        Path(args.source_npz),
        Path(args.edit_npz),
        Path(args.mask) if args.mask else None
    )
