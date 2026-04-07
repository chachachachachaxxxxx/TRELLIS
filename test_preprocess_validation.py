#!/usr/bin/env python3
"""Validation script for editing/preprocess module."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Test 2D image preprocessing
print("=" * 60)
print("Testing 2D Image Preprocessing")
print("=" * 60)

from editing.preprocess import (
    PROC_IMAGE_SIZE,
    build_auto_mask,
    build_blank_mask,
    extract_mask_channel,
    prepare_aligned_inputs,
    save_preprocessed_inputs,
)

# Create test images
test_size = (512, 512)
source_img = Image.new("RGB", test_size, color=(255, 0, 0))
edit_img = Image.new("RGB", test_size, color=(0, 255, 0))
mask_img = Image.new("L", test_size, color=128)

print(f"✓ Created test images: {test_size}")
print(f"✓ PROC_IMAGE_SIZE: {PROC_IMAGE_SIZE}")

# Test mask utilities
blank_mask = build_blank_mask(test_size)
assert blank_mask.size == test_size
assert blank_mask.mode == "L"
print("✓ build_blank_mask works")

auto_mask = build_auto_mask(source_img, edit_img, threshold=10, max_filter=3)
assert auto_mask.size == test_size
assert auto_mask.mode == "L"
print("✓ build_auto_mask works")

extracted = extract_mask_channel(mask_img)
assert extracted.mode == "L"
print("✓ extract_mask_channel works")

# Test aligned inputs preparation (without pipeline, preprocess=False)
try:
    prepared = prepare_aligned_inputs(
        source_image=source_img,
        edit_image=edit_img,
        mask_image=mask_img,
        pipeline=None,
        preprocess=False,
        mask_threshold=128,
    )
    assert prepared.source.size == (PROC_IMAGE_SIZE, PROC_IMAGE_SIZE)
    assert prepared.edit.size == (PROC_IMAGE_SIZE, PROC_IMAGE_SIZE)
    assert prepared.mask.size == (PROC_IMAGE_SIZE, PROC_IMAGE_SIZE)
    assert prepared.meta["preprocess"] is False
    assert prepared.meta["proc_size"] == [PROC_IMAGE_SIZE, PROC_IMAGE_SIZE]
    print("✓ prepare_aligned_inputs (preprocess=False) works")
    print(f"  - Output size: {prepared.source.size}")
    print(f"  - Metadata keys: {list(prepared.meta.keys())}")
except Exception as e:
    print(f"✗ prepare_aligned_inputs failed: {e}")
    sys.exit(1)

# Test 3D asset preprocessing
print("\n" + "=" * 60)
print("Testing 3D Asset Preprocessing")
print("=" * 60)

from editing.preprocess import (
    VoxelNormalization,
    coords_to_flat_indices,
    load_source_voxel_normalization,
)

# Test VoxelNormalization dataclass
norm = VoxelNormalization(available=False)
assert norm.available is False
assert norm.scale is None
print("✓ VoxelNormalization dataclass works")

norm_with_data = VoxelNormalization(
    available=True,
    transforms_path="/tmp/transforms.json",
    scale=1.5,
    offset=np.array([0.1, 0.2, 0.3]),
)
assert norm_with_data.available is True
assert norm_with_data.scale == 1.5
print("✓ VoxelNormalization with data works")

# Test coords_to_flat_indices
import torch

coords_3d = torch.tensor([[0, 0, 0], [1, 2, 3], [63, 63, 63]], dtype=torch.int32)
flat = coords_to_flat_indices(coords_3d, resolution=64)
assert flat.shape == (3,)
assert flat[0] == 0
assert flat[2] == 63 * 64 * 64 + 63 * 64 + 63
print("✓ coords_to_flat_indices works")

coords_4d = torch.tensor([[0, 0, 0, 0], [0, 1, 2, 3]], dtype=torch.int32)
flat_4d = coords_to_flat_indices(coords_4d, resolution=64)
assert flat_4d.shape == (2,)
print("✓ coords_to_flat_indices (4D) works")

# Test path resolver
print("\n" + "=" * 60)
print("Testing Path Resolver")
print("=" * 60)

from editing.io import (
    SOURCE_RENDER_CANDIDATES,
    candidate_file,
    ensure_path_exists,
    resolve_asset_dir,
)

print(f"✓ SOURCE_RENDER_CANDIDATES: {SOURCE_RENDER_CANDIDATES}")

# Test candidate_file
test_file = Path(__file__)
assert candidate_file(test_file) == test_file
assert candidate_file(Path("/nonexistent")) is None
print("✓ candidate_file works")

# Test ensure_path_exists
try:
    ensure_path_exists(test_file, "test_file")
    print("✓ ensure_path_exists works for existing file")
except Exception as e:
    print(f"✗ ensure_path_exists failed: {e}")
    sys.exit(1)

try:
    ensure_path_exists(Path("/nonexistent"), "nonexistent")
    print("✗ ensure_path_exists should have raised RuntimeError")
    sys.exit(1)
except RuntimeError:
    print("✓ ensure_path_exists raises RuntimeError for missing file")

# Test resolve_asset_dir
cwd = Path.cwd()
asset_dir = resolve_asset_dir(str(cwd))
assert asset_dir == cwd
print(f"✓ resolve_asset_dir works: {asset_dir}")

print("\n" + "=" * 60)
print("All validation tests passed!")
print("=" * 60)
