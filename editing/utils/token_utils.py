from __future__ import annotations

from typing import List, Tuple

import numpy as np
from PIL import Image


def resolve_patch_size(value) -> int:
    """Resolve patch size from various formats.

    Args:
        value: Patch size as int or tuple

    Returns:
        Patch size as integer
    """
    if isinstance(value, tuple):
        if len(value) != 2 or value[0] != value[1]:
            raise RuntimeError(f"Unsupported patch size: {value}")
        return int(value[0])
    return int(value)


def mask_to_patch_selection(
    mask: Image.Image,
    patch_size: int,
    coverage_threshold: float,
) -> Tuple[np.ndarray, List[int], np.ndarray]:
    """Convert mask to patch-level selection.

    Args:
        mask: Binary mask image
        patch_size: Size of each patch
        coverage_threshold: Minimum coverage to consider patch as edited

    Returns:
        Tuple of (edited_grid, edited_linear_indices, coverage_grid)
    """
    mask_np = np.asarray(mask.convert("L")) > 0
    if mask_np.shape[0] % patch_size != 0 or mask_np.shape[1] % patch_size != 0:
        raise RuntimeError(
            f"Mask size {mask_np.shape[::-1]} is not divisible by patch size {patch_size}."
        )

    grid_h = mask_np.shape[0] // patch_size
    grid_w = mask_np.shape[1] // patch_size
    edited_grid = np.zeros((grid_h, grid_w), dtype=bool)
    coverage_grid = np.zeros((grid_h, grid_w), dtype=np.float32)
    edited_linear_indices: List[int] = []

    for row in range(grid_h):
        for col in range(grid_w):
            y0 = row * patch_size
            y1 = (row + 1) * patch_size
            x0 = col * patch_size
            x1 = (col + 1) * patch_size
            patch = mask_np[y0:y1, x0:x1]
            coverage = float(patch.mean())
            coverage_grid[row, col] = coverage
            if coverage_threshold <= 0.0:
                is_edited = bool(patch.any())
            else:
                is_edited = bool(coverage >= coverage_threshold)
            if is_edited:
                edited_grid[row, col] = True
                edited_linear_indices.append(row * grid_w + col)

    return edited_grid, edited_linear_indices, coverage_grid


def build_image_token_metadata(
    cond: "torch.Tensor",
    mask: Image.Image,
    patch_size: int,
    patch_coverage_threshold: float,
) -> dict:
    """Build token metadata from condition and mask.

    Args:
        cond: Condition tensor with shape (batch, tokens, channels)
        mask: Binary mask image
        patch_size: Size of each patch
        patch_coverage_threshold: Minimum coverage to consider patch as edited

    Returns:
        Dictionary with token metadata
    """
    total_tokens = int(cond.shape[1])
    edited_patch_grid, edited_patch_linear_indices, coverage_grid = mask_to_patch_selection(
        mask=mask,
        patch_size=patch_size,
        coverage_threshold=patch_coverage_threshold,
    )
    grid_h, grid_w = edited_patch_grid.shape
    patch_token_count = grid_h * grid_w
    prefix_token_count = total_tokens - patch_token_count
    if prefix_token_count < 0:
        raise RuntimeError(
            f"Unexpected token layout: total_tokens={total_tokens}, patch_token_count={patch_token_count}"
        )

    patch_labels = [f"patch_r{row:02d}_c{col:02d}" for row in range(grid_h) for col in range(grid_w)]
    token_labels = ["[CLS]"] + [f"[REG{idx}]" for idx in range(max(prefix_token_count - 1, 0))] + patch_labels
    if len(token_labels) != total_tokens:
        token_labels = [f"token_{idx:04d}" for idx in range(total_tokens)]

    edited_patch_set = set(edited_patch_linear_indices)
    keep_patch_linear_indices = [idx for idx in range(patch_token_count) if idx not in edited_patch_set]
    edited_token_indices = [prefix_token_count + idx for idx in edited_patch_linear_indices]
    keep_token_indices = [prefix_token_count + idx for idx in keep_patch_linear_indices]

    selection_rule = (
        "Any patch with any mask overlap is counted as an edited visual token."
        if patch_coverage_threshold <= 0.0
        else f"Any patch with mask coverage >= {patch_coverage_threshold:.4f} is counted as an edited visual token."
    )

    return {
        "proc_size": [int(mask.width), int(mask.height)],
        "patch_grid_size": [int(grid_h), int(grid_w)],
        "patch_size": int(patch_size),
        "total_tokens": total_tokens,
        "prefix_token_count": int(prefix_token_count),
        "patch_token_count": int(patch_token_count),
        "special_token_indices": list(range(prefix_token_count)),
        "source_keep_indices": keep_token_indices,
        "edit_keep_indices": keep_token_indices,
        "keep_patch_linear_indices": keep_patch_linear_indices,
        "edited_patch_linear_indices": edited_patch_linear_indices,
        "edited_patch_grid": edited_patch_grid.astype(np.uint8).tolist(),
        "edited_patch_coverage_grid": np.round(coverage_grid, 6).tolist(),
        "edited_token_indices": edited_token_indices,
        "keep_token_labels": [token_labels[idx] for idx in keep_token_indices],
        "edited_token_labels": [token_labels[idx] for idx in edited_token_indices],
        "token_labels": token_labels,
        "selection_rule": selection_rule,
        "patch_coverage_threshold": float(patch_coverage_threshold),
    }
