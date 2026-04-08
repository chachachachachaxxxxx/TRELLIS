"""UniEdit utility functions for coordinate manipulation and statistics."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


def _coords3d(coords: torch.Tensor) -> torch.Tensor:
    """Extract 3D coordinates from tensor (handle both 3-col and 4-col formats)."""
    if coords.ndim != 2:
        raise RuntimeError(f"Expected 2D coords tensor, got shape {tuple(coords.shape)}")
    if coords.shape[1] == 3:
        return coords.int()
    if coords.shape[1] == 4:
        return coords[:, 1:].int()
    raise RuntimeError(f"Unsupported coordinate shape: {tuple(coords.shape)}")


def coords3d_to_batched(coords: torch.Tensor, batch_idx: int = 0) -> torch.Tensor:
    """Convert 3D coordinates to batched format (B, X, Y, Z)."""
    coords3 = _coords3d(coords)
    batch = torch.full(
        (coords3.shape[0], 1),
        fill_value=int(batch_idx),
        dtype=torch.int32,
        device=coords3.device,
    )
    return torch.cat([batch, coords3], dim=1)


def coords_to_flat_indices(coords: torch.Tensor, resolution: int = 64) -> torch.Tensor:
    """Convert 3D coordinates to flat indices for set operations."""
    coords3 = _coords3d(coords)
    return (
        coords3[:, 0].long() * (resolution * resolution)
        + coords3[:, 1].long() * resolution
        + coords3[:, 2].long()
    )


def build_stage2_selector(coords_target: torch.Tensor, coords_source: torch.Tensor) -> torch.Tensor:
    """Build selector tensor marking which target coords overlap with source.

    Args:
        coords_target: Target coordinates (N, 4) in batched format
        coords_source: Source coordinates (M, 4) in batched format

    Returns:
        Selector tensor (N, 1) with 1.0 for overlapping coords, 0.0 for new coords
    """
    target_codes = coords_to_flat_indices(_coords3d(coords_target))
    source_codes = coords_to_flat_indices(_coords3d(coords_source))
    overlap_mask = torch.isin(target_codes, source_codes)
    return overlap_mask.float().reshape(-1, 1)


def compose_stage1_coords(
    coords_source: torch.Tensor,
    coords_stage1_raw: torch.Tensor,
    mask_coords: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Compose Stage 1 coordinates by applying mask.

    Args:
        coords_source: Source voxel coordinates
        coords_stage1_raw: Raw Stage 1 denoised coordinates
        mask_coords: Optional mask coordinates (3D edit region)

    Returns:
        Tuple of:
        - coords_stage1_masked: Final Stage 1 coords (mask applied)
        - coords_preserve: Coordinates to preserve from source
        - stage1_meta: Statistics dictionary
    """
    coords_stage1_raw = _coords3d(coords_stage1_raw)
    raw_batched = coords3d_to_batched(coords_stage1_raw, batch_idx=0)

    if mask_coords is None:
        # No mask: use raw Stage 1 coords directly
        return raw_batched, coords_stage1_raw, {
            "mask_enabled": False,
            "stage1_raw_voxel_count": int(coords_stage1_raw.shape[0]),
            "stage1_masked_voxel_count": int(coords_stage1_raw.shape[0]),
        }

    # Apply mask: keep changes inside mask, restore source outside mask
    raw_codes = coords_to_flat_indices(coords_stage1_raw)
    mask_codes = coords_to_flat_indices(_coords3d(mask_coords))
    src_codes = coords_to_flat_indices(_coords3d(coords_source))

    # Coords inside mask from Stage 1
    raw_inside_mask = coords_stage1_raw[torch.isin(raw_codes, mask_codes)]

    # Source coords outside mask
    src_outside_mask = _coords3d(coords_source)[~torch.isin(src_codes, mask_codes)]

    # Union: Stage 1 inside mask + source outside mask
    coords_preserve = src_outside_mask
    coords_union = torch.cat([raw_inside_mask, src_outside_mask], dim=0)

    # Remove duplicates
    union_codes = coords_to_flat_indices(coords_union)
    unique_codes, unique_indices = torch.unique(union_codes, return_inverse=True)
    coords_masked = coords_union[torch.arange(coords_union.shape[0], device=coords_union.device)[
        torch.searchsorted(unique_indices, torch.arange(unique_codes.shape[0], device=coords_union.device))
    ]]

    coords_masked_batched = coords3d_to_batched(coords_masked, batch_idx=0)

    # Compute statistics
    masked_codes = coords_to_flat_indices(coords_masked)
    raw_overlap = int(torch.isin(raw_codes, src_codes).sum().item())
    masked_overlap = int(torch.isin(masked_codes, src_codes).sum().item())

    stage1_meta = {
        "mask_enabled": True,
        "mask_voxel_count": int(mask_coords.shape[0]),
        "stage1_raw_voxel_count": int(coords_stage1_raw.shape[0]),
        "stage1_masked_voxel_count": int(coords_masked.shape[0]),
        "stage1_raw_overlap_with_source": raw_overlap,
        "stage1_masked_overlap_with_source": masked_overlap,
        "stage1_raw_added_count": int((~torch.isin(raw_codes, src_codes)).sum().item()),
        "stage1_raw_removed_count": int((~torch.isin(src_codes, raw_codes)).sum().item()),
        "stage1_masked_added_count": int((~torch.isin(masked_codes, src_codes)).sum().item()),
        "stage1_masked_removed_count": int((~torch.isin(src_codes, masked_codes)).sum().item()),
        "stage1_raw_voxels_in_mask": int(torch.isin(raw_codes, mask_codes).sum().item()),
        "stage1_masked_voxels_in_mask": int(torch.isin(masked_codes, mask_codes).sum().item()),
        "preserve_voxel_count": int(coords_preserve.shape[0]),
    }

    return coords_masked_batched, coords_preserve, stage1_meta


def build_sparse_replace_index_map(
    coords_target: torch.Tensor,
    coords_source: torch.Tensor,
    selector: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build index mapping for sparse latent replacement.

    Args:
        coords_target: Target coordinates (N, 4)
        coords_source: Source coordinates (M, 4)
        selector: Selector tensor (N, 1) marking overlap

    Returns:
        Tuple of:
        - replace_target_indices: Indices in target to replace
        - replace_source_indices: Corresponding indices in source
    """
    target_codes = coords_to_flat_indices(_coords3d(coords_target))
    source_codes = coords_to_flat_indices(_coords3d(coords_source))

    # Find overlapping coords marked by selector
    overlap_mask = (selector.reshape(-1) > 0.5)
    overlap_target_codes = target_codes[overlap_mask]

    # Find matching indices in source
    replace_target_indices = []
    replace_source_indices = []

    for i, code in enumerate(overlap_target_codes):
        target_idx = torch.where(overlap_mask)[0][i]
        source_matches = torch.where(source_codes == code)[0]
        if source_matches.numel() > 0:
            replace_target_indices.append(target_idx)
            replace_source_indices.append(source_matches[0])

    if len(replace_target_indices) == 0:
        return torch.tensor([], dtype=torch.long, device=coords_target.device), \
               torch.tensor([], dtype=torch.long, device=coords_source.device)

    return torch.tensor(replace_target_indices, dtype=torch.long, device=coords_target.device), \
           torch.tensor(replace_source_indices, dtype=torch.long, device=coords_source.device)
