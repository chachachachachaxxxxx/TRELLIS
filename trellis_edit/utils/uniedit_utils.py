"""UniEdit utility functions for coordinate manipulation and statistics."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _coords3d(coords: torch.Tensor) -> torch.Tensor:
    """Extract 3D coordinates from tensor (handle both 3-col and 4-col formats)."""
    if coords.ndim != 2:
        raise RuntimeError(f"Expected 2D coords tensor, got shape {tuple(coords.shape)}")
    if coords.shape[1] == 3:
        return coords.int()
    if coords.shape[1] == 4:
        return coords[:, 1:].int()
    raise RuntimeError(f"Unsupported coordinate shape: {tuple(coords.shape)}")


def _infer_sparse_resolution(*coords_sets: Optional[torch.Tensor]) -> int:
    max_coord = 0
    found = False
    for coords in coords_sets:
        if coords is None or coords.numel() == 0:
            continue
        coords3 = _coords3d(coords)
        if coords3.numel() == 0:
            continue
        max_coord = max(max_coord, int(coords3.max().item()))
        found = True
    return max_coord + 1 if found else 64


def _dense_voxel_mask(coords: torch.Tensor, resolution: int) -> torch.Tensor:
    mask = torch.zeros((resolution, resolution, resolution), dtype=torch.bool, device=coords.device)
    if coords.numel() == 0:
        return mask
    coords3 = _coords3d(coords).long()
    mask[coords3[:, 0], coords3[:, 1], coords3[:, 2]] = True
    return mask


def _dense_mask_to_coords(mask: torch.Tensor) -> torch.Tensor:
    if mask.numel() == 0 or not mask.any():
        return torch.zeros((0, 3), dtype=torch.int32, device=mask.device)
    return torch.nonzero(mask, as_tuple=False).to(dtype=torch.int32)


def _cross_kernel(device: torch.device) -> torch.Tensor:
    kernel = torch.zeros((1, 1, 3, 3, 3), dtype=torch.float32, device=device)
    kernel[0, 0, 1, 1, 1] = 1.0
    kernel[0, 0, 0, 1, 1] = 1.0
    kernel[0, 0, 2, 1, 1] = 1.0
    kernel[0, 0, 1, 0, 1] = 1.0
    kernel[0, 0, 1, 2, 1] = 1.0
    kernel[0, 0, 1, 1, 0] = 1.0
    kernel[0, 0, 1, 1, 2] = 1.0
    return kernel


def _six_neighbor_count(mask: torch.Tensor) -> torch.Tensor:
    kernel = _cross_kernel(mask.device)
    kernel[0, 0, 1, 1, 1] = 0.0
    counts = F.conv3d(mask.float().unsqueeze(0).unsqueeze(0), kernel, padding=1)
    return counts.squeeze(0).squeeze(0).to(dtype=torch.int32)


def _erode_dense_mask(mask: torch.Tensor, steps: int) -> torch.Tensor:
    if steps <= 0 or mask.numel() == 0 or not mask.any():
        return mask.clone()
    kernel = _cross_kernel(mask.device)
    eroded = mask.clone()
    for _ in range(steps):
        if not eroded.any():
            break
        counts = F.conv3d(eroded.float().unsqueeze(0).unsqueeze(0), kernel, padding=1)
        eroded = counts.squeeze(0).squeeze(0) == 7.0
    return eroded


def _select_connected_band_targets(
    raw_band_mask: torch.Tensor,
    *,
    support_mask: torch.Tensor,
    neighbor_threshold: int,
) -> torch.Tensor:
    if raw_band_mask.numel() == 0 or not raw_band_mask.any():
        return torch.zeros_like(raw_band_mask)

    accepted = torch.zeros_like(raw_band_mask)
    threshold = int(neighbor_threshold)
    while True:
        support = support_mask | accepted
        support_neighbors = _six_neighbor_count(support)
        new_accept = raw_band_mask & ~accepted & (support_neighbors >= threshold)
        if not new_accept.any():
            break
        accepted |= new_accept
    return accepted


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


def compute_preserve_coords_from_mask(
    coords_source: torch.Tensor,
    mask_coords: torch.Tensor,
) -> torch.Tensor:
    """Return source voxels outside the edit mask."""
    source_coords3 = _coords3d(coords_source)
    mask_coords3 = _coords3d(mask_coords)
    resolution = _infer_sparse_resolution(source_coords3, mask_coords3)
    src_codes = coords_to_flat_indices(source_coords3, resolution=resolution)
    mask_codes = coords_to_flat_indices(mask_coords3, resolution=resolution)
    return source_coords3[~torch.isin(src_codes, mask_codes)]


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
    resolution = _infer_sparse_resolution(coords_source, coords_stage1_raw, mask_coords)
    raw_codes = coords_to_flat_indices(coords_stage1_raw, resolution=resolution)
    mask_codes = coords_to_flat_indices(_coords3d(mask_coords), resolution=resolution)
    src_codes = coords_to_flat_indices(_coords3d(coords_source), resolution=resolution)

    # Coords inside mask from Stage 1
    raw_inside_mask = coords_stage1_raw[torch.isin(raw_codes, mask_codes)]

    # Source coords outside mask
    src_outside_mask = compute_preserve_coords_from_mask(coords_source, mask_coords)

    # Union: Stage 1 inside mask + source outside mask
    coords_preserve = src_outside_mask
    coords_union = torch.cat([raw_inside_mask, src_outside_mask], dim=0)

    # Remove duplicates. Order is not semantically important here; we only need
    # a valid unique voxel set for later masking and stage-2 selector building.
    coords_masked = torch.unique(coords_union, dim=0)

    coords_masked_batched = coords3d_to_batched(coords_masked, batch_idx=0)

    # Compute statistics
    masked_codes = coords_to_flat_indices(coords_masked, resolution=resolution)
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


def compose_stage1_coords_restore_all_outside_mask(
    coords_source: torch.Tensor,
    coords_stage1_raw: torch.Tensor,
    mask_coords: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Compose Stage 1 coords by restoring all outside-mask voxels.

    Keep the full raw Stage 1 prediction, then union it with source voxels outside
    the edit mask. This preserves all outside-mask voxels from both source and raw.
    """
    coords_stage1_raw = _coords3d(coords_stage1_raw)
    source_coords3 = _coords3d(coords_source)
    mask_coords3 = _coords3d(mask_coords)
    resolution = _infer_sparse_resolution(source_coords3, coords_stage1_raw, mask_coords3)

    raw_codes = coords_to_flat_indices(coords_stage1_raw, resolution=resolution)
    src_codes = coords_to_flat_indices(source_coords3, resolution=resolution)
    mask_codes = coords_to_flat_indices(mask_coords3, resolution=resolution)

    coords_preserve = compute_preserve_coords_from_mask(source_coords3, mask_coords3)
    coords_union = torch.cat([coords_stage1_raw, coords_preserve], dim=0)
    coords_masked = torch.unique(coords_union, dim=0)
    coords_masked_batched = coords3d_to_batched(coords_masked, batch_idx=0)

    masked_codes = coords_to_flat_indices(coords_masked, resolution=resolution)
    stage1_meta = {
        "mask_enabled": True,
        "mask_voxel_count": int(mask_coords3.shape[0]),
        "stage1_compose_mode": "restore_all_outside_mask",
        "stage1_raw_voxel_count": int(coords_stage1_raw.shape[0]),
        "stage1_masked_voxel_count": int(coords_masked.shape[0]),
        "stage1_raw_overlap_with_source": int(torch.isin(raw_codes, src_codes).sum().item()),
        "stage1_masked_overlap_with_source": int(torch.isin(masked_codes, src_codes).sum().item()),
        "stage1_raw_added_count": int((~torch.isin(raw_codes, src_codes)).sum().item()),
        "stage1_raw_removed_count": int((~torch.isin(src_codes, raw_codes)).sum().item()),
        "stage1_masked_added_count": int((~torch.isin(masked_codes, src_codes)).sum().item()),
        "stage1_masked_removed_count": int((~torch.isin(src_codes, masked_codes)).sum().item()),
        "stage1_raw_voxels_in_mask": int(torch.isin(raw_codes, mask_codes).sum().item()),
        "stage1_masked_voxels_in_mask": int(torch.isin(masked_codes, mask_codes).sum().item()),
        "preserve_voxel_count": int(coords_preserve.shape[0]),
    }
    return coords_masked_batched, coords_preserve, stage1_meta


def compose_stage1_coords_boundary_band_restore(
    coords_source: torch.Tensor,
    coords_stage1_raw: torch.Tensor,
    mask_coords: torch.Tensor,
    *,
    band_width_voxels: int,
    band_target_neighbor_threshold: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Compose Stage 1 coordinates with a source-preserving boundary band.

    The final sparse structure is formed from:
    - raw Stage 1 voxels in the eroded mask core
    - source voxels outside the edit mask
    - source voxels inside a narrow boundary band
    - only band target voxels that stay connected to the source shell or raw core
    """
    coords_stage1_raw = _coords3d(coords_stage1_raw)
    source_coords3 = _coords3d(coords_source)
    mask_coords3 = _coords3d(mask_coords)
    resolution = _infer_sparse_resolution(source_coords3, coords_stage1_raw, mask_coords3)

    source_mask = _dense_voxel_mask(source_coords3, resolution)
    raw_mask = _dense_voxel_mask(coords_stage1_raw, resolution)
    edit_mask = _dense_voxel_mask(mask_coords3, resolution)

    core_mask = _erode_dense_mask(edit_mask, steps=int(band_width_voxels))
    band_mask = edit_mask & ~core_mask

    raw_core_mask = raw_mask & core_mask
    raw_band_mask = raw_mask & band_mask
    source_band_mask = source_mask & band_mask
    source_outside_mask = source_mask & ~edit_mask

    kept_raw_band_mask = _select_connected_band_targets(
        raw_band_mask,
        support_mask=source_mask | raw_core_mask,
        neighbor_threshold=band_target_neighbor_threshold,
    )

    final_mask = source_outside_mask | source_band_mask | raw_core_mask | kept_raw_band_mask
    preserve_mask = source_outside_mask | source_band_mask

    coords_masked = _dense_mask_to_coords(final_mask)
    coords_preserve = _dense_mask_to_coords(preserve_mask)
    coords_masked_batched = coords3d_to_batched(coords_masked, batch_idx=0)

    raw_codes = coords_to_flat_indices(coords_stage1_raw, resolution=resolution)
    src_codes = coords_to_flat_indices(source_coords3, resolution=resolution)
    mask_codes = coords_to_flat_indices(mask_coords3, resolution=resolution)
    masked_codes = coords_to_flat_indices(coords_masked, resolution=resolution)

    stage1_meta = {
        "mask_enabled": True,
        "mask_voxel_count": int(mask_coords3.shape[0]),
        "stage1_raw_voxel_count": int(coords_stage1_raw.shape[0]),
        "stage1_masked_voxel_count": int(coords_masked.shape[0]),
        "stage1_raw_overlap_with_source": int(torch.isin(raw_codes, src_codes).sum().item()),
        "stage1_masked_overlap_with_source": int(torch.isin(masked_codes, src_codes).sum().item()),
        "stage1_raw_added_count": int((~torch.isin(raw_codes, src_codes)).sum().item()),
        "stage1_raw_removed_count": int((~torch.isin(src_codes, raw_codes)).sum().item()),
        "stage1_masked_added_count": int((~torch.isin(masked_codes, src_codes)).sum().item()),
        "stage1_masked_removed_count": int((~torch.isin(src_codes, masked_codes)).sum().item()),
        "stage1_raw_voxels_in_mask": int(torch.isin(raw_codes, mask_codes).sum().item()),
        "stage1_masked_voxels_in_mask": int(torch.isin(masked_codes, mask_codes).sum().item()),
        "preserve_voxel_count": int(coords_preserve.shape[0]),
        "boundary_band_width_voxels": int(band_width_voxels),
        "boundary_band_target_neighbor_threshold": int(band_target_neighbor_threshold),
        "boundary_band_core_voxel_count": int(core_mask.sum().item()),
        "boundary_band_voxel_count": int(band_mask.sum().item()),
        "boundary_band_source_voxel_count": int(source_band_mask.sum().item()),
        "boundary_band_raw_candidate_count": int(raw_band_mask.sum().item()),
        "boundary_band_raw_kept_count": int(kept_raw_band_mask.sum().item()),
    }

    return coords_masked_batched, coords_preserve, stage1_meta
