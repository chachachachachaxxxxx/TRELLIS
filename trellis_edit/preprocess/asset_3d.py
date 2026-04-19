from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import trimesh


@dataclass(frozen=True)
class VoxelNormalization:
    """Source voxel space normalization metadata from transforms.json."""
    available: bool
    transforms_path: Optional[str] = None
    scale: Optional[float] = None
    offset: Optional[np.ndarray] = None


@dataclass(frozen=True)
class MaskGLBResult:
    """Result of loading and processing a 3D mask GLB."""
    coords: Optional[torch.Tensor]
    mesh: Optional[trimesh.Trimesh]
    meta: dict


def load_ply_positions(ply_path: Path) -> np.ndarray:
    import utils3d  # type: ignore
    position = utils3d.io.read_ply(str(ply_path))[0]
    return np.asarray(position, dtype=np.float32)


def ply_to_coords(ply_path: Path, device: torch.device, resolution: int = 64) -> torch.Tensor:
    """Convert PLY vertex positions to voxel grid coordinates.

    Maps positions from [-0.5, 0.5] to [0, resolution-1] integer coordinates.

    Args:
        ply_path: Path to PLY file
        device: Target torch device
        resolution: Voxel grid resolution (default 64)

    Returns:
        Nx3 tensor of unique integer coordinates
    """
    position = load_ply_positions(ply_path)
    coords = ((torch.from_numpy(position) + 0.5) * resolution).int()
    coords = torch.clamp(coords, min=0, max=resolution - 1)
    coords = torch.unique(coords, dim=0).contiguous()
    return coords.to(device=device)


def coords_to_voxel(coords: torch.Tensor, device: torch.device, resolution: int = 64) -> torch.Tensor:
    """Convert sparse coordinates to dense voxel grid.

    Args:
        coords: Nx3 tensor of integer coordinates
        device: Target torch device
        resolution: Voxel grid resolution (default 64)

    Returns:
        Dense voxel tensor of shape (1, 1, resolution, resolution, resolution)
    """
    voxel = torch.zeros(1, 1, resolution, resolution, resolution, dtype=torch.float32, device=device)
    voxel[:, 0, coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
    return voxel


def feats_to_slat(pipeline, feats_path: Path, SparseTensor):
    """Load SLAT features from features.npz and encode them.

    Supports two formats:
    1. New format (patch tokens): 'patchtokens' + 'indices' - needs encoding
    2. Old format (encoded SLAT): 'feats' + 'coords' - already encoded

    Args:
        pipeline: TRELLIS pipeline with slat_encoder model
        feats_path: Path to features.npz file
        SparseTensor: SparseTensor class from trellis

    Returns:
        Encoded SLAT features (SparseTensor)
    """
    feats = np.load(feats_path)

    # Check format and load accordingly
    if "patchtokens" in feats and "indices" in feats:
        # New format: patch tokens that need encoding
        sparse_tensor = SparseTensor(
            feats=torch.from_numpy(feats["patchtokens"]).float().to(pipeline.device),
            coords=torch.cat(
                [
                    torch.zeros(feats["patchtokens"].shape[0], 1, dtype=torch.int32),
                    torch.from_numpy(feats["indices"]).int(),
                ],
                dim=1,
            ).to(pipeline.device),
        )
        feats_encoder = pipeline.models["slat_encoder"]
        return feats_encoder(sparse_tensor, sample_posterior=False)

    elif "feats" in feats and "coords" in feats:
        # Old format: already encoded SLAT features
        # coords shape: (N, 4) with batch index in first column
        # feats shape: (N, 8) - encoded SLAT features
        slat_tensor = SparseTensor(
            feats=torch.from_numpy(feats["feats"]).float().to(pipeline.device),
            coords=torch.from_numpy(feats["coords"]).int().to(pipeline.device),
        )
        return slat_tensor

    else:
        slat_tensor = SparseTensor(
            feats=torch.from_numpy(feats["feats"]).float().to(pipeline.device),
            coords=torch.from_numpy(feats["coords"]).int().to(pipeline.device),
        )
        return slat_tensor


def load_source_voxel_normalization(asset_dir: Path) -> VoxelNormalization:
    """Load source voxel space normalization from transforms.json.

    Args:
        asset_dir: Directory containing transforms.json

    Returns:
        VoxelNormalization with scale and offset if available
    """
    transforms_path = asset_dir / "transforms.json"
    if not transforms_path.is_file():
        return VoxelNormalization(available=False)

    payload = json.loads(transforms_path.read_text())
    scale = float(payload["scale"])
    offset_raw = payload["offset"]
    offset = np.asarray(offset_raw, dtype=np.float32)

    return VoxelNormalization(
        available=True,
        transforms_path=str(transforms_path),
        scale=scale,
        offset=offset,
    )




def glb_to_ply(input_glb_path: str, output_ply_path: str) -> None:
    """Convert GLB to PLY using Blender Python API.

    This ensures coordinate system consistency with SLAT encoder voxels.

    Args:
        input_glb_path: Path to input GLB file
        output_ply_path: Path to output PLY file
    """
    import bpy

    # Clear scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.context.scene.render.engine = "CYCLES"

    # Import GLB
    bpy.ops.import_scene.gltf(filepath=input_glb_path)

    # Export PLY with normals (ASCII format for compatibility)
    bpy.ops.wm.ply_export(
        filepath=output_ply_path,
        export_normals=True,
        ascii_format=True
    )


def load_mask_glb_coords(
    mask_glb: str,
    device: torch.device,
    resolution: int = 64,
    asset_dir: Optional[Path] = None,
    source_normalization: Optional[VoxelNormalization] = None,
    ensure_path_exists_fn=None,
) -> MaskGLBResult:
    """Load mask voxel coordinates from GLB or pre-generated PLY.

    Strategy 1: Read pre-generated voxels_delete.ply from asset_dir
    Strategy 2: Load mask GLB and generate voxels via VoxHammer filtering
                (same approach as process_delete_ply in temp/delete_region_voxel.py)

    Args:
        mask_glb: Path to mask GLB file
        device: Target torch device
        resolution: Voxel grid resolution (default 64)
        asset_dir: Optional directory containing voxels_delete.ply
        source_normalization: Optional normalization to apply
        ensure_path_exists_fn: Function to validate path exists

    Returns:
        MaskGLBResult with coords, mesh, and metadata
    """
    # Auto-load source normalization from asset_dir if not provided
    if source_normalization is None and asset_dir is not None:
        source_normalization = load_source_voxel_normalization(asset_dir)

    # Strategy 1: read existing voxels_delete.ply
    if asset_dir is not None:
        voxels_delete_path = asset_dir / "voxels_delete.ply"
        if voxels_delete_path.is_file():
            coords = ply_to_coords(voxels_delete_path, device, resolution)
            meta = {
                "mask_source": "voxels_delete_ply",
                "voxels_delete_path": str(voxels_delete_path),
                "voxel_count": int(coords.shape[0]),
                "enabled": True,
            }
            return MaskGLBResult(coords=coords, mesh=None, meta=meta)

    # Strategy 2: generate via VoxHammer filtering (same as process_delete_ply)
    # Validate mask_glb path
    mask_glb = mask_glb.strip()
    if not mask_glb:
        return MaskGLBResult(coords=None, mesh=None, meta={"enabled": False})

    if ensure_path_exists_fn is not None:
        mask_path = ensure_path_exists_fn(Path(mask_glb).expanduser().resolve(), "mask_glb")
    else:
        mask_path = Path(mask_glb).expanduser().resolve()

    from trellis_edit.preprocess.voxel_filtering import process_voxels_with_improved_filtering

    # Use preset grid (same as process_delete_ply)
    preset_voxel_path = "assets/preset/preset_grid64.ply"
    voxel_size = 1.0 / float(resolution)

    # Determine output paths
    if asset_dir is not None:
        # Save in asset_dir (persistent)
        mesh_delete_path = asset_dir / "mesh_delete.ply"
        voxels_delete_path = asset_dir / "voxels_delete.ply"
    else:
        # Use temporary files
        mesh_delete_path = Path(tempfile.mktemp(suffix="_mesh_delete.ply"))
        voxels_delete_path = Path(tempfile.mktemp(suffix="_voxels_delete.ply"))

    # Convert GLB to PLY using bpy (same coordinate system as SLAT encoder)
    glb_to_ply(str(mask_path), str(mesh_delete_path))

    # Apply VoxHammer filtering (same as process_delete_ply)
    process_voxels_with_improved_filtering(
        preset_voxel_path,
        str(mesh_delete_path),
        str(voxels_delete_path),
        method="volume",
        voxel_size=voxel_size,
        inside=True,
    )

    # Load generated voxels
    coords = ply_to_coords(voxels_delete_path, device, resolution)

    meta = {
        "mask_glb": str(mask_path),
        "enabled": True,
        "mask_source": "voxhammer_process_voxels_with_improved_filtering",
        "voxel_count": int(coords.shape[0]),
        "mesh_delete_path": str(mesh_delete_path),
        "voxels_delete_path": str(voxels_delete_path),
    }
    return MaskGLBResult(coords=coords, mesh=None, meta=meta)


def coords_to_flat_indices(coords: torch.Tensor, resolution: int = 64) -> torch.Tensor:
    coords = coords.long()
    if coords.shape[1] == 3:
        return coords[:, 0] * resolution * resolution + coords[:, 1] * resolution + coords[:, 2]
    return (
        coords[:, 0] * resolution * resolution * resolution
        + coords[:, 1] * resolution * resolution
        + coords[:, 2] * resolution
        + coords[:, 3]
    )


def sparse_batch_slice(sparse_tensor, batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract coordinates and features for a specific batch index.

    Args:
        sparse_tensor: SparseTensor with layout attribute
        batch_idx: Batch index to extract

    Returns:
        Tuple of (coords, feats) for the batch
    """
    slc = sparse_tensor.layout[batch_idx]
    return sparse_tensor.coords[slc], sparse_tensor.feats[slc]


def project_sparse_terminal_noise(
    source_noise,
    target_coords: torch.Tensor,
    device: torch.device,
    SparseTensor,
    resolution: int = 64,
    preserve_coords: Optional[torch.Tensor] = None,
):
    if target_coords.ndim != 2 or target_coords.shape[1] not in (3, 4):
        raise ValueError(f"Expected target_coords shape [N, 3] or [N, 4], got {tuple(target_coords.shape)}")
    if target_coords.shape[1] == 3:
        target_coords = torch.cat(
            [
                torch.zeros((target_coords.shape[0], 1), dtype=torch.int32, device=target_coords.device),
                target_coords.int(),
            ],
            dim=1,
        )
    else:
        target_coords = target_coords.int()

    if preserve_coords is not None:
        if preserve_coords.ndim != 2 or preserve_coords.shape[1] not in (3, 4):
            raise ValueError(
                f"Expected preserve_coords shape [N, 3] or [N, 4], got {tuple(preserve_coords.shape)}"
            )
        if preserve_coords.shape[1] == 3:
            preserve_coords = torch.cat(
                [
                    torch.zeros((preserve_coords.shape[0], 1), dtype=torch.int32, device=preserve_coords.device),
                    preserve_coords.int(),
                ],
                dim=1,
            )
        else:
            preserve_coords = preserve_coords.int()

    batch_size = int(target_coords[:, 0].max().item()) + 1 if target_coords.numel() > 0 else 1
    feature_dim = source_noise.feats.shape[1]
    all_coords = []
    all_feats = []
    src_batch_count = source_noise.shape[0]

    for batch_idx in range(batch_size):
        tgt_mask = target_coords[:, 0] == batch_idx
        tgt_coords_batch = target_coords[tgt_mask]
        if tgt_coords_batch.shape[0] == 0:
            continue

        src_coords_full, src_feats_full = sparse_batch_slice(
            source_noise, batch_idx if src_batch_count > 1 else 0
        )
        src_coords = src_coords_full[:, 1:]
        tgt_coords = tgt_coords_batch[:, 1:]

        src_codes = coords_to_flat_indices(src_coords, resolution)
        tgt_codes = coords_to_flat_indices(tgt_coords, resolution)
        preserve_target = torch.ones_like(tgt_codes, dtype=torch.bool, device=tgt_codes.device)
        if preserve_coords is not None:
            preserve_batch = preserve_coords[preserve_coords[:, 0] == batch_idx][:, 1:]
            if preserve_batch.shape[0] == 0:
                preserve_target = torch.zeros_like(tgt_codes, dtype=torch.bool, device=tgt_codes.device)
            else:
                preserve_codes = coords_to_flat_indices(preserve_batch, resolution)
                preserve_target = torch.isin(tgt_codes, preserve_codes)
        randn_feats = torch.randn(
            tgt_coords.shape[0],
            feature_dim,
            device=device,
            dtype=source_noise.feats.dtype,
        )

        if src_codes.numel() > 0:
            src_codes_sorted, order = torch.sort(src_codes)
            insert_pos = torch.searchsorted(src_codes_sorted, tgt_codes)
            valid = insert_pos < src_codes_sorted.shape[0]
            matched = valid.clone()
            matched[valid] = src_codes_sorted[insert_pos[valid]] == tgt_codes[valid]
            matched &= preserve_target
            if matched.any():
                matched_order = order[insert_pos[matched]]
                randn_feats[matched] = src_feats_full[matched_order]

        all_coords.append(tgt_coords_batch)
        all_feats.append(randn_feats)

    coords = torch.cat(all_coords, dim=0).to(device=device)
    feats = torch.cat(all_feats, dim=0).to(device=device)
    return SparseTensor(feats=feats, coords=coords)
