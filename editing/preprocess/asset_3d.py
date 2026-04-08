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
    """Load vertex positions from a PLY file.

    Tries utils3d first, falls back to trimesh.

    Args:
        ply_path: Path to PLY file

    Returns:
        Nx3 array of vertex positions
    """
    try:
        import utils3d  # type: ignore

        position = utils3d.io.read_ply(str(ply_path))[0]
        return np.asarray(position, dtype=np.float32)
    except Exception:
        mesh = trimesh.load(str(ply_path), process=False)
        if hasattr(mesh, "vertices"):
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
        elif isinstance(mesh, trimesh.points.PointCloud):
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
        else:
            raise RuntimeError(f"Unsupported PLY payload in {ply_path}")
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise RuntimeError(f"Expected Nx3 vertices in {ply_path}, got shape {vertices.shape}")
        return vertices


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

    Args:
        pipeline: TRELLIS pipeline with slat_encoder model
        feats_path: Path to features.npz file
        SparseTensor: SparseTensor class from trellis

    Returns:
        Encoded SLAT features
    """
    feats = np.load(feats_path)
    if "patchtokens" not in feats or "indices" not in feats:
        raise RuntimeError(
            f"features.npz must contain 'patchtokens' and 'indices', got keys: {list(feats.keys())}"
        )
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
    if not isinstance(offset_raw, Sequence) or len(offset_raw) != 3:
        raise RuntimeError(f"Invalid offset field in {transforms_path}: {offset_raw!r}")
    offset = np.asarray(offset_raw, dtype=np.float32)

    return VoxelNormalization(
        available=True,
        transforms_path=str(transforms_path),
        scale=scale,
        offset=offset,
    )


def _load_mask_mesh(
    mask_glb: str,
    source_normalization: Optional[VoxelNormalization] = None,
    ensure_path_exists_fn=None,
) -> Tuple[Optional[trimesh.Trimesh], dict]:
    """Load and validate a 3D mask mesh from GLB.

    Args:
        mask_glb: Path to mask GLB file
        source_normalization: Optional normalization to apply
        ensure_path_exists_fn: Function to validate path exists

    Returns:
        Tuple of (mesh, metadata dict)
    """
    mask_glb = mask_glb.strip()
    if not mask_glb:
        return None, {"mask_glb": None, "enabled": False}

    if ensure_path_exists_fn is not None:
        mask_path = ensure_path_exists_fn(Path(mask_glb).expanduser().resolve(), "mask_glb")
    else:
        mask_path = Path(mask_glb).expanduser().resolve()
        if not mask_path.exists():
            raise RuntimeError(f"mask_glb does not exist: {mask_path}")

    loaded = trimesh.load(str(mask_path), process=False)
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.dump(concatenate=True)
        if not isinstance(mesh, trimesh.Trimesh) or mesh.vertices.shape[0] == 0:
            raise RuntimeError(f"mask_glb contains no usable triangle mesh after scene concatenation: {mask_path}")
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded.copy()
    else:
        raise RuntimeError(f"Unsupported mask_glb payload: {type(loaded)}")

    raw_vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if raw_vertices.ndim != 2 or raw_vertices.shape[1] != 3 or raw_vertices.shape[0] == 0:
        raise RuntimeError(f"mask_glb must contain valid Nx3 vertices, got shape {raw_vertices.shape}")

    raw_min = raw_vertices.min(axis=0)
    raw_max = raw_vertices.max(axis=0)
    transformed = False
    source_scale = None
    source_offset = None
    source_transforms_path = None

    if source_normalization is not None and source_normalization.available:
        source_scale = float(source_normalization.scale)
        source_offset = np.asarray(source_normalization.offset, dtype=np.float32)
        source_transforms_path = str(source_normalization.transforms_path)
        mesh.vertices = raw_vertices * source_scale + source_offset[None, :]
        transformed = True

    aligned_vertices = np.asarray(mesh.vertices, dtype=np.float32)
    aligned_min = aligned_vertices.min(axis=0)
    aligned_max = aligned_vertices.max(axis=0)
    if aligned_min.min() < -0.55 or aligned_max.max() > 0.55:
        normalization_hint = ""
        if source_transforms_path is not None:
            normalization_hint = f" Applied source normalization from {source_transforms_path},"
        else:
            normalization_hint = " No source voxel-space normalization metadata was found (expected transforms.json beside the render assets),"
        raise RuntimeError(
            "mask_glb could not be aligned to TRELLIS voxel space."
            f"{normalization_hint} got bounds min={aligned_min.tolist()} max={aligned_max.tolist()} for {mask_path}."
        )

    meta = {
        "mask_glb": str(mask_path),
        "enabled": True,
        "input_coordinate_mode": "source_render_space_to_voxel_space" if transformed else "preserve_input_coordinates",
        "raw_aabb_min": raw_min.tolist(),
        "raw_aabb_max": raw_max.tolist(),
        "voxel_space_aabb_min": aligned_min.tolist(),
        "voxel_space_aabb_max": aligned_max.tolist(),
        "source_transforms_path": source_transforms_path,
        "source_scale": source_scale,
        "source_offset": source_offset.tolist() if source_offset is not None else None,
    }
    return mesh, meta


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
    # Strategy 1: read existing voxels_delete.ply
    if asset_dir is not None:
        voxels_delete_path = asset_dir / "voxels_delete.ply"
        if voxels_delete_path.is_file():
            coords = ply_to_coords(voxels_delete_path, device, resolution)
            mesh, mesh_meta = (None, {})
            if mask_glb.strip():
                mesh, mesh_meta = _load_mask_mesh(
                    mask_glb,
                    source_normalization=source_normalization,
                    ensure_path_exists_fn=ensure_path_exists_fn,
                )
            meta = {
                **mesh_meta,
                "mask_source": "voxels_delete_ply",
                "voxels_delete_path": str(voxels_delete_path),
                "voxel_count": int(coords.shape[0]),
                "enabled": True,
            }
            return MaskGLBResult(coords=coords, mesh=mesh, meta=meta)

    # Strategy 2: generate via VoxHammer filtering
    mesh, meta = _load_mask_mesh(
        mask_glb,
        source_normalization=source_normalization,
        ensure_path_exists_fn=ensure_path_exists_fn,
    )
    if mesh is None:
        return MaskGLBResult(coords=None, mesh=None, meta=meta)

    from editing.preprocess.voxel_filtering import process_voxels_with_improved_filtering

    preset_voxel_path = "assets/preset/preset_grid64.ply"
    voxel_size = 1.0 / float(resolution)

    tmp_mask_path = tmp_out_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            tmp_mask_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            tmp_out_path = f.name
        mesh.export(tmp_mask_path, file_type="ply")
        process_voxels_with_improved_filtering(
            preset_voxel_path,
            tmp_mask_path,
            tmp_out_path,
            method="volume",
            voxel_size=voxel_size,
            inside=True,
        )
        coords = ply_to_coords(Path(tmp_out_path), device, resolution)
    finally:
        for p in (tmp_mask_path, tmp_out_path):
            if p is not None:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    if coords.shape[0] == 0:
        raise RuntimeError("VoxHammer mask filtering produced no occupied voxels.")

    meta.update({
        "mask_source": "voxhammer_process_voxels_with_improved_filtering",
        "voxel_count": int(coords.shape[0]),
    })
    return MaskGLBResult(coords=coords, mesh=mesh, meta=meta)


def coords_to_flat_indices(coords: torch.Tensor, resolution: int = 64) -> torch.Tensor:
    """Convert 3D/4D coordinates to flat indices for lookup.

    Args:
        coords: Nx3 or Nx4 tensor of coordinates
        resolution: Voxel grid resolution

    Returns:
        N tensor of flat indices
    """
    coords = coords.long()
    if coords.shape[1] == 3:
        return coords[:, 0] * resolution * resolution + coords[:, 1] * resolution + coords[:, 2]
    if coords.shape[1] == 4:
        return (
            coords[:, 0] * resolution * resolution * resolution
            + coords[:, 1] * resolution * resolution
            + coords[:, 2] * resolution
            + coords[:, 3]
        )
    raise RuntimeError(f"Unsupported coordinate shape: {tuple(coords.shape)}")


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
):
    """Project source terminal noise onto target coordinates.

    For RF inversion: maps source noise features to new target structure,
    filling unmatched voxels with random noise.

    Args:
        source_noise: Source SparseTensor with terminal noise
        target_coords: Target Nx4 coordinates (batch, x, y, z)
        device: Target torch device
        SparseTensor: SparseTensor class from trellis
        resolution: Voxel grid resolution

    Returns:
        SparseTensor with projected features
    """
    if target_coords.numel() == 0:
        raise RuntimeError("Sparse-structure denoising produced no target voxels, so SLAT projection cannot continue.")

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
            if matched.any():
                matched_order = order[insert_pos[matched]]
                randn_feats[matched] = src_feats_full[matched_order]

        all_coords.append(tgt_coords_batch)
        all_feats.append(randn_feats)

    coords = torch.cat(all_coords, dim=0).to(device=device)
    feats = torch.cat(all_feats, dim=0).to(device=device)
    return SparseTensor(feats=feats, coords=coords)
