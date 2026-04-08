from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from editing.methods.base import EditMethod, EditMethodConfig, EditMethodInputs, EditMethodOutputs
from editing.preprocess.asset_3d import feats_to_slat


def coords3d(coords: torch.Tensor) -> torch.Tensor:
    """Extract 3D coordinates from coordinate tensor."""
    if coords.ndim != 2:
        raise RuntimeError(f"Expected 2D coords tensor, got shape {tuple(coords.shape)}")
    if coords.shape[1] == 4:
        return coords[:, 1:]
    if coords.shape[1] == 3:
        return coords
    raise RuntimeError(f"Unsupported coordinate shape: {tuple(coords.shape)}")


def coords_to_flat_indices(coords: torch.Tensor, grid_size: int = 128) -> torch.Tensor:
    """Convert 3D coordinates to flat indices for comparison."""
    return coords[:, 0] * (grid_size * grid_size) + coords[:, 1] * grid_size + coords[:, 2]


def sparse_batch_slice(sparse_tensor, batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract coordinates and features for a specific batch."""
    mask = sparse_tensor.coords[:, 0] == batch_idx
    coords = sparse_tensor.coords[mask]
    feats = sparse_tensor.feats[mask]
    return coords, feats


def compose_slat_keep_source_overlap(source_slat, target_slat) -> Tuple[object, dict, np.ndarray]:
    """
    Compose a final SLAT on the target coordinates.

    If a target voxel also exists in the source SLAT, reuse the source feature.
    Otherwise keep the target feature. Source-only voxels are omitted.

    Args:
        source_slat: Source SLAT tensor
        target_slat: Target SLAT tensor

    Returns:
        Tuple of (fused_slat, stats, reuse_mask)
    """
    src_batch_count = source_slat.shape[0]
    tgt_batch_count = target_slat.shape[0]
    fused_feats = target_slat.feats.clone()
    reuse_masks = []
    batch_stats = []

    for batch_idx in range(tgt_batch_count):
        tgt_coords_full, tgt_feats = sparse_batch_slice(target_slat, batch_idx)
        src_coords_full, src_feats = sparse_batch_slice(
            source_slat,
            batch_idx if src_batch_count > 1 else 0,
        )

        tgt_codes = coords_to_flat_indices(coords3d(tgt_coords_full))
        src_codes = coords_to_flat_indices(coords3d(src_coords_full))
        batch_reuse = torch.zeros(tgt_codes.shape[0], dtype=torch.bool, device=tgt_codes.device)
        batch_fused = tgt_feats.clone()

        if src_codes.numel() > 0 and tgt_codes.numel() > 0:
            src_codes_sorted, order = torch.sort(src_codes)
            insert_pos = torch.searchsorted(src_codes_sorted, tgt_codes)
            valid = insert_pos < src_codes_sorted.shape[0]
            matched = valid.clone()
            matched[valid] = src_codes_sorted[insert_pos[valid]] == tgt_codes[valid]
            if matched.any():
                matched_src_order = order[insert_pos[matched]]
                batch_fused[matched] = src_feats[matched_src_order]
                batch_reuse[matched] = True

        fused_feats[target_slat.layout[batch_idx]] = batch_fused
        reuse_masks.append(batch_reuse.detach().cpu())

        source_only = int((~torch.isin(src_codes, tgt_codes)).sum().item())
        overlap = int(batch_reuse.sum().item())
        target_only = int((~batch_reuse).sum().item())
        batch_stats.append(
            {
                "batch_idx": batch_idx,
                "source_voxel_count": int(src_codes.shape[0]),
                "target_voxel_count": int(tgt_codes.shape[0]),
                "overlap_reused_from_source_count": overlap,
                "target_only_generated_count": target_only,
                "source_only_dropped_count": source_only,
                "symmetric_difference_count": source_only + target_only,
            }
        )

    fused_slat = target_slat.replace(fused_feats)
    reuse_mask = torch.cat(reuse_masks, dim=0).numpy().astype(np.uint8)
    stats = {
        "composition_rule": "target_coords_only__reuse_source_on_overlap__keep_target_on_target_only",
        "source_batch_count": int(src_batch_count),
        "target_batch_count": int(tgt_batch_count),
        "source_batch_broadcasted": bool(src_batch_count == 1 and tgt_batch_count > 1),
        "fused_total_voxel_count": int(fused_slat.coords.shape[0]),
        "reused_source_voxel_count_total": int(reuse_mask.sum().item()),
        "generated_target_voxel_count_total": int(reuse_mask.shape[0] - reuse_mask.sum().item()),
        "batch_stats": batch_stats,
    }
    return fused_slat, stats, reuse_mask


class ImageSlatXorFusionMethod(EditMethod):
    """SLAT XOR Fusion editing method.

    Composes a new SLAT by:
    1. Loading source SLAT from features.npz
    2. Generating target structure and SLAT from edit image
    3. Fusing on target coordinates:
       - Overlapping voxels reuse source SLAT features
       - Target-only voxels keep generated features
       - Source-only voxels are dropped
    """

    def __init__(self):
        super().__init__("image_slat_xor_fusion")

    def prepare(
        self,
        pipeline,
        inputs: EditMethodInputs,
        config: EditMethodConfig,
    ) -> Dict:
        """Prepare SLAT fusion.

        Args:
            pipeline: TRELLIS image-to-3D pipeline
            inputs: Preprocessed inputs with asset_dir containing features.npz
            config: Method configuration

        Returns:
            Dictionary with source SLAT and edit condition
        """
        # Load source SLAT from features.npz
        if inputs.asset_dir is None:
            raise RuntimeError("image_slat_xor_fusion requires asset_dir with features.npz")

        features_path = inputs.asset_dir / "features.npz"
        if not features_path.exists():
            raise RuntimeError(f"features.npz not found in {inputs.asset_dir}")

        # Load features
        from trellis.modules import sparse as sp
        source_slat = feats_to_slat(pipeline, features_path, sp.SparseTensor)

        # Encode edit condition
        edit_cond_dict = pipeline.get_cond([inputs.edit_image])

        return {
            "source_slat": source_slat,
            "edit_cond_dict": edit_cond_dict,
        }

    def run(
        self,
        pipeline,
        prepared_state: Dict,
        config: EditMethodConfig,
    ) -> EditMethodOutputs:
        """Run SLAT XOR fusion.

        Args:
            pipeline: TRELLIS image-to-3D pipeline
            prepared_state: State from prepare()
            config: Method configuration

        Returns:
            EditMethodOutputs with fused results
        """
        source_slat = prepared_state["source_slat"]
        edit_cond_dict = prepared_state["edit_cond_dict"]

        # Get sampler params
        ss_params = config.sparse_structure_sampler_params or {}
        slat_params = config.slat_sampler_params or {}

        # Generate target structure and SLAT
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        target_coords = pipeline.sample_sparse_structure(
            edit_cond_dict,
            num_samples=config.num_samples,
            sampler_params=ss_params,
        )
        target_slat = pipeline.sample_slat(
            edit_cond_dict,
            target_coords,
            sampler_params=slat_params,
        )

        # Fuse SLATs
        fused_slat, fusion_stats, reuse_mask = compose_slat_keep_source_overlap(
            source_slat, target_slat
        )

        # Decode fused SLAT
        outputs = pipeline.decode_slat(fused_slat, ["mesh", "gaussian"])

        return EditMethodOutputs(
            outputs=outputs,
            metadata={
                "fusion_stats": fusion_stats,
                "reuse_mask_shape": reuse_mask.shape,
            },
        )

    def save_artifacts(
        self,
        outputs: EditMethodOutputs,
        out_dir: Path,
        config: EditMethodConfig,
    ) -> Dict[str, str]:
        """Save method-specific artifacts.

        Args:
            outputs: Results from run()
            out_dir: Output directory
            config: Method configuration

        Returns:
            Dictionary of artifact paths
        """
        from editing.common import save_outputs, write_json

        extra = config.extra_params or {}
        skip_render = extra.get("skip_render", True)
        skip_glb = extra.get("skip_glb", False)
        skip_ply = extra.get("skip_ply", False)

        # Save main outputs
        save_outputs(
            outputs=outputs.outputs,
            out_dir=out_dir,
            skip_render=skip_render,
            skip_glb=skip_glb,
            skip_ply=skip_ply,
        )

        # Save fusion stats
        artifact_paths = {}
        if outputs.metadata:
            fusion_stats_path = out_dir / "fusion_stats.json"
            write_json(fusion_stats_path, outputs.metadata["fusion_stats"])
            artifact_paths["fusion_stats"] = str(fusion_stats_path)

        return artifact_paths

    def get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "skip_render": True,
            "skip_glb": False,
            "skip_ply": False,
        }
