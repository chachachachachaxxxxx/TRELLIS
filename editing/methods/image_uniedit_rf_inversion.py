from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, Tuple

import torch

from editing.inversion import invert_slat, invert_sparse_structure
from editing.inversion.uniedit_sampler import UniEditRFSampler
from editing.methods.base import EditMethod, EditMethodConfig, EditMethodInputs, EditMethodOutputs
from editing.preprocess.asset_3d import coords_to_voxel, feats_to_slat, ply_to_coords, project_sparse_terminal_noise
from editing.utils.uniedit_utils import build_sparse_replace_index_map, build_stage2_selector, compose_stage1_coords


class ImageUniEditRFInversionMethod(EditMethod):
    """UniEdit-style RF Inversion editing method.

    Uses RF inversion with two-stage voxel editing:
    1. Stage 1: Edit voxel structure with mask guidance
    2. Stage 2: Edit SLAT features with three ablation modes
    """

    def __init__(self):
        super().__init__("image_uniedit_rf_inversion")

    def prepare(
        self,
        pipeline,
        inputs: EditMethodInputs,
        config: EditMethodConfig,
    ) -> Dict:
        """Prepare UniEdit RF inversion.

        Args:
            pipeline: TRELLIS image-to-3D pipeline
            inputs: Preprocessed inputs with source assets and mask_glb
            config: Method configuration

        Returns:
            Dictionary with prepared state
        """
        # Validate inputs
        if inputs.source_voxels_path is None or inputs.source_features_path is None:
            raise RuntimeError("UniEdit RF inversion requires source_voxels_path and source_features_path")
        if inputs.mask_glb_path is None:
            raise RuntimeError("UniEdit RF inversion requires mask_glb_path")

        # Get extra params
        extra = config.extra_params or {}
        ss_omega = extra.get("ss_omega", 1.0)
        slat_omega = extra.get("slat_omega", 1.0)
        cfg_interval = extra.get("cfg_interval", (0.5, 1.0))
        stage2_variant = extra.get("stage2_variant", "preserve_uniedit")

        # Get resolution
        resolution = pipeline.sparse_structure_sampler_params.get("grid_size", 64)

        # Load source assets
        source_coords = ply_to_coords(inputs.source_voxels_path, pipeline.device, resolution)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        from trellis.modules import sparse as sp
        source_slat = feats_to_slat(pipeline, inputs.source_features_path, sp.SparseTensor)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Load mask coordinates
        mask_coords = None
        if inputs.mask_glb_path:
            from editing.preprocess.asset_3d import load_mask_glb_coords
            mask_result = load_mask_glb_coords(
                mask_glb=str(inputs.mask_glb_path),
                device=pipeline.device,
                resolution=resolution,
                asset_dir=inputs.source_voxels_path.parent if inputs.source_voxels_path else None,
            )
            mask_coords = mask_result.coords

        # Encode conditions
        source_cond_dict = pipeline.get_cond([inputs.source_image])
        edit_cond_dict = pipeline.get_cond([inputs.edit_image])

        source_cond = source_cond_dict["cond"]
        edit_cond = edit_cond_dict["cond"]
        neg_cond = source_cond_dict["neg_cond"]

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "source_coords": source_coords,
            "source_slat": source_slat,
            "mask_coords": mask_coords,
            "source_cond": source_cond,
            "edit_cond": edit_cond,
            "neg_cond": neg_cond,
            "ss_omega": ss_omega,
            "slat_omega": slat_omega,
            "cfg_interval": cfg_interval,
            "stage2_variant": stage2_variant,
            "resolution": resolution,
        }

    def run(
        self,
        pipeline,
        prepared_state: Dict,
        config: EditMethodConfig,
    ) -> EditMethodOutputs:
        """Run UniEdit RF inversion.

        Args:
            pipeline: TRELLIS image-to-3D pipeline
            prepared_state: State from prepare()
            config: Method configuration

        Returns:
            EditMethodOutputs with results
        """
        # Extract prepared state
        source_coords = prepared_state["source_coords"]
        source_slat = prepared_state["source_slat"]
        mask_coords = prepared_state["mask_coords"]
        source_cond = prepared_state["source_cond"]
        edit_cond = prepared_state["edit_cond"]
        neg_cond = prepared_state["neg_cond"]
        ss_omega = prepared_state["ss_omega"]
        slat_omega = prepared_state["slat_omega"]
        cfg_interval = prepared_state["cfg_interval"]
        stage2_variant = prepared_state["stage2_variant"]
        resolution = prepared_state["resolution"]

        # Get sampling params
        ss_params = pipeline.sparse_structure_sampler_params
        slat_params = pipeline.slat_sampler_params

        # Stage 0: Invert source assets
        print("Stage 0: Inverting source assets...")
        source_voxel = coords_to_voxel(source_coords, pipeline.device, resolution)

        ss_terminal_noise = invert_sparse_structure(
            pipeline=pipeline,
            cond_src={"cond": source_cond, "neg_cond": neg_cond},
            voxel_src=source_voxel,
            params=ss_params,
            cfg_interval=cfg_interval,
            verbose=True,
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Stage 0b: Invert SLAT (with trajectory caching if needed)
        slat_latent_cache = None
        if stage2_variant == "latent_replace_union":
            print("Stage 0b: Inverting SLAT with trajectory caching...")
            slat_terminal_noise, slat_latent_cache = self._invert_slat_with_cache(
                pipeline=pipeline,
                cond_src={"cond": source_cond, "neg_cond": neg_cond},
                slat_src=source_slat,
                params=slat_params,
                cfg_interval=cfg_interval,
            )
        else:
            slat_terminal_noise = invert_slat(
                pipeline=pipeline,
                cond_src={"cond": source_cond, "neg_cond": neg_cond},
                slat_src=source_slat,
                params=slat_params,
                cfg_interval=cfg_interval,
                verbose=True,
            )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Stage 1: Edit sparse structure with UniEdit
        print(f"Stage 1: Editing sparse structure (omega={ss_omega})...")
        coords_stage1_raw = self._denoise_sparse_structure_uniedit(
            pipeline=pipeline,
            source_cond={"cond": source_cond, "neg_cond": neg_cond},
            target_cond={"cond": edit_cond, "neg_cond": neg_cond},
            terminal_noise=ss_terminal_noise,
            params=ss_params,
            cfg_interval=cfg_interval,
            omega=ss_omega,
        )

        # Apply mask to Stage 1 results
        coords_stage1_masked, coords_preserve, stage1_meta = compose_stage1_coords(
            coords_source=source_coords,
            coords_stage1_raw=coords_stage1_raw,
            mask_coords=mask_coords,
        )

        print(f"Stage 1 complete: {stage1_meta['stage1_masked_voxel_count']} voxels")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Project SLAT noise to Stage 1 coordinates
        projected_slat_noise = project_sparse_terminal_noise(
            source_noise=slat_terminal_noise,
            target_coords=coords_stage1_masked.to(device=pipeline.device),
            device=pipeline.device,
            SparseTensor=type(slat_terminal_noise),
            resolution=resolution,
        )

        # Build Stage 2 selector
        from editing.utils.uniedit_utils import coords3d_to_batched
        stage2_selector = build_stage2_selector(
            coords_stage1_masked,
            coords3d_to_batched(coords_preserve, batch_idx=0),
        )

        # Stage 2: Edit SLAT features
        print(f"Stage 2: Editing SLAT features (variant={stage2_variant}, omega={slat_omega})...")

        if stage2_variant == "preserve_uniedit":
            slat_tgt = self._denoise_slat_variant(
                pipeline=pipeline,
                source_cond={"cond": source_cond, "neg_cond": neg_cond},
                target_cond={"cond": edit_cond, "neg_cond": neg_cond},
                terminal_noise=projected_slat_noise,
                params=slat_params,
                cfg_interval=cfg_interval,
                omega=slat_omega,
                selector=stage2_selector,
                mode="preserve_overlap",
            )
        elif stage2_variant == "free_target":
            slat_tgt = self._denoise_slat_variant(
                pipeline=pipeline,
                source_cond={"cond": source_cond, "neg_cond": neg_cond},
                target_cond={"cond": edit_cond, "neg_cond": neg_cond},
                terminal_noise=projected_slat_noise,
                params=slat_params,
                cfg_interval=cfg_interval,
                omega=slat_omega,
                selector=None,
                mode="target_only",
            )
        elif stage2_variant == "latent_replace_union":
            # Build replacement indices
            stage2_replace_target_idx, stage2_replace_source_idx = build_sparse_replace_index_map(
                coords_target=projected_slat_noise.coords,
                coords_source=slat_terminal_noise.coords,
                selector=stage2_selector,
            )
            slat_tgt = self._denoise_slat_latent_replace(
                pipeline=pipeline,
                target_cond={"cond": edit_cond, "neg_cond": neg_cond},
                terminal_noise=projected_slat_noise,
                params=slat_params,
                cfg_interval=cfg_interval,
                latent_cache=slat_latent_cache,
                replace_target_indices=stage2_replace_target_idx,
                replace_source_indices=stage2_replace_source_idx,
            )
        else:
            raise NotImplementedError(
                f"Stage 2 variant '{stage2_variant}' not implemented. "
                f"Supported: preserve_uniedit, free_target, latent_replace_union."
            )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Decode
        print("Decoding final result...")
        extra = config.extra_params or {}
        decode_modes = extra.get("decode_modes", ["mesh"])

        # Handle string input (e.g., "gaussian,mesh" from CLI)
        if isinstance(decode_modes, str):
            decode_modes = [m.strip() for m in decode_modes.split(",")]

        print(f"Decode modes: {decode_modes}")
        outputs = pipeline.decode_slat(slat_tgt, decode_modes)

        return EditMethodOutputs(
            outputs=outputs,
            metadata={
                "stage1_meta": stage1_meta,
                "stage2_variant": stage2_variant,
                "ss_omega": ss_omega,
                "slat_omega": slat_omega,
            },
        )

    def _denoise_sparse_structure_uniedit(
        self,
        pipeline,
        source_cond: dict,
        target_cond: dict,
        terminal_noise: torch.Tensor,
        params: dict,
        cfg_interval: Tuple[float, float],
        omega: float,
    ) -> torch.Tensor:
        """Denoise sparse structure with UniEdit fusion."""
        flow_model = pipeline.models["sparse_structure_flow_model"]
        decoder = pipeline.models["sparse_structure_decoder"]
        sampler = UniEditRFSampler()

        z_tgt = sampler.sample(
            model=flow_model,
            sample=terminal_noise,
            source_cond=source_cond["cond"],
            target_cond=target_cond["cond"],
            neg_cond=target_cond["neg_cond"],
            steps=params["steps"],
            rescale_t=params["rescale_t"],
            cfg_strength=params["cfg_strength"],
            cfg_interval=cfg_interval,
            omega=float(omega),
            selector=None,
            mode="full_uniedit",
            verbose=True,
        )

        voxel = decoder(z_tgt)
        coords = torch.argwhere(voxel > 0)[:, [0, 2, 3, 4]].int()
        if coords.shape[0] == 0:
            raise RuntimeError("Stage 1 UniEdit sparse-structure denoising produced an empty target structure.")
        return coords

    def _denoise_slat_variant(
        self,
        pipeline,
        source_cond: dict,
        target_cond: dict,
        terminal_noise,
        params: dict,
        cfg_interval: Tuple[float, float],
        omega: float,
        selector,
        mode: str,
    ):
        """Denoise SLAT with UniEdit variant."""
        flow_model = pipeline.models["slat_flow_model"]
        sampler = UniEditRFSampler()

        slat_normalized = sampler.sample(
            model=flow_model,
            sample=terminal_noise,
            source_cond=source_cond["cond"],
            target_cond=target_cond["cond"],
            neg_cond=target_cond["neg_cond"],
            steps=params["steps"],
            rescale_t=params["rescale_t"],
            cfg_strength=params["cfg_strength"],
            cfg_interval=cfg_interval,
            omega=float(omega),
            selector=selector,
            mode=mode,
            verbose=True,
        )

        # Denormalize SLAT
        from editing.inversion.rf_inversion import get_slat_norm_tensors
        mean, std = get_slat_norm_tensors(
            pipeline,
            slat_normalized.device,
            slat_normalized.feats.dtype,
        )
        return slat_normalized * std + mean

    def _invert_slat_with_cache(
        self,
        pipeline,
        cond_src: dict,
        slat_src,
        params: dict,
        cfg_interval: Tuple[float, float],
    ):
        """Invert SLAT with trajectory caching for latent_replace_union."""
        from editing.inversion.latent_replace_sampler import SparseLatentReplaceRFSampler
        from editing.inversion.rf_inversion import get_slat_norm_tensors

        flow_model = pipeline.models["slat_flow_model"]
        mean, std = get_slat_norm_tensors(pipeline, slat_src.device, slat_src.feats.dtype)
        slat_normalized = (slat_src - mean) / std
        sampler = SparseLatentReplaceRFSampler()
        return sampler.invert_with_cache(
            model=flow_model,
            sample=slat_normalized,
            cond_dict=cond_src,
            steps=params["steps"],
            rescale_t=params["rescale_t"],
            cfg_strength=params["cfg_strength"],
            cfg_interval=cfg_interval,
            verbose=True,
        )

    def _denoise_slat_latent_replace(
        self,
        pipeline,
        target_cond: dict,
        terminal_noise,
        params: dict,
        cfg_interval: Tuple[float, float],
        latent_cache: dict,
        replace_target_indices: torch.Tensor,
        replace_source_indices: torch.Tensor,
    ):
        """Denoise SLAT with latent replacement."""
        from editing.inversion.latent_replace_sampler import SparseLatentReplaceRFSampler
        from editing.inversion.rf_inversion import get_slat_norm_tensors

        flow_model = pipeline.models["slat_flow_model"]
        sampler = SparseLatentReplaceRFSampler()
        slat_normalized = sampler.sample_with_replacement(
            model=flow_model,
            sample=terminal_noise,
            cond_dict=target_cond,
            steps=params["steps"],
            rescale_t=params["rescale_t"],
            cfg_strength=params["cfg_strength"],
            cfg_interval=cfg_interval,
            latent_cache=latent_cache,
            replace_target_indices=replace_target_indices,
            replace_source_indices=replace_source_indices,
            verbose=True,
        )
        mean, std = get_slat_norm_tensors(
            pipeline,
            slat_normalized.device,
            slat_normalized.feats.dtype,
        )
        return slat_normalized * std + mean

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
        from editing.common import save_outputs

        extra = config.extra_params or {}
        skip_render = extra.get("skip_render", True)
        skip_glb = extra.get("skip_glb", False)
        skip_ply = extra.get("skip_ply", False)

        save_outputs(
            outputs=outputs.outputs,
            out_dir=out_dir,
            skip_render=skip_render,
            skip_glb=skip_glb,
            skip_ply=skip_ply,
        )

        # Save metadata
        if outputs.metadata:
            import json
            metadata_path = out_dir / "uniedit_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(outputs.metadata, f, indent=2)

        return {}

    def cleanup(self):
        """Clean up resources."""
        pass

    def get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "skip_render": True,
            "skip_glb": False,
            "skip_ply": False,
            "ss_omega": 1.0,
            "slat_omega": 1.0,
            "cfg_interval": (0.5, 1.0),
            "stage2_variant": "preserve_uniedit",  # or "free_target", "latent_replace_union"
            "decode_modes": ["gaussian", "mesh"],  # Need both for GLB/PLY export
        }
