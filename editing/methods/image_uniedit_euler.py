"""UniEdit with first-order Euler + predictor-corrector editing method."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, Tuple

import torch

from editing.inversion import invert_slat, invert_sparse_structure
from editing.inversion.uniedit_euler_sampler import UniEditEulerSampler
from editing.methods.base import EditMethod, EditMethodConfig, EditMethodInputs, EditMethodOutputs
from editing.preprocess.asset_3d import coords_to_voxel, feats_to_slat, ply_to_coords, project_sparse_terminal_noise
from editing.utils.uniedit_utils import build_sparse_replace_index_map, build_stage2_selector, compose_stage1_coords


class ImageUniEditEulerMethod(EditMethod):
    """UniEdit with first-order Euler + predictor-corrector.

    Simplified version that uses:
    - First-order Euler integration (no second-order correction)
    - Predictor-corrector for improved accuracy
    - Delayed inversion/editing with alpha parameter
    - Source/target velocity fusion with omega parameter
    """

    def __init__(self):
        super().__init__("image_uniedit_euler")

    def prepare(
        self,
        pipeline,
        inputs: EditMethodInputs,
        config: EditMethodConfig,
    ) -> Dict:
        """Prepare UniEdit Euler inversion.

        Args:
            pipeline: TRELLIS image-to-3D pipeline
            inputs: Preprocessed inputs with source assets and mask_glb
            config: Method configuration

        Returns:
            Dictionary with prepared state
        """
        # Validate inputs
        if inputs.source_voxels_path is None or inputs.source_features_path is None:
            raise RuntimeError("UniEdit Euler requires source_voxels_path and source_features_path")
        if inputs.mask_glb_path is None:
            raise RuntimeError("UniEdit Euler requires mask_glb_path")

        # Get extra params
        extra = config.extra_params or {}
        ss_omega = extra.get("ss_omega", 1.0)
        slat_omega = extra.get("slat_omega", 1.0)
        ss_alpha = extra.get("ss_alpha", 0.5)
        slat_alpha = extra.get("slat_alpha", 0.5)
        cfg_interval = extra.get("cfg_interval", (0.5, 1.0))
        zero_init = extra.get("zero_init", False)

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
            "ss_alpha": ss_alpha,
            "slat_alpha": slat_alpha,
            "cfg_interval": cfg_interval,
            "zero_init": zero_init,
            "resolution": resolution,
        }

    def run(
        self,
        pipeline,
        prepared_state: Dict,
        config: EditMethodConfig,
    ) -> EditMethodOutputs:
        """Run UniEdit Euler editing.

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
        ss_alpha = prepared_state["ss_alpha"]
        slat_alpha = prepared_state["slat_alpha"]
        cfg_interval = prepared_state["cfg_interval"]
        zero_init = prepared_state["zero_init"]
        resolution = prepared_state["resolution"]

        # Get sampling params
        ss_params = pipeline.sparse_structure_sampler_params
        slat_params = pipeline.slat_sampler_params

        # Stage 0: Invert source assets to junction point
        print(f"Stage 0: Inverting source assets (alpha={ss_alpha})...")
        source_voxel = coords_to_voxel(source_coords, pipeline.device, resolution)

        ss_junction = self._invert_sparse_structure_euler(
            pipeline=pipeline,
            cond={"cond": source_cond, "neg_cond": neg_cond},
            voxel_src=source_voxel,
            params=ss_params,
            cfg_interval=cfg_interval,
            alpha=ss_alpha,
            zero_init=zero_init,
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Stage 0b: Invert SLAT to junction point
        print(f"Stage 0b: Inverting SLAT (alpha={slat_alpha})...")
        slat_junction = self._invert_slat_euler(
            pipeline=pipeline,
            cond={"cond": source_cond, "neg_cond": neg_cond},
            slat_src=source_slat,
            params=slat_params,
            cfg_interval=cfg_interval,
            alpha=slat_alpha,
            zero_init=zero_init,
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Stage 1: Edit sparse structure with UniEdit
        print(f"Stage 1: Editing sparse structure (omega={ss_omega}, alpha={ss_alpha})...")
        coords_stage1_raw = self._denoise_sparse_structure_euler(
            pipeline=pipeline,
            source_cond={"cond": source_cond, "neg_cond": neg_cond},
            target_cond={"cond": edit_cond, "neg_cond": neg_cond},
            junction=ss_junction,
            params=ss_params,
            cfg_interval=cfg_interval,
            omega=ss_omega,
            alpha=ss_alpha,
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

        # Project SLAT junction to Stage 1 coordinates
        projected_slat_junction = project_sparse_terminal_noise(
            source_noise=slat_junction,
            target_coords=coords_stage1_masked.to(device=pipeline.device),
            device=pipeline.device,
            SparseTensor=type(slat_junction),
            resolution=resolution,
        )

        # Build Stage 2 selector
        from editing.utils.uniedit_utils import coords3d_to_batched
        stage2_selector = build_stage2_selector(
            coords_stage1_masked,
            coords3d_to_batched(coords_preserve, batch_idx=0),
        )

        # Stage 2: Edit SLAT features
        print(f"Stage 2: Editing SLAT features (omega={slat_omega}, alpha={slat_alpha})...")
        slat_tgt = self._denoise_slat_euler(
            pipeline=pipeline,
            source_cond={"cond": source_cond, "neg_cond": neg_cond},
            target_cond={"cond": edit_cond, "neg_cond": neg_cond},
            junction=projected_slat_junction,
            params=slat_params,
            cfg_interval=cfg_interval,
            omega=slat_omega,
            alpha=slat_alpha,
            selector=stage2_selector,
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Decode
        print("Decoding final result...")
        extra = config.extra_params or {}
        decode_modes = extra.get("decode_modes", ["mesh", "gaussian"])

        # Handle string input
        if isinstance(decode_modes, str):
            import json
            try:
                decode_modes = json.loads(decode_modes)
            except (json.JSONDecodeError, ValueError):
                decode_modes = [m.strip() for m in decode_modes.split(",")]

        print(f"Decode modes: {decode_modes}")
        outputs = pipeline.decode_slat(slat_tgt, decode_modes)

        return EditMethodOutputs(
            outputs=outputs,
            metadata={
                "stage1_meta": stage1_meta,
                "ss_omega": ss_omega,
                "slat_omega": slat_omega,
                "ss_alpha": ss_alpha,
                "slat_alpha": slat_alpha,
            },
        )

    def _invert_sparse_structure_euler(
        self,
        pipeline,
        cond: dict,
        voxel_src: torch.Tensor,
        params: dict,
        cfg_interval: Tuple[float, float],
        alpha: float,
        zero_init: bool,
    ) -> torch.Tensor:
        """Invert sparse structure with Euler + predictor-corrector."""
        flow_model = pipeline.models["sparse_structure_flow_model"]
        encoder = pipeline.models["sparse_structure_encoder"]
        sampler = UniEditEulerSampler()

        z_src = encoder(voxel_src)
        junction = sampler.invert(
            model=flow_model,
            sample=z_src,
            cond=cond["cond"],
            neg_cond=cond["neg_cond"],
            steps=params["steps"],
            rescale_t=params["rescale_t"],
            cfg_strength=params["cfg_strength"],
            cfg_interval=cfg_interval,
            alpha=alpha,
            zero_init=zero_init,
            verbose=True,
        )
        return junction

    def _invert_slat_euler(
        self,
        pipeline,
        cond: dict,
        slat_src,
        params: dict,
        cfg_interval: Tuple[float, float],
        alpha: float,
        zero_init: bool,
    ):
        """Invert SLAT with Euler + predictor-corrector."""
        from editing.inversion.rf_inversion import get_slat_norm_tensors

        flow_model = pipeline.models["slat_flow_model"]
        mean, std = get_slat_norm_tensors(pipeline, slat_src.device, slat_src.feats.dtype)
        slat_normalized = (slat_src - mean) / std
        sampler = UniEditEulerSampler()

        junction = sampler.invert(
            model=flow_model,
            sample=slat_normalized,
            cond=cond["cond"],
            neg_cond=cond["neg_cond"],
            steps=params["steps"],
            rescale_t=params["rescale_t"],
            cfg_strength=params["cfg_strength"],
            cfg_interval=cfg_interval,
            alpha=alpha,
            zero_init=zero_init,
            verbose=True,
        )
        return junction

    def _denoise_sparse_structure_euler(
        self,
        pipeline,
        source_cond: dict,
        target_cond: dict,
        junction: torch.Tensor,
        params: dict,
        cfg_interval: Tuple[float, float],
        omega: float,
        alpha: float,
    ) -> torch.Tensor:
        """Denoise sparse structure with Euler + UniEdit fusion."""
        flow_model = pipeline.models["sparse_structure_flow_model"]
        decoder = pipeline.models["sparse_structure_decoder"]
        sampler = UniEditEulerSampler()

        z_tgt = sampler.edit(
            model=flow_model,
            sample=junction,
            source_cond=source_cond["cond"],
            target_cond=target_cond["cond"],
            neg_cond=target_cond["neg_cond"],
            steps=params["steps"],
            rescale_t=params["rescale_t"],
            cfg_strength=params["cfg_strength"],
            cfg_interval=cfg_interval,
            omega=float(omega),
            alpha=alpha,
            selector=None,
            verbose=True,
        )

        voxel = decoder(z_tgt)
        coords = torch.argwhere(voxel > 0)[:, [0, 2, 3, 4]].int()
        if coords.shape[0] == 0:
            raise RuntimeError("Stage 1 UniEdit Euler sparse-structure denoising produced an empty target structure.")
        return coords

    def _denoise_slat_euler(
        self,
        pipeline,
        source_cond: dict,
        target_cond: dict,
        junction,
        params: dict,
        cfg_interval: Tuple[float, float],
        omega: float,
        alpha: float,
        selector,
    ):
        """Denoise SLAT with Euler + UniEdit fusion."""
        from editing.inversion.rf_inversion import get_slat_norm_tensors

        flow_model = pipeline.models["slat_flow_model"]
        sampler = UniEditEulerSampler()

        slat_normalized = sampler.edit(
            model=flow_model,
            sample=junction,
            source_cond=source_cond["cond"],
            target_cond=target_cond["cond"],
            neg_cond=target_cond["neg_cond"],
            steps=params["steps"],
            rescale_t=params["rescale_t"],
            cfg_strength=params["cfg_strength"],
            cfg_interval=cfg_interval,
            omega=float(omega),
            alpha=alpha,
            selector=selector,
            verbose=True,
        )

        # Denormalize SLAT
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
        """Save method-specific artifacts."""
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
            metadata_path = out_dir / "uniedit_euler_metadata.json"
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
            "ss_alpha": 0.5,
            "slat_alpha": 0.5,
            "cfg_interval": (0.5, 1.0),
            "zero_init": False,
            "decode_modes": ["gaussian", "mesh"],
        }
