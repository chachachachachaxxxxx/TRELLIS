from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from editing.common import save_outputs
from editing.hooks import PromptToPromptHook, StageConfig
from editing.inversion import invert_slat, invert_sparse_structure
from editing.inversion.uniedit_sampler import UniEditRFSolver
from editing.methods.base import EditMethod, EditMethodConfig, EditMethodInputs, EditMethodOutputs
from editing.preprocess.asset_3d import coords_to_voxel, feats_to_slat, ply_to_coords, project_sparse_terminal_noise
from editing.utils import build_image_token_metadata, resolve_patch_size
from editing.utils.uniedit_utils import build_sparse_replace_index_map, build_stage2_selector, compose_stage1_coords


class ImageUniEditP2PHybridMethod(EditMethod):
    """Hybrid method combining UniEdit latent replacement with P2P attention injection.

    This method uses:
    1. UniEdit's RF inversion and latent replacement for structure preservation
    2. Prompt-to-Prompt attention injection for fine-grained control

    The combination provides both strong structural preservation (from UniEdit)
    and flexible attention-based editing (from P2P).
    """

    def __init__(self):
        super().__init__("image_uniedit_p2p_hybrid")
        self.hook: PromptToPromptHook | None = None

    def prepare(
        self,
        pipeline,
        inputs: EditMethodInputs,
        config: EditMethodConfig,
    ) -> Dict:
        """Prepare hybrid UniEdit + P2P editing.

        Args:
            pipeline: TRELLIS image-to-3D pipeline
            inputs: Preprocessed inputs with source assets and masks
            config: Method configuration

        Returns:
            Dictionary with prepared state
        """
        # Validate inputs
        if inputs.source_voxels_path is None or inputs.source_features_path is None:
            raise RuntimeError("Hybrid method requires source_voxels_path and source_features_path")
        if inputs.mask_glb_path is None:
            raise RuntimeError("Hybrid method requires mask_glb_path for UniEdit")
        if inputs.mask_image is None:
            raise RuntimeError("Hybrid method requires mask_image for P2P")

        # Get extra params
        extra = config.extra_params or {}
        ss_omega = extra.get("ss_omega", 1.0)
        slat_omega = extra.get("slat_omega", 1.0)
        cfg_interval = extra.get("cfg_interval", (0.5, 1.0))
        stage2_variant = extra.get("stage2_variant", "latent_replace_union")

        # P2P params
        inject_stages = extra.get("inject_stages", ["slat"])  # Default: only inject in SLAT stage
        patch_coverage_threshold = extra.get("patch_coverage_threshold", 0.0)
        query_chunk = extra.get("query_chunk", 1024)
        p2p_strength = extra.get("p2p_strength", 0.8)  # Slightly lower to let UniEdit dominate

        # Get resolution
        resolution = pipeline.sparse_structure_sampler_params.get("grid_size", 64)

        # Load source assets
        print("Loading source assets...")
        source_coords = ply_to_coords(inputs.source_voxels_path, pipeline.device, resolution)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        from trellis.modules import sparse as sp
        source_slat = feats_to_slat(pipeline, inputs.source_features_path, sp.SparseTensor)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Load mask coordinates for UniEdit
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

        # Build P2P stage configs
        stage_configs = {}
        for stage_name in ["sparse_structure", "slat"]:
            enabled = stage_name in inject_stages
            t_start = extra.get(f"{stage_name}_t_start", 1.0 if stage_name == "sparse_structure" else 0.8)
            t_end = extra.get(f"{stage_name}_t_end", 0.3 if stage_name == "sparse_structure" else 0.0)
            strength = extra.get(f"{stage_name}_strength", p2p_strength)
            stage_configs[stage_name] = StageConfig(
                name=stage_name,
                enabled=enabled,
                t_start=t_start,
                t_end=t_end,
                strength=strength,
            )

        # Build token metadata for P2P
        patch_size = resolve_patch_size(pipeline.models["image_cond_model"].patch_size)
        token_meta = build_image_token_metadata(
            cond=edit_cond,
            mask=inputs.mask_image,
            patch_size=patch_size,
            patch_coverage_threshold=patch_coverage_threshold,
        )

        # Create P2P hook
        self.hook = PromptToPromptHook(
            source_cond=source_cond,
            edit_cond=edit_cond,
            neg_cond=neg_cond,
            token_meta=token_meta,
            stage_configs=stage_configs,
            query_chunk=query_chunk,
        )

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
            "token_meta": token_meta,
            "stage_configs": stage_configs,
        }

    def run(
        self,
        pipeline,
        prepared_state: Dict,
        config: EditMethodConfig,
    ) -> EditMethodOutputs:
        """Run hybrid UniEdit + P2P editing.

        This method combines:
        1. UniEdit's latent replacement for structural preservation
        2. P2P's attention injection for fine-grained control

        Both mechanisms work simultaneously during denoising.

        Args:
            pipeline: TRELLIS image-to-3D pipeline
            prepared_state: State from prepare()
            config: Method configuration

        Returns:
            EditMethodOutputs with results
        """
        from trellis.modules import sparse as sp

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

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        ss_terminal_noise = invert_sparse_structure(
            pipeline=pipeline,
            cond_src={"cond": source_cond, "neg_cond": neg_cond},
            voxel_src=source_voxel,
            params=ss_params,
            cfg_interval=cfg_interval,
            verbose=True,
        )

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
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
        print("Stage 1: Editing sparse structure with UniEdit...")

        # Build replace index map for stage 1
        replace_index_map = build_sparse_replace_index_map(
            source_coords=source_coords,
            mask_coords=mask_coords,
            device=pipeline.device,
        )

        # Denoise with UniEdit sampler (stage 1)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        solver_ss = UniEditRFSolver(
            model=pipeline.models["sparse_structure_flow_model"],
            cond_src={"cond": source_cond, "neg_cond": neg_cond},
            cond_tgt={"cond": edit_cond, "neg_cond": neg_cond},
            terminal_noise_src=ss_terminal_noise,
            terminal_noise_tgt=ss_terminal_noise,  # Use same noise, no projection needed
            replace_index_map=replace_index_map,
            omega=ss_omega,
            steps=ss_params["steps"],
            rescale_t=ss_params["rescale_t"],
            cfg_strength=ss_params["cfg_strength"],
            cfg_interval=cfg_interval,
        )
        coords_tgt = solver_ss.sample(verbose=True)

        # Compose final coords
        coords_final = compose_stage1_coords(
            coords_src=source_coords,
            coords_tgt=coords_tgt,
            mask_coords=mask_coords,
        )

        del ss_terminal_noise, coords_tgt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Stage 2: Edit SLAT with UniEdit + P2P hybrid
        print(f"Stage 2: Editing SLAT with UniEdit ({stage2_variant}) + P2P attention injection...")

        # Build stage 2 selector
        selector = build_stage2_selector(
            variant=stage2_variant,
            source_coords=source_coords,
            coords_final=coords_final,
            mask_coords=mask_coords,
            source_slat=source_slat,
            device=pipeline.device,
        )

        # Patch models with P2P hook BEFORE denoising
        if self.hook:
            print("Patching models with P2P attention injection...")
            self.hook.patch_model(pipeline.models["sparse_structure_flow_model"], "sparse_structure")
            self.hook.patch_model(pipeline.models["slat_flow_model"], "slat")

        # Denoise with UniEdit sampler (stage 2) - P2P hook will intercept attention
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        solver_slat = UniEditRFSolver(
            model=pipeline.models["slat_flow_model"],
            cond_src={"cond": source_cond, "neg_cond": neg_cond},
            cond_tgt={"cond": edit_cond, "neg_cond": neg_cond},
            terminal_noise_src=slat_terminal_noise,
            terminal_noise_tgt=slat_terminal_noise,  # Use same noise for SLAT
            replace_index_map=selector,
            omega=slat_omega,
            steps=slat_params["steps"],
            rescale_t=slat_params["rescale_t"],
            cfg_strength=slat_params["cfg_strength"],
            cfg_interval=cfg_interval,
        )
        slat_tgt = solver_slat.sample(coords=coords_final, verbose=True)

        del slat_terminal_noise
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Decode
        print("Decoding final result...")
        extra = config.extra_params or {}
        decode_modes = extra.get("decode_modes", ["mesh", "gaussian"])

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
            source_outputs=None,
            metadata=prepared_state["token_meta"],
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
        extra = config.extra_params or {}
        skip_render = extra.get("skip_render", False)
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

        return {}

    def cleanup(self):
        """Clean up hook state."""
        if self.hook:
            self.hook.restore()
            self.hook = None

    def get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            # UniEdit params
            "ss_omega": 1.0,
            "slat_omega": 1.0,
            "cfg_interval": (0.5, 1.0),
            "stage2_variant": "latent_replace_union",
            # P2P params
            "inject_stages": ["slat"],  # Only inject in SLAT stage by default
            "slat_t_start": 0.8,
            "slat_t_end": 0.0,
            "p2p_strength": 0.8,  # Slightly lower to let UniEdit dominate
            "patch_coverage_threshold": 0.0,
            "query_chunk": 1024,
            # Output params
            "decode_modes": ["mesh", "gaussian"],
            "skip_render": True,
            "skip_glb": False,
            "skip_ply": False,
        }

