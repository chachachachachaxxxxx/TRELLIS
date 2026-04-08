from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch

from editing.hooks import PromptToPromptHook, StageConfig
from editing.inversion import denoise_slat, denoise_sparse_structure, invert_slat, invert_sparse_structure
from editing.methods.base import EditMethod, EditMethodConfig, EditMethodInputs, EditMethodOutputs
from editing.preprocess.asset_3d import ply_to_coords, project_sparse_terminal_noise, feats_to_slat, coords_to_voxel


class ImagePromptToPromptRFInversionMethod(EditMethod):
    """Image Prompt-to-Prompt with RF Inversion initialization.

    Uses RF inversion to initialize from source assets, then applies
    Prompt-to-Prompt attention injection during denoising.
    """

    def __init__(self):
        super().__init__("image_prompt_to_prompt_rf_inversion")
        self.hook: PromptToPromptHook | None = None

    def prepare(
        self,
        pipeline,
        inputs: EditMethodInputs,
        config: EditMethodConfig,
    ) -> Dict:
        """Prepare RF inversion + Prompt-to-Prompt.

        Args:
            pipeline: TRELLIS image-to-3D pipeline
            inputs: Preprocessed inputs with source assets
            config: Method configuration

        Returns:
            Dictionary with prepared state
        """
        import gc

        # Validate inputs
        if inputs.source_voxels_path is None or inputs.source_features_path is None:
            raise RuntimeError("RF inversion requires source_voxels_path and source_features_path")

        # Get extra params
        extra = config.extra_params or {}
        inject_stages = extra.get("inject_stages", ["sparse_structure", "slat"])
        query_chunk = extra.get("query_chunk", 1024)

        # Build stage configs
        stage_configs = {}
        for stage_name in ["sparse_structure", "slat"]:
            enabled = stage_name in inject_stages
            t_start = extra.get(f"{stage_name}_t_start", 1.0)
            t_end = extra.get(f"{stage_name}_t_end", 0.0)
            strength = extra.get(f"{stage_name}_strength", 1.0)
            stage_configs[stage_name] = StageConfig(
                name=stage_name,
                enabled=enabled,
                t_start=t_start,
                t_end=t_end,
                strength=strength,
            )

        # Load source assets
        # Get resolution from pipeline params (default 64)
        resolution = pipeline.sparse_structure_sampler_params.get("grid_size", 64)
        source_coords = ply_to_coords(inputs.source_voxels_path, pipeline.device, resolution)

        # Clean up after loading coords
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        from trellis.modules import sparse as sp
        source_slat = feats_to_slat(pipeline, inputs.source_features_path, sp.SparseTensor)

        # Clean up after loading SLAT
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Encode conditions
        source_cond_dict = pipeline.get_cond([inputs.source_image])
        edit_cond_dict = pipeline.get_cond([inputs.edit_image])

        source_cond = source_cond_dict["cond"]
        edit_cond = edit_cond_dict["cond"]
        neg_cond = source_cond_dict["neg_cond"]

        # Clean up after encoding
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Build token metadata (reuse from image_prompt_to_prompt)
        from editing.utils.token_utils import build_image_token_metadata, resolve_patch_size

        patch_size = resolve_patch_size(pipeline.models["image_cond_model"].patch_size)
        token_meta = build_image_token_metadata(
            cond=edit_cond,
            mask=inputs.mask_image,
            patch_size=patch_size,
            patch_coverage_threshold=extra.get("patch_coverage_threshold", 0.0),
        )

        # Create hook
        self.hook = PromptToPromptHook(
            source_cond=source_cond,
            edit_cond=edit_cond,
            neg_cond=neg_cond,
            token_meta=token_meta,
            stage_configs=stage_configs,
            query_chunk=query_chunk,
        )

        return {
            "source_coords": source_coords,
            "source_slat": source_slat,
            "source_cond_dict": source_cond_dict,
            "edit_cond_dict": edit_cond_dict,
            "token_meta": token_meta,
            "stage_configs": stage_configs,
            "resolution": resolution,
        }

    def run(
        self,
        pipeline,
        prepared_state: Dict,
        config: EditMethodConfig,
    ) -> EditMethodOutputs:
        """Run RF inversion + Prompt-to-Prompt.

        Args:
            pipeline: TRELLIS image-to-3D pipeline
            prepared_state: State from prepare()
            config: Method configuration

        Returns:
            EditMethodOutputs with results
        """
        import gc

        source_coords = prepared_state["source_coords"]
        source_slat = prepared_state["source_slat"]
        source_cond_dict = prepared_state["source_cond_dict"]
        edit_cond_dict = prepared_state["edit_cond_dict"]
        resolution = prepared_state["resolution"]

        # Get sampler params
        ss_params = config.sparse_structure_sampler_params or pipeline.sparse_structure_sampler_params
        slat_params = config.slat_sampler_params or pipeline.slat_sampler_params

        extra = config.extra_params or {}
        cfg_interval = (extra.get("cfg_interval_start", 0.0), extra.get("cfg_interval_end", 1.0))

        # Get decode formats (default to mesh only for memory efficiency)
        decode_formats = extra.get("decode_formats", ["mesh"])

        # Step 1: Invert sparse structure
        # Convert coords to voxel tensor
        source_voxel = coords_to_voxel(source_coords, pipeline.device, resolution)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        ss_terminal_noise = invert_sparse_structure(
            pipeline=pipeline,
            cond_src=source_cond_dict,
            voxel_src=source_voxel,
            params=ss_params,
            cfg_interval=cfg_interval,
            verbose=False,
        )

        # Clean up after sparse structure inversion
        del source_voxel
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step 2: Invert SLAT
        slat_terminal_noise = invert_slat(
            pipeline=pipeline,
            cond_src=source_cond_dict,
            slat_src=source_slat,
            params=slat_params,
            cfg_interval=cfg_interval,
            verbose=False,
        )

        # Clean up source assets
        del source_slat
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Patch models with hook
        if self.hook:
            self.hook.patch_model(pipeline.models["sparse_structure_flow_model"], "sparse_structure")
            self.hook.patch_model(pipeline.models["slat_flow_model"], "slat")

        # Step 3: Denoise sparse structure with P2P
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        coords = denoise_sparse_structure(
            pipeline=pipeline,
            cond_edit=edit_cond_dict,
            terminal_noise=ss_terminal_noise,
            params=ss_params,
            cfg_interval=cfg_interval,
            verbose=False,
        )

        # Clean up after sparse structure denoising
        del ss_terminal_noise
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step 4: Project SLAT terminal noise to new coordinates
        from trellis.modules import sparse as sp
        projected_noise = project_sparse_terminal_noise(
            slat_terminal_noise, coords, coords.device, sp.SparseTensor, resolution
        )

        # Clean up original terminal noise
        del slat_terminal_noise
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step 5: Denoise SLAT with P2P
        slat = denoise_slat(
            pipeline=pipeline,
            cond_edit=edit_cond_dict,
            terminal_noise=projected_noise,
            params=slat_params,
            cfg_interval=cfg_interval,
            verbose=False,
        )

        # Clean up before decode
        del projected_noise, coords
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Decode with memory-efficient format list
        outputs = pipeline.decode_slat(slat, decode_formats)

        return EditMethodOutputs(
            outputs=outputs,
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

        # Save patch metadata
        artifact_paths = {}
        if outputs.metadata:
            patch_meta_path = out_dir / "patch_metadata.json"
            write_json(patch_meta_path, outputs.metadata)
            artifact_paths["patch_metadata"] = str(patch_meta_path)

        return artifact_paths

    def cleanup(self):
        """Clean up hook state."""
        if self.hook:
            self.hook.restore()
            self.hook = None

    def get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "inject_stages": ["sparse_structure", "slat"],
            "sparse_structure_t_start": 1.0,
            "sparse_structure_t_end": 0.3,
            "sparse_structure_strength": 1.0,
            "slat_t_start": 0.8,
            "slat_t_end": 0.0,
            "slat_strength": 1.0,
            "patch_coverage_threshold": 0.0,
            "query_chunk": 1024,
            "cfg_interval_start": 0.0,
            "cfg_interval_end": 1.0,
            "skip_render": True,
            "skip_glb": False,
            "skip_ply": False,
            "decode_formats": ["mesh", "gaussian"],  # Decode mesh and gaussian (skip radiance_field to save memory)
            # Reduce steps for memory efficiency (RF inversion is memory-intensive)
            "sparse_structure_steps": 12,  # Reduced from default 25
            "slat_steps": 12,  # Reduced from default 25
        }
