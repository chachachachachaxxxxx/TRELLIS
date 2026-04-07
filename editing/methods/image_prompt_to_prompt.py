from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image

from editing.common import save_outputs
from editing.hooks import PromptToPromptHook, StageConfig
from editing.methods.base import EditMethod, EditMethodConfig, EditMethodInputs, EditMethodOutputs
from editing.utils import build_image_token_metadata, resolve_patch_size, save_mask_overlay_preview, save_patch_grid_preview


def normalize_stage_name(name: str) -> str:
    """Normalize stage name to standard format.

    Args:
        name: Stage name (ss, st, sparse_structure, slat, etc.)

    Returns:
        Normalized stage name
    """
    key = name.strip().lower()
    if key in {"ss", "st", "sparse_structure", "sparse-structure"}:
        return "sparse_structure"
    if key in {"slat", "structured_latent", "structured-latent"}:
        return "slat"
    raise ValueError(f"Unknown stage name: {name}")


def parse_stage_list(text: str) -> List[str]:
    """Parse comma-separated stage list.

    Args:
        text: Comma-separated stage names

    Returns:
        List of normalized stage names
    """
    if not text.strip():
        return []
    raw = [item.strip() for item in text.split(",") if item.strip()]
    if not raw:
        return []
    if len(raw) == 1 and raw[0].lower() == "none":
        return []
    normalized = []
    for item in raw:
        stage = normalize_stage_name(item)
        if stage not in normalized:
            normalized.append(stage)
    return normalized


class ImagePromptToPromptMethod(EditMethod):
    """Image Prompt-to-Prompt editing method.

    Injects source attention for unmasked tokens during denoising.
    """

    def __init__(self):
        super().__init__("image_prompt_to_prompt")
        self.hook: PromptToPromptHook | None = None

    def prepare(
        self,
        pipeline,
        inputs: EditMethodInputs,
        config: EditMethodConfig,
    ) -> Dict:
        """Prepare Prompt-to-Prompt editing.

        Args:
            pipeline: TRELLIS pipeline
            inputs: Preprocessed inputs
            config: Method configuration

        Returns:
            Dictionary with prepared state
        """
        # Get extra params
        extra = config.extra_params or {}
        inject_stages = extra.get("inject_stages", ["sparse_structure", "slat"])
        patch_coverage_threshold = extra.get("patch_coverage_threshold", 0.0)
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

        # Encode conditions
        source_cond_dict = pipeline.get_cond([inputs.source_image])
        edit_cond_dict = pipeline.get_cond([inputs.edit_image])

        source_cond = source_cond_dict["cond"]
        edit_cond = edit_cond_dict["cond"]
        neg_cond = source_cond_dict["neg_cond"]

        # Build token metadata
        patch_size = resolve_patch_size(pipeline.models["image_cond_model"].patch_size)
        token_meta = build_image_token_metadata(
            cond=edit_cond,
            mask=inputs.mask_image,
            patch_size=patch_size,
            patch_coverage_threshold=patch_coverage_threshold,
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
            "source_cond_dict": source_cond_dict,
            "edit_cond_dict": edit_cond_dict,
            "token_meta": token_meta,
            "stage_configs": stage_configs,
        }

    def run(
        self,
        pipeline,
        prepared_state: Dict,
        config: EditMethodConfig,
    ) -> EditMethodOutputs:
        """Run Prompt-to-Prompt editing.

        Args:
            pipeline: TRELLIS pipeline
            prepared_state: State from prepare()
            config: Method configuration

        Returns:
            EditMethodOutputs with results
        """
        source_cond_dict = prepared_state["source_cond_dict"]
        edit_cond_dict = prepared_state["edit_cond_dict"]

        # Get sampler params
        ss_params = config.sparse_structure_sampler_params or {}
        slat_params = config.slat_sampler_params or {}

        # Run source reconstruction if needed
        source_outputs = None
        extra = config.extra_params or {}
        if not extra.get("skip_source", False):
            torch.manual_seed(config.seed)
            source_coords = pipeline.sample_sparse_structure(
                source_cond_dict,
                num_samples=config.num_samples,
                sampler_params=ss_params,
            )
            source_slat = pipeline.sample_slat(
                source_cond_dict,
                source_coords,
                sampler_params=slat_params,
            )
            source_outputs = pipeline.decode_slat(source_slat, ["mesh", "gaussian", "radiance_field"])
            del source_coords, source_slat

        # Patch models with hook
        if self.hook:
            self.hook.patch_model(pipeline.models["sparse_structure_flow_model"], "sparse_structure")
            self.hook.patch_model(pipeline.models["slat_flow_model"], "slat")

        # Run editing
        torch.manual_seed(config.seed)
        coords = pipeline.sample_sparse_structure(
            edit_cond_dict,
            num_samples=config.num_samples,
            sampler_params=ss_params,
        )
        slat = pipeline.sample_slat(
            edit_cond_dict,
            coords,
            sampler_params=slat_params,
        )
        outputs = pipeline.decode_slat(slat, ["mesh", "gaussian", "radiance_field"])

        return EditMethodOutputs(
            outputs=outputs,
            source_outputs=source_outputs,
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

        # Save token metadata visualizations
        artifact_paths = {}
        if outputs.metadata:
            # Load preprocessed images for visualization
            edit_img = Image.open(out_dir / "edit_preprocessed.png")
            mask_img = Image.open(out_dir / "mask_preprocessed.png")

            save_patch_grid_preview(outputs.metadata, out_dir / "edited_patch_grid.png")
            artifact_paths["patch_grid"] = str(out_dir / "edited_patch_grid.png")

            save_mask_overlay_preview(
                edit_image=edit_img,
                mask_image=mask_img,
                token_meta=outputs.metadata,
                path=out_dir / "mask_token_overlay.png",
            )
            artifact_paths["mask_overlay"] = str(out_dir / "mask_token_overlay.png")

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
            "skip_source": False,
            "skip_render": False,
            "skip_glb": False,
            "skip_ply": False,
        }
