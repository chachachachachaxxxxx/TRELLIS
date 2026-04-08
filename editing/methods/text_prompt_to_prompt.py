from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch

from editing.hooks import PromptToPromptHook, StageConfig
from editing.methods.base import EditMethod, EditMethodConfig, EditMethodInputs, EditMethodOutputs
from editing.utils.text_token_utils import build_text_token_metadata


def normalize_stage_name(name: str) -> str:
    """Normalize stage name to standard format."""
    key = name.strip().lower()
    if key in {"ss", "st", "sparse_structure", "sparse-structure"}:
        return "sparse_structure"
    if key in {"slat", "structured_latent", "structured-latent"}:
        return "slat"
    raise ValueError(f"Unknown stage name: {name}")


class TextPromptToPromptMethod(EditMethod):
    """Text Prompt-to-Prompt editing method.

    Injects source attention for aligned tokens during denoising.
    """

    def __init__(self):
        super().__init__("text_prompt_to_prompt")
        self.hook: PromptToPromptHook | None = None

    def prepare(
        self,
        pipeline,
        inputs: EditMethodInputs,
        config: EditMethodConfig,
    ) -> Dict:
        """Prepare Prompt-to-Prompt editing for text.

        Args:
            pipeline: TRELLIS text-to-3D pipeline
            inputs: Preprocessed inputs (source_prompt, edit_prompt)
            config: Method configuration

        Returns:
            Dictionary with prepared state
        """
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

        # Get prompts
        source_prompt = inputs.extra_inputs.get("source_prompt", "")
        edit_prompt = inputs.extra_inputs.get("edit_prompt", "")

        # Encode conditions
        source_cond_dict = pipeline.get_cond([source_prompt])
        edit_cond_dict = pipeline.get_cond([edit_prompt])

        source_cond = source_cond_dict["cond"]
        edit_cond = edit_cond_dict["cond"]
        neg_cond = source_cond_dict["neg_cond"]

        # Build token metadata
        token_meta = build_text_token_metadata(
            source_cond=source_cond,
            edit_cond=edit_cond,
            source_prompt=source_prompt,
            edit_prompt=edit_prompt,
            tokenizer=pipeline.text_cond_model['tokenizer'],
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
        """Run Prompt-to-Prompt editing for text.

        Args:
            pipeline: TRELLIS text-to-3D pipeline
            prepared_state: State from prepare()
            config: Method configuration

        Returns:
            EditMethodOutputs with results
        """
        import gc

        source_cond_dict = prepared_state["source_cond_dict"]
        edit_cond_dict = prepared_state["edit_cond_dict"]

        # Get sampler params
        ss_params = config.sparse_structure_sampler_params or {}
        slat_params = config.slat_sampler_params or {}

        # Get decode formats (default to mesh only for memory efficiency)
        extra = config.extra_params or {}
        decode_formats = extra.get("decode_formats", ["mesh"])

        # Run source reconstruction if needed
        source_outputs = None
        if not extra.get("skip_source", False):
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
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
            # Decode with memory-efficient format list
            source_outputs = pipeline.decode_slat(source_slat, decode_formats)

            # Clean up intermediate tensors
            del source_coords, source_slat
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Patch models with hook
        if self.hook:
            self.hook.patch_model(pipeline.models["sparse_structure_flow_model"], "sparse_structure")
            self.hook.patch_model(pipeline.models["slat_flow_model"], "slat")

        # Run editing
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        coords = pipeline.sample_sparse_structure(
            edit_cond_dict,
            num_samples=config.num_samples,
            sampler_params=ss_params,
        )

        # Clean up after sparse structure sampling
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        slat = pipeline.sample_slat(
            edit_cond_dict,
            coords,
            sampler_params=slat_params,
        )

        # Clean up coords before decode
        del coords
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Decode with memory-efficient format list
        outputs = pipeline.decode_slat(slat, decode_formats)

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
        from editing.common import save_outputs, write_json

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

        # Save token metadata
        artifact_paths = {}
        if outputs.metadata:
            token_meta_path = out_dir / "token_alignment.json"
            write_json(token_meta_path, outputs.metadata)
            artifact_paths["token_alignment"] = str(token_meta_path)

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
            "query_chunk": 1024,
            "skip_source": False,
            "skip_render": True,  # Skip video rendering by default
            "skip_glb": False,
            "skip_ply": False,
            "decode_formats": ["mesh", "gaussian"],  # Decode mesh and gaussian (skip radiance_field to save memory)
        }
