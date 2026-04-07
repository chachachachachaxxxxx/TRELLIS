from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PIL import Image

from editing.common import build_experiment_output_layout, ensure_dir, write_json
from editing.preprocess import prepare_aligned_inputs, save_preprocessed_inputs

from .base import EditMethod, EditMethodConfig, EditMethodInputs, EditMethodOutputs


class EditMethodRunner:
    """Orchestrates the execution of an edit method.

    Handles:
    1. Input preprocessing
    2. Method preparation
    3. Pipeline execution
    4. Output saving
    5. Cleanup
    """

    def __init__(self, method: EditMethod, pipeline):
        self.method = method
        self.pipeline = pipeline

    def run(
        self,
        source_image: Image.Image,
        edit_image: Image.Image,
        mask_image: Optional[Image.Image],
        config: EditMethodConfig,
        case_name: str,
        preprocess: bool = True,
        mask_threshold: int = 128,
        skip_source: bool = False,
        source_voxels_path: Optional[Path] = None,
        source_features_path: Optional[Path] = None,
        mask_glb_path: Optional[Path] = None,
        asset_dir: Optional[Path] = None,
    ) -> EditMethodOutputs:
        """Run the complete edit method pipeline.

        Args:
            source_image: Source image
            edit_image: Edit image
            mask_image: Optional mask image
            config: Method configuration
            case_name: Case name for output organization
            preprocess: Whether to apply preprocessing
            mask_threshold: Mask binarization threshold
            skip_source: Skip source reconstruction
            source_voxels_path: Path to source voxels.ply (for RF inversion)
            source_features_path: Path to source features.npz (for RF inversion)
            mask_glb_path: Path to 3D mask GLB (for UniEdit)
            asset_dir: Asset directory (for RF inversion)

        Returns:
            EditMethodOutputs with results
        """
        # Set up output directories
        output_layout = build_experiment_output_layout(config.method_name, case_name)
        out_dir = ensure_dir(output_layout.edit_dir)
        source_out_dir = output_layout.source_original_dir

        # Preprocess inputs
        if mask_image is None:
            from editing.preprocess import build_blank_mask
            mask_image = build_blank_mask(source_image.size)

        prepared = prepare_aligned_inputs(
            source_image=source_image,
            edit_image=edit_image,
            mask_image=mask_image,
            pipeline=self.pipeline,
            preprocess=preprocess,
            mask_threshold=mask_threshold,
        )

        # Save preprocessing artifacts
        save_preprocessed_inputs(out_dir, prepared, include_mask=True)

        # Build method inputs
        method_inputs = EditMethodInputs(
            source_image=prepared.source,
            edit_image=prepared.edit,
            mask_image=prepared.mask,
            source_voxels_path=source_voxels_path,
            source_features_path=source_features_path,
            mask_glb_path=mask_glb_path,
            asset_dir=asset_dir,
        )

        # Save config
        config_dict = {
            "method_name": config.method_name,
            "case_name": case_name,
            "seed": config.seed,
            "num_samples": config.num_samples,
            "sparse_structure_sampler_params": config.sparse_structure_sampler_params,
            "slat_sampler_params": config.slat_sampler_params,
            "preprocess": preprocess,
            "mask_threshold": mask_threshold,
            "skip_source": skip_source,
            "output_root_dir": str(output_layout.root_dir),
            "output_case_dir": str(output_layout.case_dir),
            "output_edit_dir": str(out_dir),
            "output_source_dir": str(source_out_dir),
        }
        if config.extra_params:
            config_dict.update(config.extra_params)

        write_json(out_dir / "config.json", config_dict)

        try:
            # Prepare method
            prepared_state = self.method.prepare(self.pipeline, method_inputs, config)

            # Run method
            outputs = self.method.run(self.pipeline, prepared_state, config)

            # Save artifacts
            artifact_paths = self.method.save_artifacts(outputs, out_dir, config)

            # Save source reconstruction if requested
            if not skip_source and outputs.source_outputs is not None:
                ensure_dir(source_out_dir)
                self._save_source_outputs(outputs.source_outputs, source_out_dir)

            # Save metadata
            if outputs.metadata:
                write_json(out_dir / "method_metadata.json", outputs.metadata)

            print(f"✓ Edit results saved to: {out_dir}")
            if not skip_source and outputs.source_outputs is not None:
                print(f"✓ Source results saved to: {source_out_dir}")

            return outputs

        finally:
            # Always cleanup
            self.method.cleanup()

    def _save_source_outputs(self, source_outputs, out_dir: Path):
        """Save source reconstruction outputs."""
        from editing.common import save_outputs

        save_outputs(
            outputs=source_outputs,
            out_dir=out_dir,
            skip_render=False,
            skip_glb=False,
            skip_ply=False,
        )
