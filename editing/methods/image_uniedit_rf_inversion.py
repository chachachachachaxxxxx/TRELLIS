from __future__ import annotations

from pathlib import Path
from typing import Dict

from editing.methods.base import EditMethod, EditMethodConfig, EditMethodInputs, EditMethodOutputs


class ImageUniEditRFInversionMethod(EditMethod):
    """UniEdit-style RF Inversion editing method.

    Uses RF inversion with two-stage voxel editing:
    1. Stage 1: Edit voxel structure with mask guidance
    2. Stage 2: Edit SLAT features

    TODO: This is a placeholder implementation. Full UniEdit logic needs to be implemented.
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

        # TODO: Implement full UniEdit preparation logic
        raise NotImplementedError(
            "UniEdit RF inversion method is not fully implemented yet. "
            "This requires complex two-stage editing logic with mask guidance. "
            "Please use the script version: example_image_uniedit_rf_inversion.py"
        )

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
        raise NotImplementedError("See prepare() for details")

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

        return {}

    def get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "skip_render": True,
            "skip_glb": False,
            "skip_ply": False,
        }
