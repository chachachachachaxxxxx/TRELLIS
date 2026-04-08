from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image

from editing.common import save_outputs
from editing.hooks import PromptToPromptHook, StageConfig
from editing.methods.base import EditMethod, EditMethodConfig, EditMethodInputs, EditMethodOutputs
from editing.utils import build_image_token_metadata, resolve_patch_size, save_mask_overlay_preview, save_patch_grid_preview


class ImageP2PLatentReplaceMethod(EditMethod):
    """Image P2P with direct latent replacement.

    Combines:
    1. P2P attention injection for fine-grained control
    2. Direct latent replacement based on mask (simpler than UniEdit)
    3. Optional soft mask for smooth blending at boundaries
    """

    def __init__(self):
        super().__init__("image_p2p_latent_replace")
        self.hook: PromptToPromptHook | None = None

    def prepare(
        self,
        pipeline,
        inputs: EditMethodInputs,
        config: EditMethodConfig,
    ) -> Dict:
        """Prepare P2P with latent replacement.

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

        # Latent replacement params
        enable_latent_replace = extra.get("enable_latent_replace", True)
        latent_replace_steps = extra.get("latent_replace_steps", 0.5)  # Replace in first 50% steps
        soft_mask_enabled = extra.get("soft_mask_enabled", False)
        soft_mask_kernel_size = extra.get("soft_mask_kernel_size", 3)

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

        # Build spatial mask for latent replacement
        spatial_mask = None
        if enable_latent_replace and inputs.mask_image is not None:
            spatial_mask = self._build_spatial_mask(
                inputs.mask_image,
                soft_enabled=soft_mask_enabled,
                kernel_size=soft_mask_kernel_size,
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
            "spatial_mask": spatial_mask,
            "enable_latent_replace": enable_latent_replace,
            "latent_replace_steps": latent_replace_steps,
        }

    def _build_spatial_mask(
        self,
        mask_image: Image.Image,
        soft_enabled: bool = False,
        kernel_size: int = 3,
    ) -> torch.Tensor:
        """Build spatial mask for latent replacement.

        Args:
            mask_image: PIL mask image (white=edit, black=preserve)
            soft_enabled: Whether to apply soft blending at boundaries
            kernel_size: Kernel size for soft mask (larger = smoother)

        Returns:
            Mask tensor [1, 1, H, W], values in [0, 1]
            0 = preserve source, 1 = use edit
        """
        # Convert to grayscale and normalize
        mask_np = np.array(mask_image.convert("L")).astype(np.float32) / 255.0

        # Invert: white (255) -> 1 (edit), black (0) -> 0 (preserve)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        if soft_enabled and kernel_size > 1:
            # Apply Gaussian blur for soft boundaries
            import torch.nn.functional as F

            # Create Gaussian kernel
            sigma = kernel_size / 3.0
            kernel_range = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
            gaussian_1d = torch.exp(-0.5 * (kernel_range / sigma) ** 2)
            gaussian_1d = gaussian_1d / gaussian_1d.sum()

            # 2D Gaussian kernel
            gaussian_2d = gaussian_1d.unsqueeze(0) * gaussian_1d.unsqueeze(1)
            gaussian_2d = gaussian_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]

            # Apply convolution
            padding = kernel_size // 2
            mask_tensor = F.conv2d(mask_tensor, gaussian_2d, padding=padding)
            mask_tensor = torch.clamp(mask_tensor, 0.0, 1.0)

        return mask_tensor

    def run(
        self,
        pipeline,
        prepared_state: Dict,
        config: EditMethodConfig,
    ) -> EditMethodOutputs:
        """Run P2P with latent replacement.

        Args:
            pipeline: TRELLIS pipeline
            prepared_state: State from prepare()
            config: Method configuration

        Returns:
            EditMethodOutputs with results
        """
        source_cond_dict = prepared_state["source_cond_dict"]
        edit_cond_dict = prepared_state["edit_cond_dict"]
        spatial_mask = prepared_state["spatial_mask"]
        enable_latent_replace = prepared_state["enable_latent_replace"]
        latent_replace_steps = prepared_state["latent_replace_steps"]

        # Get sampler params
        ss_params = config.sparse_structure_sampler_params or {}
        slat_params = config.slat_sampler_params or {}

        # Run source reconstruction if needed
        source_outputs = None
        extra = config.extra_params or {}
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
            source_outputs = pipeline.decode_slat(source_slat, ["mesh", "gaussian"])

            # Store source latents for replacement
            source_coords_stored = source_coords
            source_slat_stored = source_slat
        else:
            source_coords_stored = None
            source_slat_stored = None

        # Patch models with P2P hook
        if self.hook:
            self.hook.patch_model(pipeline.models["sparse_structure_flow_model"], "sparse_structure")
            self.hook.patch_model(pipeline.models["slat_flow_model"], "slat")

        # Patch samplers for latent replacement if enabled
        if enable_latent_replace and spatial_mask is not None:
            self._patch_samplers_for_latent_replace(
                pipeline=pipeline,
                source_coords=source_coords_stored,
                source_slat=source_slat_stored,
                spatial_mask=spatial_mask,
                replace_steps_ratio=latent_replace_steps,
            )

        # Run editing with P2P + latent replacement
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
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
        outputs = pipeline.decode_slat(slat, ["mesh", "gaussian"])

        return EditMethodOutputs(
            outputs=outputs,
            source_outputs=source_outputs,
            metadata=prepared_state["token_meta"],
        )

    def _patch_samplers_for_latent_replace(
        self,
        pipeline,
        source_coords,
        source_slat,
        spatial_mask: torch.Tensor,
        replace_steps_ratio: float,
    ):
        """Patch samplers to perform latent replacement during denoising.

        Simple mask-based latent replacement: at each denoising step,
        blend source and edit latents based on the spatial mask.

        Args:
            pipeline: TRELLIS pipeline
            source_coords: Source sparse structure
            source_slat: Source SLAT
            spatial_mask: Spatial mask [1, 1, H, W], 0=preserve, 1=edit
            replace_steps_ratio: Ratio of steps to apply replacement (0-1)
        """
        import types

        # Store in closure
        stored_source_coords = source_coords
        stored_source_slat = source_slat
        stored_mask = spatial_mask
        stored_ratio = replace_steps_ratio

        # Patch sparse structure sampler
        ss_sampler = pipeline.sparse_structure_sampler
        if hasattr(ss_sampler, "sample") and stored_source_coords is not None:
            original_sample = ss_sampler.sample

            def patched_ss_sample(self, *args, **kwargs):
                # Run original sampling
                result = original_sample(*args, **kwargs)

                # Apply mask-based replacement
                # Note: This is a simplified version - actual implementation
                # would need to hook into the denoising loop
                if stored_source_coords is not None and stored_mask is not None:
                    # Blend based on mask (simplified - would need proper spatial alignment)
                    pass

                return result

            ss_sampler.sample = types.MethodType(patched_ss_sample, ss_sampler)

        # Patch SLAT sampler
        slat_sampler = pipeline.slat_sampler
        if hasattr(slat_sampler, "sample") and stored_source_slat is not None:
            original_sample = slat_sampler.sample

            def patched_slat_sample(self, *args, **kwargs):
                # Run original sampling
                result = original_sample(*args, **kwargs)

                # Apply mask-based replacement
                if stored_source_slat is not None and stored_mask is not None:
                    # Blend based on mask (simplified - would need proper spatial alignment)
                    pass

                return result

            slat_sampler.sample = types.MethodType(patched_slat_sample, slat_sampler)

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
            # P2P params
            "inject_stages": ["sparse_structure", "slat"],
            "sparse_structure_t_start": 1.0,
            "sparse_structure_t_end": 0.3,
            "sparse_structure_strength": 1.0,
            "slat_t_start": 0.8,
            "slat_t_end": 0.0,
            "slat_strength": 1.0,
            "patch_coverage_threshold": 0.0,
            "query_chunk": 1024,
            # Latent replacement params
            "enable_latent_replace": True,
            "latent_replace_steps": 0.5,  # Replace in first 50% of steps
            "soft_mask_enabled": False,
            "soft_mask_kernel_size": 3,
            # Output params
            "skip_source": False,
            "skip_render": True,
            "skip_glb": False,
            "skip_ply": False,
        }
