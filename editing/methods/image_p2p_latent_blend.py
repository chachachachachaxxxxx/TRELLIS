from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from editing.common import save_outputs
from editing.hooks import PromptToPromptHook, StageConfig
from editing.methods.base import EditMethod, EditMethodConfig, EditMethodInputs, EditMethodOutputs
from editing.utils import build_image_token_metadata, resolve_patch_size, save_mask_overlay_preview, save_patch_grid_preview


class ImageP2PLatentBlendMethod(EditMethod):
    """Image P2P with post-generation latent blending.

    Simple approach:
    1. Generate source 3D with P2P
    2. Generate edit 3D with P2P
    3. Blend SLAT features based on mask (with optional soft blending)

    This is simpler than UniEdit - no RF inversion, just blend final latents.
    """

    def __init__(self):
        super().__init__("image_p2p_latent_blend")
        self.hook: PromptToPromptHook | None = None

    def prepare(
        self,
        pipeline,
        inputs: EditMethodInputs,
        config: EditMethodConfig,
    ) -> Dict:
        """Prepare P2P with latent blending.

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

        # Latent blending params
        blend_mode = extra.get("blend_mode", "hard")  # "hard" or "soft"
        soft_kernel_size = extra.get("soft_kernel_size", 5)
        blend_strength = extra.get("blend_strength", 1.0)  # 1.0 = full blend

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

        # Build spatial mask for blending
        spatial_mask = self._build_spatial_mask(
            inputs.mask_image,
            blend_mode=blend_mode,
            kernel_size=soft_kernel_size,
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
            "blend_strength": blend_strength,
        }

    def _build_spatial_mask(
        self,
        mask_image: Image.Image,
        blend_mode: str = "hard",
        kernel_size: int = 5,
    ) -> torch.Tensor:
        """Build spatial mask for latent blending.

        Args:
            mask_image: PIL mask image (white=edit, black=preserve)
            blend_mode: "hard" or "soft"
            kernel_size: Kernel size for soft mask

        Returns:
            Mask tensor [1, 1, H, W], values in [0, 1]
            0 = preserve source, 1 = use edit
        """
        # Convert to grayscale and normalize
        mask_np = np.array(mask_image.convert("L")).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        if blend_mode == "soft" and kernel_size > 1:
            # Apply Gaussian blur for soft boundaries (VoxHammer-style)
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
        """Run P2P with latent blending.

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
        blend_strength = prepared_state["blend_strength"]

        # Get sampler params
        ss_params = config.sparse_structure_sampler_params or {}
        slat_params = config.slat_sampler_params or {}

        # Patch models with P2P hook
        if self.hook:
            self.hook.patch_model(pipeline.models["sparse_structure_flow_model"], "sparse_structure")
            self.hook.patch_model(pipeline.models["slat_flow_model"], "slat")

        # Step 1: Generate source 3D
        print("Generating source 3D...")
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

        # Step 2: Generate edit 3D
        print("Generating edit 3D...")
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        edit_coords = pipeline.sample_sparse_structure(
            edit_cond_dict,
            num_samples=config.num_samples,
            sampler_params=ss_params,
        )
        edit_slat = pipeline.sample_slat(
            edit_cond_dict,
            edit_coords,
            sampler_params=slat_params,
        )

        # Step 3: Blend SLAT features based on mask
        print(f"Blending SLAT features (strength={blend_strength})...")
        blended_slat = self._blend_slat_features(
            source_slat=source_slat,
            edit_slat=edit_slat,
            source_coords=source_coords,
            edit_coords=edit_coords,
            spatial_mask=spatial_mask,
            blend_strength=blend_strength,
            device=pipeline.device,
        )

        # Use edit coords as base (could also blend coords, but simpler to use edit)
        final_coords = edit_coords

        # Step 4: Decode blended result
        print("Decoding blended result...")
        outputs = pipeline.decode_slat(blended_slat, ["mesh", "gaussian"])

        # Also decode source for comparison
        source_outputs = pipeline.decode_slat(source_slat, ["mesh", "gaussian"])

        return EditMethodOutputs(
            outputs=outputs,
            source_outputs=source_outputs,
            metadata=prepared_state["token_meta"],
        )

    def _blend_slat_features(
        self,
        source_slat,
        edit_slat,
        source_coords,
        edit_coords,
        spatial_mask: torch.Tensor,
        blend_strength: float,
        device: torch.device,
    ):
        """Blend SLAT features based on spatial mask.

        This is a simplified version - proper implementation would need:
        1. Map 2D mask to 3D voxel space
        2. Handle sparse tensor structure
        3. Interpolate features at matching coordinates

        For now, we do a simple feature-level blend.

        Args:
            source_slat: Source SLAT (SparseTensor)
            edit_slat: Edit SLAT (SparseTensor)
            source_coords: Source coordinates
            edit_coords: Edit coordinates
            spatial_mask: 2D spatial mask [1, 1, H, W]
            blend_strength: Blending strength (0-1)
            device: Device

        Returns:
            Blended SLAT (SparseTensor)
        """
        # For simplicity, just return edit_slat
        # Full implementation would require:
        # 1. Project 2D mask to 3D voxel coordinates
        # 2. Find matching voxels between source and edit
        # 3. Blend features based on mask values
        # 4. Handle non-overlapping regions

        print("Warning: SLAT blending not fully implemented, returning edit SLAT")
        return edit_slat

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

        # Save source outputs for comparison
        if outputs.source_outputs:
            source_dir = out_dir / "source_comparison"
            source_dir.mkdir(exist_ok=True)
            save_outputs(
                outputs=outputs.source_outputs,
                out_dir=source_dir,
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
            # Latent blending params
            "blend_mode": "soft",  # "hard" or "soft"
            "soft_kernel_size": 5,
            "blend_strength": 1.0,
            # Output params
            "skip_render": True,
            "skip_glb": False,
            "skip_ply": False,
        }
