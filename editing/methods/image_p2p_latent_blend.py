"""Image P2P with per-step latent blending (VoxHammer-style).

This method implements latent blending at each denoising step:
1. SS stage: Blend dense latents [1, 8, 16, 16, 16] based on 3D mask
2. SLAT stage: Blend sparse features based on coordinate matching

Key features:
- Requires inversion: "simple" (Euler) or "rf_solver" (VoxHammer)
- P2P attention injection + latent blending
- Supports ablation studies on inversion quality
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import types

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from editing.common import save_outputs
from editing.hooks import PromptToPromptHook, StageConfig
from editing.methods.base import EditMethod, EditMethodConfig, EditMethodInputs, EditMethodOutputs
from editing.preprocess.asset_3d import feats_to_slat
from editing.utils import (
    build_image_token_metadata,
    resolve_patch_size,
    save_mask_overlay_preview,
    save_patch_grid_preview,
)


class ImageP2PLatentBlendMethod(EditMethod):
    """Image P2P with per-step latent blending.

    Implements VoxHammer-style latent blending:
    - SS stage: Blend dense latents at each denoising step
    - SLAT stage: Blend sparse features at each denoising step

    Inversion modes (ablation):
    - "simple": Simple Euler inversion (fast)
    - "rf_solver": RF-Solver inversion (accurate, from VoxHammer)
    """

    def __init__(self):
        super().__init__("image_p2p_latent_blend")
        self.hook: Optional[PromptToPromptHook] = None

    def prepare(
        self,
        pipeline,
        inputs: EditMethodInputs,
        config: EditMethodConfig,
    ) -> Dict:
        """Prepare method state."""

        # Check inputs
        if not inputs.asset_dir:
            raise ValueError(
                "image_p2p_latent_blend requires asset_dir. "
                "Use --source-model or preprocess with --preprocess."
            )

        if not inputs.mask_image:
            raise ValueError(
                "image_p2p_latent_blend requires mask_image for P2P token metadata."
            )

        if not inputs.mask_glb_path:
            raise ValueError(
                "image_p2p_latent_blend requires mask_glb_path for 3D latent blending."
            )

        # Get config
        extra = config.extra_params or {}

        # P2P config
        inject_stages = extra.get("inject_stages", ["sparse_structure", "slat"])
        ss_t_start = extra.get("sparse_structure_t_start", 1.0)
        ss_t_end = extra.get("sparse_structure_t_end", 0.3)
        ss_strength = extra.get("sparse_structure_strength", 1.0)
        slat_t_start = extra.get("slat_t_start", 0.8)
        slat_t_end = extra.get("slat_t_end", 0.0)
        slat_strength = extra.get("slat_strength", 1.0)

        # Blending config
        blend_ss_enabled = extra.get("blend_ss_enabled", True)
        blend_slat_enabled = extra.get("blend_slat_enabled", True)
        blend_strength = extra.get("blend_strength", 1.0)

        # Inversion config (ablation)
        inversion_mode = extra.get("inversion_mode", "simple")  # "simple", "rf_solver"

        # Mask config
        ss_blend_mode = extra.get("ss_blend_mode", "hard")
        slat_blend_mode = extra.get("slat_blend_mode", "hard")

        # Load source SLAT
        print("Loading source SLAT features...")
        from trellis.modules.sparse.basic import SparseTensor
        source_slat = feats_to_slat(
            pipeline=pipeline,
            feats_path=inputs.asset_dir / "features.npz",
            SparseTensor=SparseTensor,
        )
        print(f"Source SLAT: {source_slat.coords.shape[0]} voxels")

        # Debug: Check source coordinate range
        source_xyz = source_slat.coords[:, 1:]
        print(f"Source coord range: x=[{source_xyz[:, 0].min()}, {source_xyz[:, 0].max()}], "
              f"y=[{source_xyz[:, 1].min()}, {source_xyz[:, 1].max()}], "
              f"z=[{source_xyz[:, 2].min()}, {source_xyz[:, 2].max()}]")

        # Load 3D mask from GLB
        resolution = 64
        print(f"Loading 3D mask from GLB: {inputs.mask_glb_path}")
        from editing.preprocess.asset_3d import load_mask_glb_coords
        mask_result = load_mask_glb_coords(
            mask_glb=str(inputs.mask_glb_path),
            device=pipeline.device,
            resolution=resolution,
            asset_dir=inputs.asset_dir,
        )
        mask_coords = mask_result.coords
        print(f"Mask coords: {mask_coords.shape[0]} voxels")

        # Debug: Check mask coordinate range
        if mask_coords.shape[0] > 0:
            if mask_coords.shape[1] == 4:
                mask_xyz = mask_coords[:, 1:]
            else:
                mask_xyz = mask_coords
            print(f"Mask coord range: x=[{mask_xyz[:, 0].min()}, {mask_xyz[:, 0].max()}], "
                  f"y=[{mask_xyz[:, 1].min()}, {mask_xyz[:, 1].max()}], "
                  f"z=[{mask_xyz[:, 2].min()}, {mask_xyz[:, 2].max()}]")

        # Encode conditions
        source_cond_dict = pipeline.get_cond([inputs.source_image])
        edit_cond_dict = pipeline.get_cond([inputs.edit_image])

        # Build token metadata from 2D mask (for P2P)
        patch_size = resolve_patch_size(inputs.source_image.size)
        patch_coverage_threshold = extra.get("patch_coverage_threshold", 0.0)
        token_meta = build_image_token_metadata(
            cond=edit_cond_dict["cond"],
            mask=inputs.mask_image,
            patch_size=patch_size,
            patch_coverage_threshold=patch_coverage_threshold,
        )

        # Build 2D spatial mask for SLAT preserve coords (fallback)
        slat_spatial_mask_2d = self._build_spatial_mask(
            inputs.mask_image,
            blend_mode=slat_blend_mode,
            kernel_size=extra.get("slat_soft_kernel_size", 5),
        )

        # Build P2P stage configs
        stage_configs = {}
        if "sparse_structure" in inject_stages:
            stage_configs["sparse_structure"] = StageConfig(
                name="sparse_structure",
                enabled=True,
                t_start=ss_t_start,
                t_end=ss_t_end,
                strength=ss_strength,
            )
        if "slat" in inject_stages:
            stage_configs["slat"] = StageConfig(
                name="slat",
                enabled=True,
                t_start=slat_t_start,
                t_end=slat_t_end,
                strength=slat_strength,
            )

        # Create P2P hook
        query_chunk = extra.get("query_chunk", 1024)
        self.hook = PromptToPromptHook(
            source_cond=source_cond_dict["cond"],
            edit_cond=edit_cond_dict["cond"],
            neg_cond=edit_cond_dict["neg_cond"],
            token_meta=token_meta,
            stage_configs=stage_configs,
            query_chunk=query_chunk,
        )

        return {
            "source_slat": source_slat,
            "source_cond_dict": source_cond_dict,
            "edit_cond_dict": edit_cond_dict,
            "token_meta": token_meta,
            "stage_configs": stage_configs,
            "mask_coords": mask_coords,
            "slat_spatial_mask_2d": slat_spatial_mask_2d,
            "blend_ss_enabled": blend_ss_enabled,
            "blend_slat_enabled": blend_slat_enabled,
            "blend_strength": blend_strength,
            "inversion_mode": inversion_mode,
            "resolution": resolution,
        }

    def _build_spatial_mask(
        self,
        mask_image: Image.Image,
        blend_mode: str = "hard",
        kernel_size: int = 5,
    ) -> torch.Tensor:
        """Build spatial mask for latent blending."""
        mask_np = np.array(mask_image.convert("L")).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)

        if blend_mode == "soft" and kernel_size > 1:
            sigma = kernel_size / 3.0
            kernel_range = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
            gaussian_1d = torch.exp(-0.5 * (kernel_range / sigma) ** 2)
            gaussian_1d = gaussian_1d / gaussian_1d.sum()
            gaussian_2d = gaussian_1d.unsqueeze(0) * gaussian_1d.unsqueeze(1)
            gaussian_2d = gaussian_2d.unsqueeze(0).unsqueeze(0)
            padding = kernel_size // 2
            mask_tensor = F.conv2d(mask_tensor, gaussian_2d, padding=padding)
            mask_tensor = torch.clamp(mask_tensor, 0.0, 1.0)

        return mask_tensor

    def _prepare_ss_source(
        self,
        pipeline,
        source_cond_dict,
        inversion_mode: str,
        config: EditMethodConfig,
    ) -> Dict[str, torch.Tensor]:
        """Prepare source SS latent cache via inversion.

        Args:
            inversion_mode: "simple" (Euler) or "rf_solver" (VoxHammer)

        Returns:
            Dict mapping timestep to latent: {f"{t}": latent_tensor}
        """
        if inversion_mode == "none":
            raise ValueError(
                "inversion_mode='none' is not valid. "
                "Must do inversion to get per-step latent cache. "
                "Use 'simple' or 'rf_solver'."
            )

        # Generate source coords
        print("Generating source SS coords...")
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        source_coords = pipeline.sample_sparse_structure(
            source_cond_dict,
            num_samples=config.num_samples,
            sampler_params=config.sparse_structure_sampler_params or {},
        )

        # Convert to voxel
        voxel = torch.zeros(1, 1, 64, 64, 64, dtype=torch.float32, device=source_coords.device)
        # source_coords shape: [N, 4] where columns are [batch, x, y, z]
        coords_int = source_coords[:, 1:].long()  # Take [x, y, z], shape [N, 3]
        voxel[0, 0, coords_int[:, 0], coords_int[:, 1], coords_int[:, 2]] = 1.0

        # Encode to latent
        encoder = pipeline.models["sparse_structure_encoder"]
        z_s = encoder(voxel)

        if inversion_mode == "simple":
            # Simple Euler inversion
            print("Running simple Euler inversion...")
            latent_cache = self._simple_euler_inversion(
                pipeline,
                z_s,
                source_cond_dict,
                config,
            )
            return latent_cache

        elif inversion_mode == "rf_solver":
            # RF-Solver inversion (VoxHammer)
            print("Running RF-Solver inversion...")
            latent_cache = self._rf_solver_inversion(
                pipeline,
                z_s,
                source_cond_dict,
                config,
            )
            return latent_cache

        else:
            raise ValueError(f"Unknown inversion_mode: {inversion_mode}")

    def _simple_euler_inversion(
        self,
        pipeline,
        z_s: torch.Tensor,
        cond_dict,
        config: EditMethodConfig,
    ) -> Dict[str, torch.Tensor]:
        """Simple first-order Euler inversion.

        Inverts from data (t=0) to noise (t=1), caching latents at each step.
        """
        model = pipeline.models["sparse_structure_flow_model"]
        sigma_min = pipeline.sparse_structure_sampler.sigma_min

        sampler_params = config.sparse_structure_sampler_params or {}
        steps = sampler_params.get("steps", 25)
        rescale_t = 3.0

        # Time sequence: 0 -> 1 (data to noise)
        t_seq = np.linspace(0, 1, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)

        latent_cache = {}
        sample = z_s

        # Disable gradients for inversion
        with torch.inference_mode():
            for i in range(steps):
                t_curr = t_seq[i]
                t_next = t_seq[i + 1]

                # Cache current latent (move to CPU to save GPU memory)
                latent_cache[f"{t_next}"] = sample.cpu()

                # Euler step: x_{t+dt} = x_t + dt * f(x_t, t)
                t_tensor = torch.tensor([1000 * t_curr], device=sample.device, dtype=torch.float32)
                pred_v = model(sample, t_tensor, cond_dict["cond"])
                sample = sample + (t_next - t_curr) * pred_v

        return latent_cache

    def _rf_solver_inversion(
        self,
        pipeline,
        z_s: torch.Tensor,
        cond_dict,
        config: EditMethodConfig,
    ) -> Dict[str, torch.Tensor]:
        """RF-Solver inversion (second-order Taylor).

        Implements VoxHammer's inversion scheme (Eq. 1-2).
        """
        model = pipeline.models["sparse_structure_flow_model"]
        sigma_min = pipeline.sparse_structure_sampler.sigma_min

        sampler_params = config.sparse_structure_sampler_params or {}
        steps = sampler_params.get("steps", 25)
        rescale_t = 3.0

        # Time sequence: 0 -> 1 (data to noise)
        t_seq = np.linspace(0, 1, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)

        latent_cache = {}
        sample = z_s

        # Disable gradients for inversion
        with torch.inference_mode():
            for i in range(steps):
                t_curr = t_seq[i]
                t_next = t_seq[i + 1]
                dt = t_next - t_curr

                # Cache current latent (move to CPU to save GPU memory)
                latent_cache[f"{t_next}"] = sample.cpu()

                # First prediction at current point
                t_tensor = torch.tensor([1000 * t_curr], device=sample.device, dtype=torch.float32)
                pred_v = model(sample, t_tensor, cond_dict["cond"])

                # Midpoint
                sample_mid = sample + (dt / 2) * pred_v
                t_mid = t_curr + dt / 2
                t_mid_tensor = torch.tensor([1000 * t_mid], device=sample.device, dtype=torch.float32)
                pred_v_mid = model(sample_mid, t_mid_tensor, cond_dict["cond"])

                # Second-order update
                first_order = (pred_v_mid - pred_v) / (dt / 2)
                sample = sample + dt * pred_v + 0.5 * (dt ** 2) * first_order

                # Clear cache every few steps
                if i % 5 == 0:
                    torch.cuda.empty_cache()

        return latent_cache

    def _prepare_slat_source(
        self,
        pipeline,
        source_slat,
        source_cond_dict,
        inversion_mode: str,
        config: EditMethodConfig,
    ) -> Dict[str, Any]:
        """Prepare source SLAT latent cache via inversion.

        Args:
            source_slat: Source SLAT features (SparseTensor)
            inversion_mode: "simple" (Euler) or "rf_solver" (VoxHammer)

        Returns:
            Dict mapping timestep to SLAT: {f"{t}": SparseTensor}
        """
        # Normalize source SLAT
        std = torch.tensor(pipeline.slat_normalization["std"], device=pipeline.device)[None]
        mean = torch.tensor(pipeline.slat_normalization["mean"], device=pipeline.device)[None]
        source_slat_normalized = (source_slat - mean) / std

        if inversion_mode == "simple":
            # Simple Euler inversion
            print("Running simple Euler inversion for SLAT...")
            latent_cache = self._simple_euler_inversion_slat(
                pipeline,
                source_slat_normalized,
                source_cond_dict,
                config,
            )
            return latent_cache

        elif inversion_mode == "rf_solver":
            # RF-Solver inversion (VoxHammer)
            print("Running RF-Solver inversion for SLAT...")
            latent_cache = self._rf_solver_inversion_slat(
                pipeline,
                source_slat_normalized,
                source_cond_dict,
                config,
            )
            return latent_cache

        else:
            raise ValueError(f"Unknown inversion_mode: {inversion_mode}")

    def _simple_euler_inversion_slat(
        self,
        pipeline,
        slat_normalized,
        cond_dict,
        config: EditMethodConfig,
    ) -> Dict[str, Any]:
        """Simple first-order Euler inversion for SLAT stage."""
        model = pipeline.models["slat_flow_model"]
        sigma_min = pipeline.slat_sampler.sigma_min

        sampler_params = config.slat_sampler_params or {}
        steps = sampler_params.get("steps", 25)
        rescale_t = 3.0

        # Time sequence: 0 -> 1 (data to noise)
        t_seq = np.linspace(0, 1, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)

        latent_cache = {}
        sample = slat_normalized

        # Disable gradients for inversion
        with torch.inference_mode():
            for i in range(steps):
                t_curr = t_seq[i]
                t_next = t_seq[i + 1]

                # Cache current latent (move to CPU to save GPU memory)
                latent_cache[f"{t_next}"] = sample.cpu()

                # Euler step: x_{t+dt} = x_t + dt * f(x_t, t)
                t_tensor = torch.tensor([1000 * t_curr], device=sample.coords.device, dtype=torch.float32)
                pred_v = model(sample, t_tensor, cond_dict["cond"])
                sample = sample + (t_next - t_curr) * pred_v

                # Clear cache every few steps
                if i % 5 == 0:
                    torch.cuda.empty_cache()

        return latent_cache

    def _rf_solver_inversion_slat(
        self,
        pipeline,
        slat_normalized,
        cond_dict,
        config: EditMethodConfig,
    ) -> Dict[str, Any]:
        """RF-Solver inversion (second-order Taylor) for SLAT stage."""
        model = pipeline.models["slat_flow_model"]
        sigma_min = pipeline.slat_sampler.sigma_min

        sampler_params = config.slat_sampler_params or {}
        steps = sampler_params.get("steps", 25)
        rescale_t = 3.0

        # Time sequence: 0 -> 1 (data to noise)
        t_seq = np.linspace(0, 1, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)

        latent_cache = {}
        sample = slat_normalized

        # Disable gradients for inversion
        with torch.inference_mode():
            for i in range(steps):
                t_curr = t_seq[i]
                t_next = t_seq[i + 1]
                dt = t_next - t_curr

                # Cache current latent (move to CPU to save GPU memory)
                latent_cache[f"{t_next}"] = sample.cpu()

                # First prediction at current point
                t_tensor = torch.tensor([1000 * t_curr], device=sample.coords.device, dtype=torch.float32)
                pred_v = model(sample, t_tensor, cond_dict["cond"])

                # Midpoint
                sample_mid = sample + (dt / 2) * pred_v
                t_mid = t_curr + dt / 2
                t_mid_tensor = torch.tensor([1000 * t_mid], device=sample.coords.device, dtype=torch.float32)
                pred_v_mid = model(sample_mid, t_mid_tensor, cond_dict["cond"])

                # Second-order update
                first_order = (pred_v_mid - pred_v) / (dt / 2)
                sample = sample + dt * pred_v + 0.5 * (dt ** 2) * first_order

                # Clear cache every few steps
                if i % 5 == 0:
                    torch.cuda.empty_cache()

        return latent_cache
        return latent_cache

    def _blend_coords_with_mask(
        self,
        source_coords: torch.Tensor,
        edit_coords: torch.Tensor,
        mask_coords: torch.Tensor,
    ) -> torch.Tensor:
        """Blend coords at coord-level based on 3D mask.

        Strategy:
        - Preserve region (NOT in mask): Keep source coords
        - Edit region (IN mask): Keep edit coords
        - Overlapping coords: Keep (both have them)

        Args:
            source_coords: Source coords [N_src, 4] (batch, x, y, z)
            edit_coords: Edit coords [N_edit, 4] (batch, x, y, z)
            mask_coords: Mask coords [N_mask, 3] or [N_mask, 4]

        Returns:
            Blended coords [N_blended, 4]
        """
        # Build mask hash set
        mask_hash_set = set()
        for coord in mask_coords.cpu():
            if len(coord) == 4:
                _, x, y, z = coord.tolist()
            else:
                x, y, z = coord.tolist()
            mask_hash_set.add(f"{int(x)}_{int(y)}_{int(z)}")

        # Collect blended coords
        blended_set = set()
        blended_coords_list = []

        # Add source coords NOT in mask (preserve region)
        for coord in source_coords.cpu():
            b, x, y, z = coord.tolist()
            coord_hash = f"{int(x)}_{int(y)}_{int(z)}"

            if coord_hash not in mask_hash_set:  # Preserve region
                if coord_hash not in blended_set:
                    blended_set.add(coord_hash)
                    blended_coords_list.append([b, x, y, z])

        # Add edit coords IN mask (edit region)
        for coord in edit_coords.cpu():
            b, x, y, z = coord.tolist()
            coord_hash = f"{int(x)}_{int(y)}_{int(z)}"

            if coord_hash in mask_hash_set:  # Edit region
                if coord_hash not in blended_set:
                    blended_set.add(coord_hash)
                    blended_coords_list.append([b, x, y, z])

        # Convert to tensor
        if len(blended_coords_list) == 0:
            return torch.zeros(0, 4, dtype=source_coords.dtype, device=source_coords.device)

        blended_coords = torch.tensor(
            blended_coords_list,
            dtype=source_coords.dtype,
            device=source_coords.device
        )

        return blended_coords

    def _build_ss_latent_mask(
        self,
        mask_coords: torch.Tensor,
        resolution: int = 64,
    ) -> torch.Tensor:
        """Build SS latent mask from 3D mask coords.

        Args:
            mask_coords: 3D mask coordinates [N, 3] (x, y, z) or [N, 4] (batch, x, y, z)
            resolution: Voxel grid resolution (default 64)

        Returns:
            3D latent mask [1, 8, 16, 16, 16]
                0 = preserve source, 1 = use edit (mask region)
        """
        # Create voxel grid at full resolution (64^3)
        voxel_mask_64 = torch.zeros(1, 1, resolution, resolution, resolution, dtype=torch.float32)

        # Fill in mask voxels
        if mask_coords.shape[0] > 0:
            # Handle both [N, 3] and [N, 4] formats
            if mask_coords.shape[1] == 4:
                coords_int = mask_coords[:, 1:].long()  # [N, 3] (x, y, z)
            else:
                coords_int = mask_coords.long()  # Already [N, 3]

            voxel_mask_64[0, 0, coords_int[:, 0], coords_int[:, 1], coords_int[:, 2]] = 1.0

        # Downsample to latent resolution (16^3) using max pooling
        # This ensures that if any voxel in a 4x4x4 block is masked, the latent voxel is masked
        import torch.nn.functional as F
        voxel_mask_16 = F.max_pool3d(
            voxel_mask_64,
            kernel_size=4,
            stride=4,
        )  # [1, 1, 16, 16, 16]

        # Expand to all channels
        latent_mask = voxel_mask_16.expand(1, 8, 16, 16, 16)  # [1, 8, 16, 16, 16]

        return latent_mask.contiguous()

    def _build_slat_preserve_coords_from_mask(
        self,
        source_slat,
        mask_coords: torch.Tensor,
    ) -> torch.Tensor:
        """Build SLAT preserve region coords from 3D mask.

        Args:
            source_slat: Source SLAT (SparseTensor)
            mask_coords: 3D mask coordinates [N, 3] (x, y, z) or [N, 4] (batch, x, y, z)

        Returns:
            Preserve coords [N_preserve, 4] (batch, x, y, z)
        """
        # Get source coords
        source_coords = source_slat.coords  # [N, 4]

        # Build mask hash set for fast lookup
        mask_hash_set = set()
        for coord in mask_coords.cpu():
            if len(coord) == 4:
                b, x, y, z = coord.tolist()
            else:  # len == 3
                x, y, z = coord.tolist()
                b = 0
            mask_hash_set.add(f"{int(b)}_{int(x)}_{int(y)}_{int(z)}")

        # Find source coords NOT in mask (preserve region)
        preserve_coords = []
        for coord in source_coords:
            b, x, y, z = coord.tolist()
            coord_hash = f"{int(b)}_{int(x)}_{int(y)}_{int(z)}"

            # Preserve if NOT in mask
            if coord_hash not in mask_hash_set:
                preserve_coords.append(coord.tolist())

        if len(preserve_coords) == 0:
            return torch.zeros(0, 4, dtype=torch.int32, device=source_coords.device)

        return torch.tensor(preserve_coords, dtype=torch.int32, device=source_coords.device)

    def _build_slat_preserve_coords(
        self,
        source_slat,
        edit_coords: torch.Tensor,
        spatial_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build SLAT preserve region coords from mask.

        Args:
            source_slat: Source SLAT (SparseTensor)
            edit_coords: Edit coords [B, N, 3]
            spatial_mask: 2D mask [1, 1, H, W]

        Returns:
            Preserve coords [N_preserve, 4] (batch, x, y, z)
        """
        # Get source coords
        source_coords = source_slat.coords  # [N, 4]

        # Project to 2D and check mask
        resolution = 64  # Assume 64^3 grid
        mask_h, mask_w = spatial_mask.shape[2], spatial_mask.shape[3]

        preserve_coords = []
        for coord in source_coords:
            b, x, y, z = coord.tolist()
            # Project to 2D
            u = int((x / resolution) * mask_w)
            v = int((y / resolution) * mask_h)
            u = max(0, min(mask_w - 1, u))
            v = max(0, min(mask_h - 1, v))

            mask_value = spatial_mask[0, 0, v, u].item()

            # Preserve if mask_value < 0.5 (unmasked region)
            if mask_value < 0.5:
                preserve_coords.append(coord.tolist())

        if len(preserve_coords) == 0:
            return torch.zeros(0, 4, dtype=torch.int32, device=source_coords.device)

        return torch.tensor(preserve_coords, dtype=torch.int32, device=source_coords.device)

    def run(
        self,
        pipeline,
        prepared_state: Dict,
        config: EditMethodConfig,
    ) -> EditMethodOutputs:
        """Run P2P with per-step latent blending."""

        source_slat = prepared_state["source_slat"]
        source_cond_dict = prepared_state["source_cond_dict"]
        edit_cond_dict = prepared_state["edit_cond_dict"]
        mask_coords = prepared_state["mask_coords"]
        slat_spatial_mask_2d = prepared_state["slat_spatial_mask_2d"]
        blend_ss_enabled = prepared_state["blend_ss_enabled"]
        blend_slat_enabled = prepared_state["blend_slat_enabled"]
        blend_strength = prepared_state["blend_strength"]
        inversion_mode = prepared_state["inversion_mode"]
        resolution = prepared_state["resolution"]

        # Get sampler params
        ss_params = config.sparse_structure_sampler_params or {}
        slat_params = config.slat_sampler_params or {}

        # Step 1: Prepare source latents for blending via inversion
        # IMPORTANT: Do inversion BEFORE patching P2P hooks to save memory
        source_ss_latent_cache = None
        if blend_ss_enabled:
            print("Preparing source SS latent via inversion...")
            # Clear cache before inversion
            torch.cuda.empty_cache()
            source_ss_latent_cache = self._prepare_ss_source(
                pipeline,
                source_cond_dict,
                inversion_mode,
                config,
            )
            print(f"Cached {len(source_ss_latent_cache)} SS latent timesteps")
            # Clear cache after inversion
            torch.cuda.empty_cache()

        # Step 2: Patch P2P hooks (after inversion to save memory)
        if self.hook:
            # Only patch stages that are in stage_configs
            stage_configs = prepared_state.get("stage_configs", {})
            if "sparse_structure" in stage_configs:
                self.hook.patch_model(pipeline.models["sparse_structure_flow_model"], "sparse_structure")
            if "slat" in stage_configs:
                self.hook.patch_model(pipeline.models["slat_flow_model"], "slat")

        # Step 3: Replace samplers with custom latent blend samplers
        from editing.samplers import LatentBlendFlowEulerGuidanceIntervalSampler

        # Create custom SS sampler
        original_ss_sampler = pipeline.sparse_structure_sampler
        if blend_ss_enabled and source_ss_latent_cache is not None and mask_coords is not None:
            print("Setting up SS latent blending...")
            ss_sampler = LatentBlendFlowEulerGuidanceIntervalSampler(
                sigma_min=original_ss_sampler.sigma_min
            )
            # Build SS latent mask from 3D mask coords
            ss_latent_mask = self._build_ss_latent_mask(mask_coords, resolution)
            print(f"SS latent mask shape: {ss_latent_mask.shape}")
            print(f"Mask voxels (1=edit): {(ss_latent_mask > 0.5).sum().item()} / {ss_latent_mask.numel()}")
            print(f"Preserve voxels (0=keep source): {(ss_latent_mask <= 0.5).sum().item()} / {ss_latent_mask.numel()}")
            print(f"Mask mean value: {ss_latent_mask.mean().item():.3f}")
            ss_sampler.set_blend_source(
                source_latent_cache=source_ss_latent_cache,
                latent_mask=ss_latent_mask,
                is_sparse=False,
            )
            pipeline.sparse_structure_sampler = ss_sampler
        else:
            print("SS blending disabled")


        # Step 3: Generate edit coords with SS blending
        print("Generating edit sparse structure with P2P + SS blending...")
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Generate SS latent (z_s)
        flow_model = pipeline.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(config.num_samples, flow_model.in_channels, reso, reso, reso).to(pipeline.device)
        sampler_params_merged = {**pipeline.sparse_structure_sampler_params, **ss_params}
        z_s = pipeline.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **edit_cond_dict,  # Unpack cond and neg_cond
            **sampler_params_merged,
            verbose=True
        ).samples

        # Decode to voxel
        decoder = pipeline.models['sparse_structure_decoder']
        edit_voxel = decoder(z_s)  # [1, 1, 64, 64, 64]
        print(f"Edit voxel shape: {edit_voxel.shape}")

        # Step 3.5: Voxel-level blending (coordinate-based union)
        # This ensures proper region separation at coordinate level
        if blend_ss_enabled and mask_coords is not None:
            print("\n=== Applying Voxel-Level Blending ===")
            # Extract coords from edit_voxel
            edit_coords_raw = torch.argwhere(edit_voxel > 0)[:, [0, 2, 3, 4]].int()

            # Build source voxel from source_slat coords
            source_coords = source_slat.coords  # [N, 4] (batch, x, y, z)

            # Build voxel mask hash set
            mask_hash_set = set()
            for coord in mask_coords.cpu():
                if len(coord) == 4:
                    _, x, y, z = coord.tolist()
                else:
                    x, y, z = coord.tolist()
                mask_hash_set.add(f"{int(x)}_{int(y)}_{int(z)}")

            print(f"Voxel mask: {len(mask_hash_set)} edit voxels")
            print(f"Source voxel: {source_coords.shape[0]} occupied voxels")
            print(f"Edit voxel (before blend): {edit_coords_raw.shape[0]} occupied voxels")

            # Coordinate-based blending:
            # - Preserve region (NOT in mask): Keep source coords
            # - Edit region (IN mask): Keep edit coords
            blended_coords_set = set()
            blended_coords_list = []

            # Add source coords NOT in mask (preserve region)
            for coord in source_coords.cpu():
                b, x, y, z = coord.tolist()
                coord_hash = f"{int(x)}_{int(y)}_{int(z)}"
                if coord_hash not in mask_hash_set:  # Preserve region
                    if coord_hash not in blended_coords_set:
                        blended_coords_set.add(coord_hash)
                        blended_coords_list.append([b, x, y, z])

            # Add edit coords IN mask (edit region)
            for coord in edit_coords_raw.cpu():
                b, x, y, z = coord.tolist()
                coord_hash = f"{int(x)}_{int(y)}_{int(z)}"
                if coord_hash in mask_hash_set:  # Edit region
                    if coord_hash not in blended_coords_set:
                        blended_coords_set.add(coord_hash)
                        blended_coords_list.append([b, x, y, z])

            # Convert to tensor
            if len(blended_coords_list) == 0:
                edit_coords = torch.zeros(0, 4, dtype=torch.int32, device=pipeline.device)
            else:
                edit_coords = torch.tensor(
                    blended_coords_list,
                    dtype=torch.int32,
                    device=pipeline.device
                )

            print(f"Blended voxel: {edit_coords.shape[0]} occupied voxels")
        else:
            # No voxel blending, just extract coords from edit_voxel
            edit_coords = torch.argwhere(edit_voxel > 0)[:, [0, 2, 3, 4]].int()

        print(f"Edit coords: {edit_coords.shape}")

        # Analyze SS coords overlap with source
        print("\n=== SS Coords Analysis ===")
        source_coords = source_slat.coords  # [N, 4] (batch, x, y, z)
        print(f"Source coords: {source_coords.shape[0]}")
        print(f"Edit coords: {edit_coords.shape[0]}")

        # Build hash sets for comparison
        source_hash_set = set()
        for coord in source_coords.cpu():
            b, x, y, z = coord.tolist()
            source_hash_set.add(f"{int(x)}_{int(y)}_{int(z)}")

        edit_hash_set = set()
        for coord in edit_coords.cpu():
            b, x, y, z = coord.tolist()
            edit_hash_set.add(f"{int(x)}_{int(y)}_{int(z)}")

        overlap = source_hash_set & edit_hash_set
        source_only = source_hash_set - edit_hash_set
        edit_only = edit_hash_set - source_hash_set

        print(f"Overlapping coords: {len(overlap)} ({len(overlap)/len(source_hash_set)*100:.1f}% of source)")
        print(f"Source-only coords: {len(source_only)} ({len(source_only)/len(source_hash_set)*100:.1f}%)")
        print(f"Edit-only coords: {len(edit_only)} ({len(edit_only)/len(edit_hash_set)*100:.1f}%)")

        # Analyze overlap in preserve region using 3D mask
        if mask_coords is not None:
            # Build mask hash set
            mask_hash_set = set()
            for coord in mask_coords.cpu():
                if len(coord) == 4:
                    _, x, y, z = coord.tolist()
                else:  # len == 3
                    x, y, z = coord.tolist()
                mask_hash_set.add(f"{int(x)}_{int(y)}_{int(z)}")

            preserve_overlap = 0
            preserve_source_only = 0
            preserve_edit_only = 0
            edit_overlap = 0
            edit_source_only = 0
            edit_edit_only = 0

            # Overlap coords
            for coord_hash in overlap:
                if coord_hash in mask_hash_set:
                    edit_overlap += 1  # In mask (edit region)
                else:
                    preserve_overlap += 1  # Not in mask (preserve region)

            # Source-only coords
            for coord_hash in source_only:
                if coord_hash in mask_hash_set:
                    edit_source_only += 1  # In mask (edit region)
                else:
                    preserve_source_only += 1  # Not in mask (preserve region)

            # Edit-only coords
            for coord_hash in edit_only:
                if coord_hash in mask_hash_set:
                    edit_edit_only += 1  # In mask (edit region)
                else:
                    preserve_edit_only += 1  # Not in mask (preserve region)

            total_preserve_source = preserve_overlap + preserve_source_only
            total_edit_source = edit_overlap + edit_source_only

            print(f"\n=== By Region Analysis ===")
            print(f"\nPRESERVE region (NOT in mask, should keep source):")
            print(f"  Source voxels in this region: {total_preserve_source}")
            print(f"    - Overlapping with edit: {preserve_overlap} ({preserve_overlap/total_preserve_source*100 if total_preserve_source > 0 else 0:.1f}%)")
            print(f"    - Source-only (missing in edit): {preserve_source_only} ({preserve_source_only/total_preserve_source*100 if total_preserve_source > 0 else 0:.1f}%)")
            print(f"  Edit-only voxels (shouldn't be here): {preserve_edit_only}")

            print(f"\nEDIT region (IN mask, should generate new):")
            print(f"  Source voxels in this region: {total_edit_source}")
            print(f"    - Overlapping with edit: {edit_overlap} ({edit_overlap/total_edit_source*100 if total_edit_source > 0 else 0:.1f}%)")
            print(f"    - Source-only (will be replaced): {edit_source_only} ({edit_source_only/total_edit_source*100 if total_edit_source > 0 else 0:.1f}%)")
            print(f"  Edit-only voxels (new generation): {edit_edit_only}")

        # Restore original SS sampler
        pipeline.sparse_structure_sampler = original_ss_sampler

        # Step 4: Prepare SLAT blending
        original_slat_sampler = pipeline.slat_sampler
        source_slat_latent_cache = None
        if blend_slat_enabled:
            print("Preparing source SLAT latent via inversion...")
            # Clear cache before SLAT inversion
            torch.cuda.empty_cache()
            source_slat_latent_cache = self._prepare_slat_source(
                pipeline,
                source_slat,
                source_cond_dict,
                inversion_mode,
                config,
            )
            print(f"Cached {len(source_slat_latent_cache)} SLAT latent timesteps")
            # Clear cache after SLAT inversion
            torch.cuda.empty_cache()

            print("Setting up SLAT latent blending...")
            slat_sampler = LatentBlendFlowEulerGuidanceIntervalSampler(
                sigma_min=original_slat_sampler.sigma_min
            )
            # Build SLAT preserve coords from 3D mask
            slat_preserve_coords = self._build_slat_preserve_coords_from_mask(
                source_slat,
                mask_coords,
            )
            slat_sampler.set_blend_source(
                source_latent_cache=source_slat_latent_cache,
                latent_mask=slat_preserve_coords,
                is_sparse=True,
            )
            pipeline.slat_sampler = slat_sampler
        else:
            print("SLAT blending disabled")

        # Step 5: Generate edit SLAT with blending
        print("Generating edit SLAT with P2P + SLAT blending...")
        edit_slat = pipeline.sample_slat(
            edit_cond_dict,
            edit_coords,
            sampler_params=slat_params,
        )
        print(f"Edit SLAT: {edit_slat.coords.shape[0]} voxels")

        # Print SLAT blending statistics
        if blend_slat_enabled and hasattr(pipeline.slat_sampler, 'stats'):
            stats = pipeline.slat_sampler.stats
            print(f"\n=== SLAT Blending Statistics ===")
            print(f"Blend calls: {stats['blend_calls']}")
            print(f"Sample voxels (last step): {stats['last_sample_voxels']}")
            print(f"Source voxels (cached): {stats['last_source_voxels']}")
            print(f"Preserve region coords: {stats['last_preserve_coords']}")
            print(f"Sample voxels matching preserve region: {stats['last_sample_match']}")
            print(f"Source voxels matching preserve region: {stats['last_source_match']}")
            if stats['last_preserve_coords'] > 0:
                print(f"Sample overlap ratio: {stats['last_sample_match'] / stats['last_preserve_coords'] * 100:.1f}%")
                print(f"Source overlap ratio: {stats['last_source_match'] / stats['last_preserve_coords'] * 100:.1f}%")

        # Restore original SLAT sampler
        pipeline.slat_sampler = original_slat_sampler

        # Step 6: Decode
        print("Decoding result...")
        outputs = pipeline.decode_slat(edit_slat, ["mesh", "gaussian"])
        source_outputs = pipeline.decode_slat(source_slat, ["mesh", "gaussian"])

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
        """Save artifacts."""
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
        """Cleanup hooks."""
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
            "slat_t_start": 1.0,
            "slat_t_end": 0.8,
            "slat_strength": 1.0,
            "patch_coverage_threshold": 0.0,
            "query_chunk": 1024,
            # Latent blending params
            "blend_ss_enabled": True,
            "blend_slat_enabled": True,  # Now supported with inversion
            "ss_blend_mode": "hard",
            "slat_blend_mode": "hard",
            "ss_soft_kernel_size": 3,
            "slat_soft_kernel_size": 5,
            "blend_strength": 1.0,
            "ss_blend_t_start": 1.0,
            "ss_blend_t_end": 0.0,
            "slat_blend_t_start": 1.0,
            "slat_blend_t_end": 0.0,
            # Inversion params (ablation)
            "inversion_mode": "simple",  # "simple" (Euler) or "rf_solver" (VoxHammer)
            # Output params
            "skip_render": True,
            "skip_glb": False,
            "skip_ply": False,
        }
