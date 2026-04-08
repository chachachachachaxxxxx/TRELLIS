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
from editing.preprocess.asset_3d import feats_to_slat
from editing.utils import build_image_token_metadata, resolve_patch_size, save_mask_overlay_preview, save_patch_grid_preview


class ImageP2PLatentBlendMethod(EditMethod):
    """Image P2P with post-generation latent blending.

    Approach:
    1. Load source 3D SLAT features (from preprocessed assets)
    2. Generate edit 3D with P2P + per-step latent blending
    3. Blend latents at each denoising step based on mask

    This is simpler than UniEdit - no RF inversion, just blend latents during generation.
    Uses real source features instead of regenerating.
    """

    def __init__(self):
        super().__init__("image_p2p_latent_blend")
        self.hook: PromptToPromptHook | None = None
        self.latent_blend_hook: LatentBlendHook | None = None

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
        # Check that asset_dir is provided
        if not inputs.asset_dir:
            raise ValueError(
                "image_p2p_latent_blend requires asset_dir (preprocessed 3D assets). "
                "Use --source-model path/to/assets or preprocess with --preprocess."
            )

        # Get extra params
        extra = config.extra_params or {}
        inject_stages = extra.get("inject_stages", ["sparse_structure", "slat"])
        patch_coverage_threshold = extra.get("patch_coverage_threshold", 0.0)
        query_chunk = extra.get("query_chunk", 1024)

        # Latent blending params
        blend_ss_enabled = extra.get("blend_ss_enabled", False)  # Sparse structure blending
        blend_slat_enabled = extra.get("blend_slat_enabled", True)  # SLAT blending
        ss_blend_mode = extra.get("ss_blend_mode", "hard")  # "hard" or "soft"
        slat_blend_mode = extra.get("slat_blend_mode", "soft")  # "hard" or "soft"
        ss_soft_kernel = extra.get("ss_soft_kernel_size", 3)
        slat_soft_kernel = extra.get("slat_soft_kernel_size", 5)
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

        # Load source SLAT features from preprocessed assets
        print("Loading source SLAT features from preprocessed assets...")
        from trellis.modules.sparse.basic import SparseTensor
        source_slat = feats_to_slat(
            pipeline=pipeline,
            feats_path=inputs.asset_dir / "features.npz",
            SparseTensor=SparseTensor,
        )
        print(f"Loaded source SLAT: {source_slat.coords.shape[0]} voxels")

        # Encode conditions (only need edit, not source)
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

        # Build spatial masks for blending (separate for SS and SLAT)
        ss_spatial_mask = None
        slat_spatial_mask = None

        if blend_ss_enabled:
            ss_spatial_mask = self._build_spatial_mask(
                inputs.mask_image,
                blend_mode=ss_blend_mode,
                kernel_size=ss_soft_kernel,
            )

        if blend_slat_enabled:
            slat_spatial_mask = self._build_spatial_mask(
                inputs.mask_image,
                blend_mode=slat_blend_mode,
                kernel_size=slat_soft_kernel,
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
            "source_slat": source_slat,
            "source_cond_dict": source_cond_dict,
            "edit_cond_dict": edit_cond_dict,
            "token_meta": token_meta,
            "stage_configs": stage_configs,
            "ss_spatial_mask": ss_spatial_mask,
            "slat_spatial_mask": slat_spatial_mask,
            "blend_ss_enabled": blend_ss_enabled,
            "blend_slat_enabled": blend_slat_enabled,
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
        source_slat = prepared_state["source_slat"]
        source_cond_dict = prepared_state["source_cond_dict"]
        edit_cond_dict = prepared_state["edit_cond_dict"]
        ss_spatial_mask = prepared_state["ss_spatial_mask"]
        slat_spatial_mask = prepared_state["slat_spatial_mask"]
        blend_ss_enabled = prepared_state["blend_ss_enabled"]
        blend_slat_enabled = prepared_state["blend_slat_enabled"]
        blend_strength = prepared_state["blend_strength"]
        blend_strength = prepared_state["blend_strength"]

        # Get sampler params
        ss_params = config.sparse_structure_sampler_params or {}
        slat_params = config.slat_sampler_params or {}

        # Patch models with P2P hook
        if self.hook:
            self.hook.patch_model(pipeline.models["sparse_structure_flow_model"], "sparse_structure")
            self.hook.patch_model(pipeline.models["slat_flow_model"], "slat")

        # Create and patch SS latent blend hook if enabled
        self.ss_blend_hook = None
        if blend_ss_enabled and ss_spatial_mask is not None:
            print(f"Creating SS latent blend hook (strength={blend_strength})...")
            resolution = pipeline.sparse_structure_sampler_params.get("grid_size", 64)

            # Get blending time range from config
            extra = config.extra_params or {}
            ss_blend_t_start = extra.get("ss_blend_t_start", 1.0)
            ss_blend_t_end = extra.get("ss_blend_t_end", 0.0)

            # Generate source SS coords for blending
            print("Generating source sparse structure for SS blending...")
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
            source_coords = pipeline.sample_sparse_structure(
                source_cond_dict,
                num_samples=config.num_samples,
                sampler_params=ss_params,
            )

            self.ss_blend_hook = LatentBlendHook(
                source_slat=source_coords,  # Use coords as "slat" for SS stage
                spatial_mask=ss_spatial_mask,
                blend_strength=blend_strength,
                resolution=resolution,
                t_start=ss_blend_t_start,
                t_end=ss_blend_t_end,
            )
            self.ss_blend_hook.patch_sampler(pipeline.sparse_structure_sampler)
            print(f"SS latent blending enabled: t=[{ss_blend_t_start}, {ss_blend_t_end}]")

        # Create and patch SLAT latent blend hook if enabled
        self.latent_blend_hook = None
        if blend_slat_enabled and slat_spatial_mask is not None:
            print(f"Creating SLAT latent blend hook (strength={blend_strength})...")
            resolution = pipeline.sparse_structure_sampler_params.get("grid_size", 64)

            # Get blending time range from config
            extra = config.extra_params or {}
            blend_t_start = extra.get("blend_t_start", 1.0)
            blend_t_end = extra.get("blend_t_end", 0.0)

            self.latent_blend_hook = LatentBlendHook(
                source_slat=source_slat,
                spatial_mask=slat_spatial_mask,
                blend_strength=blend_strength,
                resolution=resolution,
                t_start=blend_t_start,
                t_end=blend_t_end,
            )
            self.latent_blend_hook.patch_sampler(pipeline.slat_sampler)
            print(f"SLAT latent blending enabled: t=[{blend_t_start}, {blend_t_end}]")

        # Step 1: Use loaded source SLAT (no generation needed)
        print("Using loaded source SLAT features...")
        source_coords = source_slat.coords
        print(f"Source SLAT: {source_coords.shape[0]} voxels")

        # Step 2: Generate edit 3D with P2P + latent blending
        print("Generating edit 3D with P2P + per-step latent blending...")
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        edit_coords = pipeline.sample_sparse_structure(
            edit_cond_dict,
            num_samples=config.num_samples,
            sampler_params=ss_params,
        )

        # Generate edit SLAT with latent blending hook active
        edit_slat = pipeline.sample_slat(
            edit_cond_dict,
            edit_coords,
            sampler_params=slat_params,
        )

        # Step 3: Decode result
        print("Decoding blended result...")
        outputs = pipeline.decode_slat(edit_slat, ["mesh", "gaussian"])

        # Print blending statistics
        if self.ss_blend_hook:
            stats = self.ss_blend_hook.stats
            print(f"\n=== SS Voxel Overlap Statistics ===")
            print(f"Source SS voxels: {stats['total_source_voxels']}")
            print(f"Edit SS voxels (final): {stats['last_edit_voxels']}")
            print(f"Overlapping voxels: {stats['last_overlap_count']}")
            print(f"Overlap ratio (vs source): {stats['last_overlap_count'] / stats['total_source_voxels'] * 100:.2f}%")
            print(f"Overlap ratio (vs edit): {stats['last_overlap_count'] / stats['last_edit_voxels'] * 100:.2f}%")
            print(f"Edit-only voxels (newly generated): {stats['last_edit_voxels'] - stats['last_overlap_count']} ({(stats['last_edit_voxels'] - stats['last_overlap_count']) / stats['last_edit_voxels'] * 100:.2f}%)")
            print(f"Blend calls: {stats['blend_calls']}")

        if self.latent_blend_hook:
            stats = self.latent_blend_hook.stats
            print(f"\n=== SLAT Voxel Overlap Statistics ===")
            print(f"Source SLAT voxels: {stats['total_source_voxels']}")
            print(f"Edit SLAT voxels (final): {stats['last_edit_voxels']}")
            print(f"Overlapping voxels: {stats['last_overlap_count']}")
            print(f"Overlap ratio (vs source): {stats['last_overlap_count'] / stats['total_source_voxels'] * 100:.2f}%")
            print(f"Overlap ratio (vs edit): {stats['last_overlap_count'] / stats['last_edit_voxels'] * 100:.2f}%")
            print(f"Edit-only voxels (newly generated): {stats['last_edit_voxels'] - stats['last_overlap_count']} ({(stats['last_edit_voxels'] - stats['last_overlap_count']) / stats['last_edit_voxels'] * 100:.2f}%)")
            print(f"Blend calls: {stats['blend_calls']}")

        # Also decode source for comparison
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
        if self.latent_blend_hook:
            self.latent_blend_hook.restore()
            self.latent_blend_hook = None

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
            # Latent blending params - two stages independently controlled
            "blend_ss_enabled": True,  # Sparse structure blending (critical for coord alignment!)
            "blend_slat_enabled": True,  # SLAT blending (main feature)
            "ss_blend_mode": "hard",  # "hard" or "soft" for sparse structure
            "slat_blend_mode": "hard",  # "hard" or "soft" for SLAT
            "ss_soft_kernel_size": 3,  # Kernel size for SS soft mask
            "slat_soft_kernel_size": 5,  # Kernel size for SLAT soft mask
            "blend_strength": 1.0,  # Overall blending strength
            "ss_blend_t_start": 1.0,  # Start timestep for SS blending
            "ss_blend_t_end": 0.0,  # End timestep for SS blending
            "blend_t_start": 1.0,  # Start timestep for SLAT blending (1.0 = beginning)
            "blend_t_end": 0.5,  # End timestep for SLAT blending (0.0 = end, full range)
            # Output params
            "skip_render": True,
            "skip_glb": False,
            "skip_ply": False,
        }


class LatentBlendHook:
    """Hook for blending latents during denoising process.

    This hook intercepts the denoising steps and blends the current latent
    with a reference latent (source) based on a spatial mask.
    """

    def __init__(
        self,
        source_slat,
        spatial_mask: torch.Tensor,
        blend_strength: float,
        resolution: int,
        t_start: float = 1.0,
        t_end: float = 0.0,
    ):
        """Initialize latent blend hook.

        Args:
            source_slat: Source SLAT features (SparseTensor)
            spatial_mask: 2D spatial mask [1, 1, H, W], 0=preserve source, 1=use edit
            blend_strength: Blending strength (0-1)
            resolution: Voxel grid resolution
            t_start: Start timestep for blending (normalized 0-1)
            t_end: End timestep for blending (normalized 0-1)
        """
        self.source_slat = source_slat
        self.spatial_mask = spatial_mask
        self.blend_strength = blend_strength
        self.resolution = resolution
        self.t_start = t_start
        self.t_end = t_end

        # Build coordinate hash for fast lookup
        self.src_coords_3d = source_slat.coords[:, 1:].cpu()  # [N_src, 3]
        self.src_hash_to_idx = self._build_coord_hash(self.src_coords_3d)
        self.src_feats = source_slat.feats

        # Statistics tracking
        self.stats = {
            "total_source_voxels": len(self.src_coords_3d),
            "blend_calls": 0,
            "last_overlap_count": 0,
            "last_edit_voxels": 0,
        }

        # Cache mask resolution
        self.mask_h = spatial_mask.shape[2]
        self.mask_w = spatial_mask.shape[3]

        self._original_sample_once = None

    def _build_coord_hash(self, coords_3d):
        """Build hash map from 3D coords to indices."""
        hash_to_idx = {}
        for i, (x, y, z) in enumerate(coords_3d.tolist()):
            hash_str = f"{int(x)}_{int(y)}_{int(z)}"
            hash_to_idx[hash_str] = i
        return hash_to_idx

    def should_blend(self, t_norm: float) -> bool:
        """Check if blending should be applied at this timestep."""
        lo = min(self.t_start, self.t_end)
        hi = max(self.t_start, self.t_end)
        return lo <= t_norm <= hi

    def blend_latent(self, latent, t_norm: float):
        """Blend latent with source based on mask.

        Args:
            latent: Current latent (SparseTensor)
            t_norm: Normalized timestep (0-1)

        Returns:
            Blended latent (SparseTensor)
        """
        if not self.should_blend(t_norm):
            return latent

        # Get edit coordinates and features
        edit_coords_3d = latent.coords[:, 1:].cpu()  # [N_edit, 3]
        edit_feats = latent.feats

        # Build edit hash
        edit_hash_to_idx = self._build_coord_hash(edit_coords_3d)

        # Find overlapping voxels
        overlap_hashes = set(self.src_hash_to_idx.keys()) & set(edit_hash_to_idx.keys())

        # Update statistics
        self.stats["blend_calls"] += 1
        self.stats["last_overlap_count"] = len(overlap_hashes)
        self.stats["last_edit_voxels"] = len(edit_coords_3d)

        if len(overlap_hashes) == 0:
            return latent

        # Blend features
        blended_feats = edit_feats.clone()

        for hash_str in overlap_hashes:
            src_idx = self.src_hash_to_idx[hash_str]
            edit_idx = edit_hash_to_idx[hash_str]

            # Get 3D coordinate
            x, y, _z = self.src_coords_3d[src_idx]

            # Project to 2D mask space (simple top-down projection)
            u = int((x / self.resolution) * self.mask_w)
            v = int((y / self.resolution) * self.mask_h)
            u = max(0, min(self.mask_w - 1, u))
            v = max(0, min(self.mask_h - 1, v))

            # Get mask value
            mask_value = self.spatial_mask[0, 0, v, u].item()

            # Compute blend weight: mask_value=0 -> use source, mask_value=1 -> use edit
            blend_weight = (1.0 - mask_value) * self.blend_strength

            # Blend features
            src_feat = self.src_feats[src_idx]
            edit_feat = edit_feats[edit_idx]
            blended_feat = blend_weight * src_feat + (1.0 - blend_weight) * edit_feat

            blended_feats[edit_idx] = blended_feat

        # Create blended latent
        from trellis.modules import sparse as sp
        blended_latent = sp.SparseTensor(
            feats=blended_feats,
            coords=latent.coords,
        )

        return blended_latent

    def patch_sampler(self, sampler):
        """Patch sampler to inject latent blending."""
        import types

        original_sample_once = sampler.sample_once
        self._original_sample_once = original_sample_once
        hook = self

        def wrapped_sample_once(_sampler_self, model, x_t, t, t_prev, cond=None, **kwargs):
            # Call original sample_once
            result = original_sample_once(model, x_t, t, t_prev, cond, **kwargs)

            # Get timestep value (t is already a float in sample_once)
            t_norm = float(t)

            # Blend the pred_x_prev result
            if hasattr(result, 'pred_x_prev'):
                pred_x_prev = result.pred_x_prev
                if hasattr(pred_x_prev, 'feats') and hasattr(pred_x_prev, 'coords'):
                    blended = hook.blend_latent(pred_x_prev, t_norm)
                    result.pred_x_prev = blended

            return result

        sampler.sample_once = types.MethodType(wrapped_sample_once, sampler)

    def restore(self):
        """Restore original sampler."""
        # Note: This is tricky because we need to keep a reference to the sampler
        # For now, we'll just set the hook to None
        pass

