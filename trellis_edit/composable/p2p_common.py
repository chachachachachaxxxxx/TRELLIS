from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from trellis_edit.hooks import (
    KVBlendHook,
    KVBlendStageConfig,
    PromptToPromptHook,
    StageConfig,
)
from trellis_edit.inversion.rf_sampler import (
    build_denoise_t_pairs,
    build_inversion_t_pairs,
    resolve_inversion_steps,
)
from trellis_edit.samplers import SparseLatentBlendMask, blend_sparse_features
from trellis_edit.utils import build_image_token_metadata, resolve_patch_size
from trellis_edit.utils.uniedit_utils import coords_to_flat_indices


class P2PLatentBlendCommonMixin:
    """Shared helpers for composable P2P stages; blending stays stage-configurable."""

    _STAGE_DEFAULTS = {
        "ss": (1.0, 0.3, 1.0),
        "slat": (0.8, 0.0, 1.0),
    }

    @classmethod
    def _validate_stage_list(cls, stages: Iterable[str]) -> list[str]:
        if isinstance(stages, str):
            raise TypeError("enabled_stages must be an iterable of canonical stage names, not a string")

        validated: list[str] = []
        for stage_name in stages:
            if stage_name not in cls._STAGE_DEFAULTS:
                valid = ", ".join(sorted(cls._STAGE_DEFAULTS))
                raise ValueError(f"Unknown stage name: {stage_name!r}. Expected one of: {valid}")
            if stage_name in validated:
                raise ValueError(f"Duplicate stage name: {stage_name!r}")
            validated.append(stage_name)

        if not validated:
            raise ValueError("enabled_stages cannot be empty")

        return validated

    def _create_p2p_hook(
        self,
        *,
        inputs: Any,
        extra: Dict[str, Any],
        source_cond_dict: Dict[str, Any],
        edit_cond_dict: Dict[str, Any],
        enabled_stages: Iterable[str],
    ) -> Tuple[PromptToPromptHook, Dict[str, Any], Dict[str, StageConfig]]:
        patch_size = resolve_patch_size(inputs.source_image.size)
        token_meta = build_image_token_metadata(
            cond=edit_cond_dict["cond"],
            mask=inputs.mask_image,
            patch_size=patch_size,
            patch_coverage_threshold=extra.get("patch_coverage_threshold", 0.0),
        )
        enabled_stages = self._validate_stage_list(enabled_stages)
        stage_configs = {
            stage_name: self._build_stage_config(stage_name, extra)
            for stage_name in enabled_stages
        }
        hook = PromptToPromptHook(
            source_cond=source_cond_dict["cond"],
            edit_cond=edit_cond_dict["cond"],
            neg_cond=edit_cond_dict["neg_cond"],
            token_meta=token_meta,
            stage_configs=stage_configs,
        )
        return hook, token_meta, stage_configs

    def _build_stage_config(
        self,
        stage_name: str,
        extra: Dict[str, Any],
    ) -> StageConfig:
        if stage_name not in self._STAGE_DEFAULTS:
            valid = ", ".join(sorted(self._STAGE_DEFAULTS))
            raise ValueError(f"Unknown stage name: {stage_name!r}. Expected one of: {valid}")
        default_t_start, default_t_end, default_strength = self._STAGE_DEFAULTS[stage_name]
        return StageConfig(
            name=stage_name,
            enabled=True,
            t_start=extra.get(f"{stage_name}_t_start", default_t_start),
            t_end=extra.get(f"{stage_name}_t_end", default_t_end),
            strength=extra.get(f"{stage_name}_strength", default_strength),
        )

    def _create_kv_blend_hook(
        self,
        *,
        inputs: Any,
        cond_dict: Dict[str, Any],
        enabled_stages: Iterable[str],
        stage_masks: Dict[str, Dict[str, Any]],
        extra: Dict[str, Any],
    ) -> tuple[KVBlendHook, dict, Dict[str, KVBlendStageConfig]]:
        patch_size = resolve_patch_size(inputs.source_image.size)
        token_meta = build_image_token_metadata(
            cond=cond_dict["cond"],
            mask=inputs.mask_image,
            patch_size=patch_size,
            patch_coverage_threshold=0.0,
        )
        enabled_stages = self._validate_stage_list(enabled_stages)
        stage_configs = {
            stage_name: self._build_kv_blend_stage_config(stage_name, extra)
            for stage_name in enabled_stages
        }
        hook = KVBlendHook(
            neg_cond=cond_dict["neg_cond"],
            stage_configs=stage_configs,
            stage_masks=stage_masks,
        )
        return hook, token_meta, stage_configs

    @classmethod
    def _build_kv_blend_stage_config(
        cls,
        stage_name: str,
        extra: Dict[str, Any],
    ) -> KVBlendStageConfig:
        if stage_name not in cls._STAGE_DEFAULTS:
            valid = ", ".join(sorted(cls._STAGE_DEFAULTS))
            raise ValueError(f"Unknown stage name: {stage_name!r}. Expected one of: {valid}")
        return KVBlendStageConfig(
            name=stage_name,
            enabled=True,
            t_start=float(extra.get(f"{stage_name}_kv_t_start", 1.0)),
            t_end=float(extra.get(f"{stage_name}_kv_t_end", 0.0)),
            self_attention=bool(extra.get(f"{stage_name}_kv_self_attention", True)),
            cross_attention=bool(extra.get(f"{stage_name}_kv_cross_attention", True)),
        )

    @staticmethod
    def _build_cfg_interval(
        start: Any,
        end: Any,
        key_name: str,
    ) -> Optional[Tuple[float, float]]:
        if start is None and end is None:
            return None
        if start is None or end is None:
            raise ValueError(f"{key_name}_start and {key_name}_end must be set together.")

        start = float(start)
        end = float(end)
        if start > end:
            raise ValueError(f"{key_name} requires start <= end, got {start} > {end}")
        return (start, end)

    def _resolve_stage_sampler_params(
        self,
        stage_prefix: str,
        base_params: Optional[Dict[str, Any]],
        extra: Dict[str, Any],
    ) -> Dict[str, Any]:
        params = dict(base_params or {})

        strength = extra.get(f"{stage_prefix}_denoise_cfg_strength")
        if strength is not None:
            params["cfg_strength"] = float(strength)

        interval = self._build_cfg_interval(
            extra.get(f"{stage_prefix}_denoise_cfg_interval_start"),
            extra.get(f"{stage_prefix}_denoise_cfg_interval_end"),
            f"{stage_prefix}_denoise_cfg_interval",
        )
        if interval is not None:
            params["cfg_interval"] = interval

        return params

    def _resolve_inversion_cfg(
        self,
        stage_prefix: str,
        extra: Dict[str, Any],
    ) -> Dict[str, Any]:
        strength = extra.get(f"{stage_prefix}_inversion_cfg_strength")
        strength = 0.0 if strength is None else float(strength)

        interval = self._build_cfg_interval(
            extra.get(f"{stage_prefix}_inversion_cfg_interval_start"),
            extra.get(f"{stage_prefix}_inversion_cfg_interval_end"),
            f"{stage_prefix}_inversion_cfg_interval",
        )
        if interval is None:
            interval = (0.0, 1.0)

        return {
            "cfg_strength": strength,
            "cfg_interval": interval,
        }

    @staticmethod
    def _resolve_inversion_steps(
        stage_prefix: str,
        total_steps: int,
        extra: Dict[str, Any],
    ) -> int:
        return resolve_inversion_steps(
            total_steps=total_steps,
            inversion_steps=extra.get(f"{stage_prefix}_inversion_steps"),
        )

    @staticmethod
    def _resolve_predictor_corrector_steps(
        stage_prefix: str,
        extra: Dict[str, Any],
    ) -> int:
        raw_value = extra.get(f"{stage_prefix}_predictor_corrector_steps", 0)
        steps = int(raw_value)
        if steps < 0:
            raise ValueError(
                f"{stage_prefix}_predictor_corrector_steps must be >= 0, got {steps}"
            )
        return steps

    @staticmethod
    def _gaussian_kernel1d(
        sigma: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        sigma = float(sigma)
        if sigma <= 0.0:
            return torch.ones(1, dtype=dtype, device=device)

        radius = max(int(np.ceil(3.0 * sigma)), 1)
        offsets = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        kernel = torch.exp(-(offsets ** 2) / (2.0 * sigma * sigma))
        return kernel / kernel.sum()

    @classmethod
    def _gaussian_blur3d(
        cls,
        mask: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        kernel_1d = cls._gaussian_kernel1d(sigma, mask.dtype, mask.device)
        radius = kernel_1d.shape[0] // 2
        kernel_3d = kernel_1d[:, None, None] * kernel_1d[None, :, None] * kernel_1d[None, None, :]
        kernel_3d = kernel_3d.view(1, 1, *kernel_3d.shape)
        kernel_3d = kernel_3d.expand(mask.shape[1], 1, *kernel_3d.shape[-3:]).contiguous()
        return F.conv3d(mask, kernel_3d, padding=radius, groups=mask.shape[1])

    @classmethod
    def _gaussian_blur2d(
        cls,
        mask: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        kernel_1d = cls._gaussian_kernel1d(sigma, mask.dtype, mask.device)
        radius = kernel_1d.shape[0] // 2
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d = kernel_2d.view(1, 1, *kernel_2d.shape)
        kernel_2d = kernel_2d.expand(mask.shape[1], 1, *kernel_2d.shape[-2:]).contiguous()
        return F.conv2d(mask, kernel_2d, padding=radius, groups=mask.shape[1])

    @classmethod
    def _make_soft_mask3d(
        cls,
        mask: torch.Tensor,
        *,
        dilation: int = 2,
        sigma: float = 1.0,
    ) -> torch.Tensor:
        hard_mask = mask.float()
        dilation = int(dilation)
        sigma = float(sigma)

        if dilation > 0:
            kernel_size = 2 * dilation + 1
            base_mask = F.max_pool3d(
                hard_mask,
                kernel_size=kernel_size,
                stride=1,
                padding=dilation,
            )
        elif dilation < 0:
            erosion = -dilation
            kernel_size = 2 * erosion + 1
            base_mask = 1.0 - F.max_pool3d(
                1.0 - hard_mask,
                kernel_size=kernel_size,
                stride=1,
                padding=erosion,
            )
        else:
            base_mask = hard_mask

        if sigma > 0.0:
            soft_mask = cls._gaussian_blur3d(base_mask, sigma)
            if dilation >= 0:
                soft_mask = torch.maximum(hard_mask, soft_mask)
            else:
                soft_mask = torch.minimum(hard_mask, soft_mask)
        else:
            soft_mask = base_mask

        return soft_mask.clamp_(0.0, 1.0)

    @classmethod
    def _make_soft_mask2d(
        cls,
        mask: torch.Tensor,
        *,
        dilation: int = 2,
        sigma: float = 1.0,
    ) -> torch.Tensor:
        hard_mask = mask.float()
        dilation = int(dilation)
        sigma = float(sigma)

        if dilation > 0:
            kernel_size = 2 * dilation + 1
            base_mask = F.max_pool2d(
                hard_mask,
                kernel_size=kernel_size,
                stride=1,
                padding=dilation,
            )
        elif dilation < 0:
            erosion = -dilation
            kernel_size = 2 * erosion + 1
            base_mask = 1.0 - F.max_pool2d(
                1.0 - hard_mask,
                kernel_size=kernel_size,
                stride=1,
                padding=erosion,
            )
        else:
            base_mask = hard_mask

        if sigma > 0.0:
            soft_mask = cls._gaussian_blur2d(base_mask, sigma)
            if dilation >= 0:
                soft_mask = torch.maximum(hard_mask, soft_mask)
            else:
                soft_mask = torch.minimum(hard_mask, soft_mask)
        else:
            soft_mask = base_mask

        return soft_mask.clamp_(0.0, 1.0)

    def _build_slat_soft_edit_weights(
        self,
        edit_coords: torch.Tensor,
        preserve_coords: torch.Tensor,
        *,
        resolution: int = 64,
        dilation: int = 2,
        sigma: float = 1.0,
    ) -> torch.Tensor:
        if preserve_coords.shape[0] == 0:
            return torch.zeros(0, 1, dtype=torch.float32, device=edit_coords.device)

        preserve_codes = coords_to_flat_indices(preserve_coords, resolution)
        edit_codes = coords_to_flat_indices(edit_coords, resolution)
        coords_edit = edit_coords[~torch.isin(edit_codes, preserve_codes)]

        if coords_edit.shape[0] == 0:
            return torch.zeros(
                preserve_coords.shape[0],
                1,
                dtype=torch.float32,
                device=edit_coords.device,
            )

        edit_mask = torch.zeros(
            1,
            1,
            resolution,
            resolution,
            resolution,
            dtype=torch.float32,
            device=edit_coords.device,
        )
        edit_mask[0, 0, coords_edit[:, 1], coords_edit[:, 2], coords_edit[:, 3]] = 1.0
        soft_mask = self._make_soft_mask3d(
            edit_mask,
            dilation=dilation,
            sigma=sigma,
        )
        preserve_weights = soft_mask[
            0,
            0,
            preserve_coords[:, 1].long(),
            preserve_coords[:, 2].long(),
            preserve_coords[:, 3].long(),
        ]
        return preserve_weights.unsqueeze(1).contiguous().float()

    @classmethod
    def _build_cross_kv_token_mask(
        cls,
        token_meta: dict,
        *,
        soft_mask_enabled: bool = False,
        soft_mask_dilation: int = 2,
        soft_mask_sigma: float = 1.0,
    ) -> torch.Tensor:
        token_mask = torch.zeros(int(token_meta["total_tokens"]), dtype=torch.float32)
        token_mask[token_meta["special_token_indices"]] = 1.0
        prefix_token_count = int(token_meta["prefix_token_count"])
        if soft_mask_enabled:
            patch_grid = torch.tensor(
                token_meta["edited_patch_grid"],
                dtype=torch.float32,
            ).view(1, 1, *token_meta["patch_grid_size"])
            patch_mask = cls._make_soft_mask2d(
                patch_grid,
                dilation=soft_mask_dilation,
                sigma=soft_mask_sigma,
            )
            token_mask[prefix_token_count:] = patch_mask.reshape(-1).clamp_(0.0, 1.0)
        else:
            token_mask[token_meta["edited_token_indices"]] = 1.0
        return token_mask.view(1, -1).contiguous()

    def _build_ss_self_kv_token_mask(
        self,
        mask_coords: torch.Tensor,
        *,
        resolution: int,
        latent_resolution: int,
        hard_mask_mode: str = "edit_all",
        soft_mask_enabled: bool = False,
        soft_mask_dilation: int = 2,
        soft_mask_sigma: float = 1.0,
    ) -> torch.Tensor:
        latent_mask = self._build_ss_latent_mask(
            mask_coords,
            resolution=resolution,
            channels=1,
            latent_resolution=latent_resolution,
            hard_mask_mode=hard_mask_mode,
            soft_mask_enabled=soft_mask_enabled,
            soft_mask_dilation=soft_mask_dilation,
            soft_mask_sigma=soft_mask_sigma,
        )
        return latent_mask[:, 0].reshape(1, -1).contiguous().float().cpu()

    @staticmethod
    def _downsample_sparse_coords(
        coords: torch.Tensor,
        *,
        factor: tuple[int, int, int] = (2, 2, 2),
    ) -> torch.Tensor:
        if coords.shape[0] == 0:
            return torch.zeros(0, 4, dtype=torch.int32, device=coords.device)

        if coords.shape[1] == 3:
            coords = torch.cat(
                [torch.zeros(coords.shape[0], 1, dtype=torch.int32, device=coords.device), coords.int()],
                dim=1,
            )
        else:
            coords = coords.int()

        coords_edit = coords.clone()
        coord_edit = list(coords_edit.unbind(dim=-1))
        for axis, divisor in enumerate(factor):
            coord_edit[axis + 1] = coord_edit[axis + 1] // divisor

        max_vals = [int(coord_edit[axis + 1].max().item()) + 1 for axis in range(3)]
        offsets = torch.cumprod(torch.tensor(max_vals[::-1]), 0).tolist()[::-1] + [1]
        coord_code = sum([coord * offset for coord, offset in zip(coord_edit, offsets)])
        coord_code = coord_code.unique()
        coords = torch.stack(
            [coord_code // offsets[0]]
            + [(coord_code // offsets[idx + 1]) % max_vals[idx] for idx in range(3)],
            dim=-1,
        )
        return coords.contiguous().to(device=coords_edit.device, dtype=torch.int32)

    def _build_slat_self_kv_mask(
        self,
        edit_coords: torch.Tensor,
        preserve_coords: torch.Tensor,
        *,
        resolution: int,
        soft_mask_enabled: bool = False,
        soft_mask_dilation: int = 2,
        soft_mask_sigma: float = 1.0,
    ) -> torch.Tensor | SparseLatentBlendMask:
        preserve_coords_ds = self._downsample_sparse_coords(preserve_coords)
        if preserve_coords_ds.shape[0] == 0 or not soft_mask_enabled:
            return preserve_coords_ds

        edit_coords_ds = self._downsample_sparse_coords(edit_coords)
        latent_resolution = max(int(resolution) // 2, 1)
        latent_dilation = int(np.sign(soft_mask_dilation) * np.ceil(abs(float(soft_mask_dilation)) / 2.0))
        latent_sigma = float(soft_mask_sigma) / 2.0
        edit_weights = self._build_slat_soft_edit_weights(
            edit_coords_ds,
            preserve_coords_ds,
            resolution=latent_resolution,
            dilation=latent_dilation,
            sigma=latent_sigma,
        )
        return SparseLatentBlendMask(
            coords=preserve_coords_ds,
            edit_weights=edit_weights,
        )

    @staticmethod
    def _sample_device(sample) -> torch.device:
        if hasattr(sample, "device"):
            return sample.device
        return sample.coords.device

    def _predict_with_optional_cfg(
        self,
        model,
        sample,
        t_value: float,
        cond_dict: Dict[str, Any],
        cfg_strength: float,
        cfg_interval: Tuple[float, float],
        *,
        hook: Any | None = None,
        phase: str | None = None,
        cache_t_value: float | None = None,
    ):
        device = self._sample_device(sample)
        t_tensor = torch.tensor([1000 * t_value], device=device, dtype=torch.float32)
        if hook is not None and hasattr(hook, "set_phase"):
            hook.set_phase(phase, cache_t_norm=cache_t_value)
        pred = model(sample, t_tensor, cond_dict["cond"])
        if cfg_strength <= 0.0 or not (cfg_interval[0] <= t_value <= cfg_interval[1]):
            return pred

        neg_pred = model(sample, t_tensor, cond_dict["neg_cond"])
        return pred + cfg_strength * (pred - neg_pred)

    def _invert_sample(
        self,
        model,
        sample,
        cond_dict,
        steps: int,
        rescale_t: float,
        stage_prefix: str,
        config: Any,
        inversion_mode: str,
        return_terminal_noise: bool = False,
        hook: Any | None = None,
    ) -> Dict[str, Any] | tuple[Dict[str, Any], Any]:
        if inversion_mode not in {"simple", "rf_solver"}:
            raise ValueError(f"Unknown inversion_mode: {inversion_mode}")

        extra = config.extra_params or {}
        inversion_cfg = self._resolve_inversion_cfg(stage_prefix, extra)
        predictor_corrector_steps = self._resolve_predictor_corrector_steps(stage_prefix, extra)
        inversion_steps = self._resolve_inversion_steps(stage_prefix, steps, extra)
        t_pairs, _ = build_inversion_t_pairs(
            steps=steps,
            rescale_t=rescale_t,
            inversion_steps=inversion_steps,
        )

        latent_cache = {}

        with torch.inference_mode():
            for i, (t_curr, t_next) in enumerate(t_pairs):
                sample = self._sample_with_solver(
                    model,
                    sample,
                    t_curr,
                    t_next,
                    cond_dict,
                    cfg_strength=inversion_cfg["cfg_strength"],
                    cfg_interval=inversion_cfg["cfg_interval"],
                    solver_mode=inversion_mode,
                    predictor_corrector_steps=predictor_corrector_steps,
                    hook=hook,
                    phase="inversion",
                )
                latent_cache[f"{float(t_next)}"] = sample.cpu()

                if i % 5 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if return_terminal_noise:
            return latent_cache, sample
        return latent_cache

    def _sample_with_solver(
        self,
        model,
        sample,
        t_curr: float,
        t_next: float,
        cond_dict: Dict[str, Any],
        *,
        cfg_strength: float,
        cfg_interval: Tuple[float, float],
        solver_mode: str,
        predictor_corrector_steps: int,
        hook: Any | None = None,
        phase: str | None = None,
    ):
        if solver_mode not in {"simple", "rf_solver"}:
            raise ValueError(f"Unknown solver_mode: {solver_mode}")

        hook_cache_t_value = None
        if phase == "inversion" and solver_mode == "simple":
            # Keep 1st-order KV cache aligned with latent_cache[t_next], so the
            # first denoising step can read the cached 1.0 state.
            hook_cache_t_value = t_next

        pred_v = self._predict_with_optional_cfg(
            model,
            sample,
            t_curr,
            cond_dict,
            cfg_strength=cfg_strength,
            cfg_interval=cfg_interval,
            hook=hook,
            phase=phase,
            cache_t_value=hook_cache_t_value,
        )
        dt = t_next - t_curr
        if solver_mode == "simple":
            next_sample = sample + dt * pred_v
        else:
            # Use one shared RF step for inversion and denoising. With
            # first_order := (pred_mid - pred_v) / (0.5 * dt), the Taylor
            # correction enters with a plus sign.
            sample_mid = sample + 0.5 * dt * pred_v
            pred_v_mid = self._predict_with_optional_cfg(
                model,
                sample_mid,
                t_curr + 0.5 * dt,
                cond_dict,
                cfg_strength=cfg_strength,
                cfg_interval=cfg_interval,
                hook=hook,
                phase=phase,
                cache_t_value=hook_cache_t_value,
            )
            first_order = (pred_v_mid - pred_v) / (0.5 * dt)
            next_sample = sample + dt * pred_v + 0.5 * (dt ** 2) * first_order

        if predictor_corrector_steps <= 0:
            return next_sample

        # UniInv-style correction is an inversion-only refinement axis:
        # start from the predictor result at t_next, then repeatedly
        # re-evaluate a denoising-like velocity at t_next and update the
        # same step from the original low-noise state.
        corrected_sample = next_sample
        for _ in range(predictor_corrector_steps):
            corrected_v = self._predict_with_optional_cfg(
                model,
                corrected_sample,
                t_next,
                cond_dict,
                cfg_strength=cfg_strength,
                cfg_interval=cfg_interval,
                hook=hook,
                phase=phase,
                cache_t_value=t_next,
            )
            corrected_sample = sample + dt * corrected_v
        return corrected_sample

    def _prepare_ss_latent_from_coords(
        self,
        pipeline,
        source_coords: torch.Tensor,
        source_cond_dict: Dict[str, Any],
        inversion_mode: str,
        config: Any,
        resolution: int,
        hook: Any | None = None,
    ) -> Dict[str, torch.Tensor]:
        latent_cache, _ = self._prepare_ss_latent_and_terminal_from_coords(
            pipeline,
            source_coords,
            source_cond_dict,
            inversion_mode,
            config,
            resolution,
            hook=hook,
        )
        return latent_cache

    def _prepare_ss_latent_and_terminal_from_coords(
        self,
        pipeline,
        source_coords: torch.Tensor,
        source_cond_dict: Dict[str, Any],
        inversion_mode: str,
        config: Any,
        resolution: int,
        hook: Any | None = None,
    ) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        if inversion_mode == "none":
            raise ValueError(
                "inversion_mode='none' is not valid. "
                "Must do inversion to get per-step latent cache. "
                "Use 'simple' or 'rf_solver'."
            )

        from trellis_edit.preprocess.asset_3d import coords_to_voxel

        voxel = coords_to_voxel(source_coords, pipeline.device, resolution)
        z_s = pipeline.models["sparse_structure_encoder"](voxel)
        extra = config.extra_params or {}
        stage_params = self._resolve_stage_sampler_params(
            "ss",
            config.sparse_structure_sampler_params,
            extra,
        )
        sampler_params = {**getattr(pipeline, "sparse_structure_sampler_params", {}), **stage_params}
        return self._invert_sample(
            model=pipeline.models["sparse_structure_flow_model"],
            sample=z_s,
            cond_dict=source_cond_dict,
            steps=int(sampler_params.get("steps", 25)),
            rescale_t=float(sampler_params.get("rescale_t", 3.0)),
            stage_prefix="ss",
            config=config,
            inversion_mode=inversion_mode,
            return_terminal_noise=True,
            hook=hook,
        )

    def _build_ss_latent_mask(
        self,
        mask_coords: torch.Tensor,
        *,
        resolution: int = 64,
        channels: int = 8,
        latent_resolution: int = 16,
        hard_mask_mode: str = "edit_all",
        soft_mask_enabled: bool = False,
        soft_mask_dilation: int = 2,
        soft_mask_sigma: float = 1.0,
    ) -> torch.Tensor:
        if resolution % latent_resolution != 0:
            raise ValueError(
                f"SS mask requires resolution divisible by latent_resolution, got "
                f"{resolution} and {latent_resolution}"
            )

        voxel_mask = torch.zeros(
            1,
            1,
            resolution,
            resolution,
            resolution,
            dtype=torch.float32,
            device=mask_coords.device,
        )

        if mask_coords.shape[0] > 0:
            coords_int = mask_coords[:, 1:].long() if mask_coords.shape[1] == 4 else mask_coords.long()
            voxel_mask[0, 0, coords_int[:, 0], coords_int[:, 1], coords_int[:, 2]] = 1.0

        pooling_stride = resolution // latent_resolution
        if soft_mask_enabled:
            voxel_mask = self._make_soft_mask3d(
                voxel_mask,
                dilation=soft_mask_dilation,
                sigma=soft_mask_sigma,
            )
            pooled_mask = F.avg_pool3d(
                voxel_mask,
                kernel_size=pooling_stride,
                stride=pooling_stride,
            )
        else:
            if hard_mask_mode == "edit_any":
                pooled_mask = F.max_pool3d(
                    voxel_mask,
                    kernel_size=pooling_stride,
                    stride=pooling_stride,
                )
            elif hard_mask_mode == "edit_all":
                pooled_mask = F.avg_pool3d(
                    voxel_mask,
                    kernel_size=pooling_stride,
                    stride=pooling_stride,
                ).eq(1.0).to(dtype=voxel_mask.dtype)
            else:
                raise ValueError(f"Unknown hard_mask_mode: {hard_mask_mode}")
        if channels == 1:
            return pooled_mask.contiguous()
        return pooled_mask.expand(1, channels, latent_resolution, latent_resolution, latent_resolution).contiguous()

    def _denoise_ss_with_step_blend(
        self,
        pipeline,
        edit_cond_dict: Dict[str, Any],
        config: Any,
        *,
        initial_sample: Optional[torch.Tensor] = None,
        source_latent_cache: Optional[Dict[str, torch.Tensor]] = None,
        latent_mask: Optional[torch.Tensor] = None,
        blend_strength: float = 1.0,
        solver_mode: str = "simple",
        verbose: bool = False,
        hook: Any | None = None,
    ) -> torch.Tensor:
        if solver_mode not in {"simple", "rf_solver"}:
            raise ValueError(f"Unknown solver_mode: {solver_mode}")

        extra = config.extra_params or {}
        stage_params = self._resolve_stage_sampler_params(
            "ss",
            config.sparse_structure_sampler_params,
            extra,
        )
        sampler_params = {**getattr(pipeline, "sparse_structure_sampler_params", {}), **stage_params}

        flow_model = pipeline.models["sparse_structure_flow_model"]
        steps = sampler_params.get("steps", 25)
        rescale_t = sampler_params.get("rescale_t", 3.0)
        cfg_strength = float(sampler_params.get("cfg_strength", 0.0))
        cfg_interval = tuple(sampler_params.get("cfg_interval", (0.0, 1.0)))
        blend_strength = float(blend_strength)
        denoise_start_step = (
            self._resolve_inversion_steps("ss", steps, extra)
            if initial_sample is not None
            else int(steps)
        )
        t_pairs, _ = build_denoise_t_pairs(
            steps=steps,
            rescale_t=rescale_t,
            start_step=denoise_start_step,
        )

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if initial_sample is None:
            sample = torch.randn(
                config.num_samples,
                flow_model.in_channels,
                flow_model.resolution,
                flow_model.resolution,
                flow_model.resolution,
                device=pipeline.device,
            )
        else:
            sample = initial_sample.to(device=pipeline.device).clone()
            if sample.shape[0] == 1 and config.num_samples > 1:
                sample = sample.repeat(config.num_samples, 1, 1, 1, 1)
            elif sample.shape[0] != config.num_samples:
                raise ValueError(
                    "initial_sample batch size does not match requested num_samples: "
                    f"{sample.shape[0]} vs {config.num_samples}"
                )
        mask = None if latent_mask is None else latent_mask.to(sample.device, sample.dtype)

        if verbose:
            init_mode = "inverted terminal noise" if initial_sample is not None else "random noise"
            print(
                "[Denoising] Starting SS denoising "
                f"(total_steps={steps}, start_step={denoise_start_step}, init={init_mode})"
            )

        with torch.no_grad():
            for i, (t_curr, t_next) in enumerate(t_pairs):

                if source_latent_cache is not None and mask is not None:
                    source_key = f"{float(t_curr)}"
                    if source_key in source_latent_cache:
                        source_latent = source_latent_cache[source_key].to(sample.device, sample.dtype)
                        sample = mask * sample + (1 - mask) * (
                            blend_strength * source_latent + (1 - blend_strength) * sample
                        )

                sample = self._sample_with_solver(
                    flow_model,
                    sample,
                    t_curr,
                    t_next,
                    edit_cond_dict,
                    cfg_strength=cfg_strength,
                    cfg_interval=cfg_interval,
                    solver_mode=solver_mode,
                    predictor_corrector_steps=0,
                    hook=hook,
                    phase="denoise",
                )

                if i % 5 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return sample

    def _denoise_slat_with_step_blend(
        self,
        pipeline,
        edit_cond_dict: Dict[str, Any],
        config: Any,
        *,
        edit_coords,
        initial_sample=None,
        source_latent_cache: Optional[Dict[str, Any]] = None,
        latent_mask: Optional[torch.Tensor | SparseLatentBlendMask] = None,
        solver_mode: str = "simple",
        verbose: bool = False,
        hook: Any | None = None,
    ):
        if solver_mode not in {"simple", "rf_solver"}:
            raise ValueError(f"Unknown solver_mode: {solver_mode}")
        if int(config.num_samples) != 1:
            raise RuntimeError("Custom SLAT denoising currently supports num_samples=1 only.")

        extra = config.extra_params or {}
        stage_params = self._resolve_stage_sampler_params(
            "slat",
            config.slat_sampler_params,
            extra,
        )
        sampler_params = {**getattr(pipeline, "slat_sampler_params", {}), **stage_params}

        flow_model = pipeline.models["slat_flow_model"]
        steps = sampler_params.get("steps", 25)
        rescale_t = sampler_params.get("rescale_t", 3.0)
        cfg_strength = float(sampler_params.get("cfg_strength", 0.0))
        cfg_interval = tuple(sampler_params.get("cfg_interval", (0.0, 1.0)))
        denoise_start_step = (
            self._resolve_inversion_steps("slat", steps, extra)
            if initial_sample is not None
            else int(steps)
        )
        t_pairs, _ = build_denoise_t_pairs(
            steps=steps,
            rescale_t=rescale_t,
            start_step=denoise_start_step,
        )

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if initial_sample is None:
            from trellis.modules.sparse.basic import SparseTensor

            sample = SparseTensor(
                feats=torch.randn(edit_coords.shape[0], flow_model.in_channels, device=pipeline.device),
                coords=edit_coords.to(device=pipeline.device),
            )
        else:
            sample = initial_sample.to(device=pipeline.device)

        if verbose:
            init_mode = "projected terminal noise" if initial_sample is not None else "random noise"
            print(
                "[Denoising] Starting SLAT denoising "
                f"(total_steps={steps}, start_step={denoise_start_step}, init={init_mode})"
            )

        with torch.no_grad():
            for i, (t_curr, t_next) in enumerate(t_pairs):

                if source_latent_cache is not None and latent_mask is not None:
                    source_key = f"{float(t_curr)}"
                    if source_key in source_latent_cache:
                        source_latent = source_latent_cache[source_key]
                        sample, _ = blend_sparse_features(sample, source_latent, latent_mask)

                sample = self._sample_with_solver(
                    flow_model,
                    sample,
                    t_curr,
                    t_next,
                    edit_cond_dict,
                    cfg_strength=cfg_strength,
                    cfg_interval=cfg_interval,
                    solver_mode=solver_mode,
                    predictor_corrector_steps=0,
                    hook=hook,
                    phase="denoise",
                )

                if i % 5 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return sample
