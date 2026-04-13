from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from trellis_edit.hooks import PromptToPromptHook, StageConfig
from trellis_edit.utils import build_image_token_metadata, resolve_patch_size


class P2PLatentBlendCommonMixin:
    """Shared helpers for composable P2P latent-blend stages."""

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
            query_chunk=extra.get("query_chunk", 1024),
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
    ):
        device = self._sample_device(sample)
        t_tensor = torch.tensor([1000 * t_value], device=device, dtype=torch.float32)
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
        stage_prefix: str,
        config: Any,
        inversion_mode: str,
    ) -> Dict[str, Any]:
        if inversion_mode not in {"simple", "rf_solver"}:
            raise ValueError(f"Unknown inversion_mode: {inversion_mode}")

        extra = config.extra_params or {}
        inversion_cfg = self._resolve_inversion_cfg(stage_prefix, extra)
        t_seq = np.linspace(0, 1, steps + 1)
        t_seq = 3.0 * t_seq / (1 + 2.0 * t_seq)

        latent_cache = {}

        with torch.inference_mode():
            for i in range(steps):
                t_curr = t_seq[i]
                t_next = t_seq[i + 1]
                latent_cache[f"{t_next}"] = sample.cpu()

                pred_v = self._predict_with_optional_cfg(
                    model,
                    sample,
                    t_curr,
                    cond_dict,
                    cfg_strength=inversion_cfg["cfg_strength"],
                    cfg_interval=inversion_cfg["cfg_interval"],
                )

                if inversion_mode == "simple":
                    sample = sample + (t_next - t_curr) * pred_v
                    continue

                dt = t_next - t_curr
                sample_mid = sample + (dt / 2) * pred_v
                pred_v_mid = self._predict_with_optional_cfg(
                    model,
                    sample_mid,
                    t_curr + dt / 2,
                    cond_dict,
                    cfg_strength=inversion_cfg["cfg_strength"],
                    cfg_interval=inversion_cfg["cfg_interval"],
                )
                first_order = (pred_v_mid - pred_v) / (dt / 2)
                sample = sample + dt * pred_v + 0.5 * (dt ** 2) * first_order

                if i % 5 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return latent_cache

    def _prepare_ss_latent_from_coords(
        self,
        pipeline,
        source_coords: torch.Tensor,
        source_cond_dict: Dict[str, Any],
        inversion_mode: str,
        config: Any,
        resolution: int,
    ) -> Dict[str, torch.Tensor]:
        if inversion_mode == "none":
            raise ValueError(
                "inversion_mode='none' is not valid. "
                "Must do inversion to get per-step latent cache. "
                "Use 'simple' or 'rf_solver'."
            )

        from trellis_edit.preprocess.asset_3d import coords_to_voxel

        voxel = coords_to_voxel(source_coords, pipeline.device, resolution)
        z_s = pipeline.models["sparse_structure_encoder"](voxel)
        steps = (config.sparse_structure_sampler_params or {}).get("steps", 25)
        return self._invert_sample(
            model=pipeline.models["sparse_structure_flow_model"],
            sample=z_s,
            cond_dict=source_cond_dict,
            steps=steps,
            stage_prefix="ss",
            config=config,
            inversion_mode=inversion_mode,
        )

    def _build_ss_latent_mask(
        self,
        mask_coords: torch.Tensor,
        *,
        resolution: int = 64,
        channels: int = 8,
        latent_resolution: int = 16,
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

        pooled_mask = F.max_pool3d(
            voxel_mask,
            kernel_size=resolution // latent_resolution,
            stride=resolution // latent_resolution,
        )
        if channels == 1:
            return pooled_mask.contiguous()
        return pooled_mask.expand(1, channels, latent_resolution, latent_resolution, latent_resolution).contiguous()

    def _denoise_ss_with_step_blend(
        self,
        pipeline,
        edit_cond_dict: Dict[str, Any],
        config: Any,
        *,
        source_latent_cache: Optional[Dict[str, torch.Tensor]] = None,
        latent_mask: Optional[torch.Tensor] = None,
        blend_strength: float = 1.0,
        verbose: bool = False,
    ) -> torch.Tensor:
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

        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        sample = torch.randn(
            config.num_samples,
            flow_model.in_channels,
            flow_model.resolution,
            flow_model.resolution,
            flow_model.resolution,
            device=pipeline.device,
        )
        mask = None if latent_mask is None else latent_mask.to(sample.device, sample.dtype)

        if verbose:
            print(f"[Denoising] Starting SS denoising ({steps} steps)")

        with torch.no_grad():
            for i in range(steps):
                t_curr = t_seq[i]
                t_next = t_seq[i + 1]

                pred_v = self._predict_with_optional_cfg(
                    flow_model,
                    sample,
                    t_curr,
                    edit_cond_dict,
                    cfg_strength=cfg_strength,
                    cfg_interval=cfg_interval,
                )
                sample = sample + (t_next - t_curr) * pred_v

                if source_latent_cache is not None and mask is not None:
                    source_key = f"{t_next}"
                    if source_key in source_latent_cache:
                        source_latent = source_latent_cache[source_key].to(sample.device, sample.dtype)
                        sample = mask * sample + (1 - mask) * (
                            blend_strength * source_latent + (1 - blend_strength) * sample
                        )

                if i % 5 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return sample
