from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple
from dataclasses import dataclass

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
from trellis_edit.composable.runtime import (
    PredictorStepResult,
    RefinementResult,
    SolverStepResult,
    SourceTrace,
    StepContext,
    TraceEvalState,
)
from trellis_edit.samplers import SparseLatentBlendMask, blend_sparse_features
from trellis_edit.utils import build_image_token_metadata, resolve_patch_size
from trellis_edit.utils.uniedit_utils import coords_to_flat_indices


@dataclass(frozen=True)
class SolverEvalSpec:
    actual_t: float
    logical_t: float
    eval_idx: int


class ControlledDenoiseCommonMixin:
    """Shared helpers for controlled-denoise stages with optional stage controls."""

    _STAGE_DEFAULTS = {
        "ss": (1.0, 0.3, 1.0),
        "slat": (0.8, 0.0, 1.0),
    }
    _SUPPORTED_SOLVER_MODES = {"simple", "rf_solver", "voxhammer_rf_solver"}
    _RF_FAMILY_SOLVER_MODES = {"rf_solver", "voxhammer_rf_solver"}
    _SUPPORTED_KV_SOURCE_BUNDLE_MODES = {"solver_aligned", "external_voxhammer"}

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

    @classmethod
    def _validate_solver_mode(cls, solver_mode: str) -> str:
        if solver_mode not in cls._SUPPORTED_SOLVER_MODES:
            valid = ", ".join(sorted(cls._SUPPORTED_SOLVER_MODES))
            raise ValueError(f"Unknown solver_mode: {solver_mode!r}. Expected one of: {valid}")
        return solver_mode

    @classmethod
    def _is_rf_family_solver(cls, solver_mode: str) -> bool:
        cls._validate_solver_mode(solver_mode)
        return solver_mode in cls._RF_FAMILY_SOLVER_MODES

    @classmethod
    def _solver_display_name(cls, solver_mode: str) -> str:
        cls._validate_solver_mode(solver_mode)
        if solver_mode == "simple":
            return "simple Euler"
        if solver_mode == "rf_solver":
            return "RF-Solver"
        return "VoxHammer RF-Solver"

    @classmethod
    def _validate_kv_source_bundle_mode(cls, bundle_mode: str) -> str:
        if bundle_mode not in cls._SUPPORTED_KV_SOURCE_BUNDLE_MODES:
            valid = ", ".join(sorted(cls._SUPPORTED_KV_SOURCE_BUNDLE_MODES))
            raise ValueError(
                f"Unknown kv source bundle mode: {bundle_mode!r}. Expected one of: {valid}"
            )
        return bundle_mode

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
    def _resolve_refinement_steps(
        stage_prefix: str,
        extra: Dict[str, Any],
    ) -> int:
        raw_value = extra.get(f"{stage_prefix}_refinement_steps", 0)
        steps = int(raw_value)
        if steps < 0:
            raise ValueError(
                f"{stage_prefix}_refinement_steps must be >= 0, got {steps}"
            )
        return steps

    @classmethod
    def _resolve_refinement_solver_mode(
        cls,
        stage_prefix: str,
        extra: Dict[str, Any],
        *,
        predictor_solver_mode: str,
    ) -> str:
        raw_value = str(extra.get(f"{stage_prefix}_refinement_solver_mode", predictor_solver_mode))
        if raw_value == "inherit":
            raw_value = predictor_solver_mode
        return cls._validate_solver_mode(raw_value)

    @classmethod
    def _resolve_kv_source_bundle_mode(
        cls,
        stage_prefix: str,
        extra: Dict[str, Any],
    ) -> str:
        raw_value = str(extra.get(f"{stage_prefix}_kv_source_bundle_mode", "solver_aligned"))
        return cls._validate_kv_source_bundle_mode(raw_value)

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
        source_coords: torch.Tensor | None = None,
        resolution: int,
        latent_resolution: int,
        hard_mask_mode: str = "edit_all",
        edit_region_mode: str = "mask_only",
        soft_mask_enabled: bool = False,
        soft_mask_dilation: int = 2,
        soft_mask_sigma: float = 1.0,
    ) -> torch.Tensor:
        latent_mask = self._build_ss_latent_mask(
            mask_coords,
            source_coords=source_coords,
            resolution=resolution,
            channels=1,
            latent_resolution=latent_resolution,
            hard_mask_mode=hard_mask_mode,
            edit_region_mode=edit_region_mode,
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

    @staticmethod
    def _resolve_solver_logical_t(
        *,
        phase: str | None,
        t_curr: float,
        t_next: float,
    ) -> float:
        if phase == "inversion":
            return float(t_next)
        return float(t_curr)

    def _build_predictor_eval_specs(
        self,
        *,
        step_ctx: StepContext,
        solver_mode: str,
    ) -> list[SolverEvalSpec]:
        self._validate_solver_mode(solver_mode)

        dt = float(step_ctx.t_next - step_ctx.t_curr)
        eval_specs = [
            SolverEvalSpec(
                actual_t=float(step_ctx.t_curr),
                logical_t=float(step_ctx.logical_t),
                eval_idx=1,
            )
        ]
        if self._is_rf_family_solver(solver_mode):
            eval_specs.append(
                SolverEvalSpec(
                    actual_t=float(step_ctx.t_curr + 0.5 * dt),
                    logical_t=float(step_ctx.logical_t),
                    eval_idx=2,
                )
            )
        return eval_specs

    def _build_refinement_eval_specs(
        self,
        *,
        step_ctx: StepContext,
        predictor_eval_count: int,
        refinement_steps: int,
        refinement_solver_mode: str,
    ) -> list[SolverEvalSpec]:
        self._validate_solver_mode(refinement_solver_mode)

        eval_specs: list[SolverEvalSpec] = []
        next_eval_idx = int(predictor_eval_count) + 1
        midpoint_t = float(step_ctx.t_curr + 0.5 * (step_ctx.t_next - step_ctx.t_curr))
        for _ in range(int(refinement_steps)):
            eval_specs.append(
                SolverEvalSpec(
                    actual_t=float(step_ctx.t_next),
                    logical_t=float(step_ctx.logical_t),
                    eval_idx=next_eval_idx,
                )
            )
            next_eval_idx += 1
            if self._is_rf_family_solver(refinement_solver_mode):
                eval_specs.append(
                    SolverEvalSpec(
                        actual_t=midpoint_t,
                        logical_t=float(step_ctx.logical_t),
                        eval_idx=next_eval_idx,
                    )
                )
                next_eval_idx += 1
        return eval_specs

    def _predict_with_optional_cfg(
        self,
        model,
        sample,
        cond_dict: Dict[str, Any],
        cfg_strength: float,
        cfg_interval: Tuple[float, float],
        *,
        eval_spec: SolverEvalSpec,
        hook: Any | None = None,
        phase: str | None = None,
    ):
        actual_t = float(eval_spec.actual_t)
        logical_t = float(eval_spec.logical_t)
        device = self._sample_device(sample)
        t_tensor = torch.tensor([1000 * actual_t], device=device, dtype=torch.float32)
        if hook is not None and hasattr(hook, "set_eval_context"):
            hook.set_eval_context(phase, eval_spec=eval_spec)
        try:
            pred = model(sample, t_tensor, cond_dict["cond"])
            if cfg_strength <= 0.0 or not (cfg_interval[0] <= logical_t <= cfg_interval[1]):
                return pred

            neg_pred = model(sample, t_tensor, cond_dict["neg_cond"])
            return pred + cfg_strength * (pred - neg_pred)
        finally:
            if hook is not None and hasattr(hook, "set_eval_context"):
                hook.set_eval_context(None, eval_spec=None)

    def _predictor_step(
        self,
        model,
        sample,
        cond_dict: Dict[str, Any],
        *,
        step_ctx: StepContext,
        cfg_strength: float,
        cfg_interval: Tuple[float, float],
        solver_mode: str,
        hook: Any | None = None,
    ) -> PredictorStepResult:
        self._validate_solver_mode(solver_mode)

        eval_specs = self._build_predictor_eval_specs(
            step_ctx=step_ctx,
            solver_mode=solver_mode,
        )
        eval_states = [
            TraceEvalState(
                actual_t=float(eval_specs[0].actual_t),
                logical_t=float(eval_specs[0].logical_t),
                eval_idx=int(eval_specs[0].eval_idx),
                sample=sample,
            )
        ]
        pred_v = self._predict_with_optional_cfg(
            model,
            sample,
            cond_dict,
            cfg_strength=cfg_strength,
            cfg_interval=cfg_interval,
            eval_spec=eval_specs[0],
            hook=hook,
            phase=step_ctx.phase,
        )
        dt = float(step_ctx.t_next - step_ctx.t_curr)
        if solver_mode == "simple":
            return PredictorStepResult(
                x_pred=sample + dt * pred_v,
                eval_states=tuple(eval_states),
            )

        sample_mid = sample + 0.5 * dt * pred_v
        eval_states.append(
            TraceEvalState(
                actual_t=float(eval_specs[1].actual_t),
                logical_t=float(eval_specs[1].logical_t),
                eval_idx=int(eval_specs[1].eval_idx),
                sample=sample_mid,
            )
        )
        pred_v_mid = self._predict_with_optional_cfg(
            model,
            sample_mid,
            cond_dict,
            cfg_strength=cfg_strength,
            cfg_interval=cfg_interval,
            eval_spec=eval_specs[1],
            hook=hook,
            phase=step_ctx.phase,
        )
        first_order = (pred_v_mid - pred_v) / (0.5 * dt)
        if solver_mode == "rf_solver":
            x_pred = sample + dt * pred_v + 0.5 * (dt ** 2) * first_order
        else:
            x_pred = sample + dt * pred_v - 0.5 * (dt ** 2) * first_order
        return PredictorStepResult(
            x_pred=x_pred,
            eval_states=tuple(eval_states),
        )

    def _refine_step(
        self,
        model,
        sample,
        predictor_result: PredictorStepResult,
        cond_dict: Dict[str, Any],
        *,
        step_ctx: StepContext,
        cfg_strength: float,
        cfg_interval: Tuple[float, float],
        solver_mode: str,
        refinement_steps: int,
        refinement_solver_mode: str,
        hook: Any | None = None,
    ) -> RefinementResult:
        self._validate_solver_mode(solver_mode)
        self._validate_solver_mode(refinement_solver_mode)

        if refinement_steps <= 0:
            return RefinementResult(x_next=predictor_result.x_pred)

        eval_specs = self._build_refinement_eval_specs(
            step_ctx=step_ctx,
            predictor_eval_count=len(
                self._build_predictor_eval_specs(step_ctx=step_ctx, solver_mode=solver_mode)
            ),
            refinement_steps=refinement_steps,
            refinement_solver_mode=refinement_solver_mode,
        )

        # Refinement is inversion-only: each round starts from the current
        # t_next anchor, but always recomputes the step update from the
        # original low-noise sample x_t. RF-family refinement therefore
        # materializes both {t_next, midpoint} sidecars per round.
        corrected_sample = predictor_result.x_pred
        dt = float(step_ctx.t_next - step_ctx.t_curr)
        refinement_states: list[TraceEvalState] = []
        eval_iter = iter(eval_specs)
        for _ in range(int(refinement_steps)):
            eval_spec = next(eval_iter)
            refinement_states.append(
                TraceEvalState(
                    actual_t=float(eval_spec.actual_t),
                    logical_t=float(eval_spec.logical_t),
                    eval_idx=int(eval_spec.eval_idx),
                    sample=corrected_sample,
                )
            )
            corrected_v = self._predict_with_optional_cfg(
                model,
                corrected_sample,
                cond_dict,
                cfg_strength=cfg_strength,
                cfg_interval=cfg_interval,
                eval_spec=eval_spec,
                hook=hook,
                phase=step_ctx.phase,
            )
            if refinement_solver_mode == "simple":
                corrected_sample = sample + dt * corrected_v
                continue

            sample_mid = sample + 0.5 * dt * corrected_v
            mid_eval_spec = next(eval_iter)
            refinement_states.append(
                TraceEvalState(
                    actual_t=float(mid_eval_spec.actual_t),
                    logical_t=float(mid_eval_spec.logical_t),
                    eval_idx=int(mid_eval_spec.eval_idx),
                    sample=sample_mid,
                )
            )
            corrected_v_mid = self._predict_with_optional_cfg(
                model,
                sample_mid,
                cond_dict,
                cfg_strength=cfg_strength,
                cfg_interval=cfg_interval,
                eval_spec=mid_eval_spec,
                hook=hook,
                phase=step_ctx.phase,
            )
            if refinement_solver_mode == "rf_solver":
                corrected_sample = sample + dt * corrected_v_mid
            else:
                corrected_sample = sample + dt * (2.0 * corrected_v - corrected_v_mid)
        return RefinementResult(
            x_next=corrected_sample,
            eval_states=tuple(refinement_states),
        )

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
    ) -> SourceTrace | tuple[SourceTrace, Any]:
        self._validate_solver_mode(inversion_mode)

        extra = config.extra_params or {}
        inversion_cfg = self._resolve_inversion_cfg(stage_prefix, extra)
        refinement_steps = self._resolve_refinement_steps(stage_prefix, extra)
        refinement_solver_mode = self._resolve_refinement_solver_mode(
            stage_prefix,
            extra,
            predictor_solver_mode=inversion_mode,
        )
        inversion_steps = self._resolve_inversion_steps(stage_prefix, steps, extra)
        t_pairs, _ = build_inversion_t_pairs(
            steps=steps,
            rescale_t=rescale_t,
            inversion_steps=inversion_steps,
        )

        source_trace = SourceTrace()
        source_trace.add_entry(
            step_index=-1,
            logical_t=0.0,
            sample=self._trace_sample_to_cpu(sample),
        )

        with torch.inference_mode():
            for i, (t_curr, t_next) in enumerate(t_pairs):
                step_ctx = StepContext(
                    stage=stage_prefix,
                    phase="inversion",
                    step_index=i,
                    t_curr=float(t_curr),
                    t_next=float(t_next),
                    logical_t=self._resolve_solver_logical_t(
                        phase="inversion",
                        t_curr=t_curr,
                        t_next=t_next,
                    ),
                )
                step_result = self._sample_with_solver(
                    model,
                    sample,
                    cond_dict,
                    step_ctx=step_ctx,
                    cfg_strength=inversion_cfg["cfg_strength"],
                    cfg_interval=inversion_cfg["cfg_interval"],
                    solver_mode=inversion_mode,
                    refinement_steps=refinement_steps,
                    refinement_solver_mode=refinement_solver_mode,
                    hook=hook,
                )
                sample = step_result.x_next
                source_trace.add_entry(
                    step_index=i,
                    logical_t=step_ctx.logical_t,
                    sample=self._trace_sample_to_cpu(sample),
                    predictor_states=self._trace_eval_states_to_cpu(step_result.predictor_states),
                    refinement_states=self._trace_eval_states_to_cpu(step_result.refinement_states),
                )

                if i % 5 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if return_terminal_noise:
            return source_trace, sample
        return source_trace

    def _sample_with_solver(
        self,
        model,
        sample,
        cond_dict: Dict[str, Any],
        *,
        step_ctx: StepContext,
        cfg_strength: float,
        cfg_interval: Tuple[float, float],
        solver_mode: str,
        refinement_steps: int,
        refinement_solver_mode: str,
        hook: Any | None = None,
    ) -> SolverStepResult:
        self._validate_solver_mode(solver_mode)
        self._validate_solver_mode(refinement_solver_mode)

        predictor_result = self._predictor_step(
            model,
            sample,
            cond_dict,
            step_ctx=step_ctx,
            cfg_strength=cfg_strength,
            cfg_interval=cfg_interval,
            solver_mode=solver_mode,
            hook=hook,
        )
        refinement_result = self._refine_step(
            model,
            sample,
            predictor_result,
            cond_dict,
            step_ctx=step_ctx,
            cfg_strength=cfg_strength,
            cfg_interval=cfg_interval,
            solver_mode=solver_mode,
            refinement_steps=refinement_steps,
            refinement_solver_mode=refinement_solver_mode,
            hook=hook,
        )
        return SolverStepResult(
            x_next=refinement_result.x_next,
            predictor_states=tuple(predictor_result.eval_states),
            refinement_states=tuple(refinement_result.eval_states),
        )

    def _prepare_ss_latent_from_coords(
        self,
        pipeline,
        source_coords: torch.Tensor,
        source_cond_dict: Dict[str, Any],
        inversion_mode: str,
        config: Any,
        resolution: int,
        hook: Any | None = None,
    ) -> SourceTrace:
        source_trace, _ = self._prepare_ss_latent_and_terminal_from_coords(
            pipeline,
            source_coords,
            source_cond_dict,
            inversion_mode,
            config,
            resolution,
            hook=hook,
        )
        return source_trace

    def _prepare_ss_latent_and_terminal_from_coords(
        self,
        pipeline,
        source_coords: torch.Tensor,
        source_cond_dict: Dict[str, Any],
        inversion_mode: str,
        config: Any,
        resolution: int,
        hook: Any | None = None,
    ) -> tuple[SourceTrace, torch.Tensor]:
        if inversion_mode == "none":
            raise ValueError(
                "inversion_mode='none' is not valid. "
                "Must do inversion to get per-step latent cache. "
                "Use 'simple', 'rf_solver', or 'voxhammer_rf_solver'."
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
        source_coords: torch.Tensor | None = None,
        resolution: int = 64,
        channels: int = 8,
        latent_resolution: int = 16,
        hard_mask_mode: str = "edit_all",
        edit_region_mode: str = "mask_only",
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

        mask_coords_int = mask_coords[:, 1:].long() if mask_coords.shape[1] == 4 else mask_coords.long()
        if edit_region_mode == "mask_only":
            if mask_coords_int.shape[0] > 0:
                voxel_mask[0, 0, mask_coords_int[:, 0], mask_coords_int[:, 1], mask_coords_int[:, 2]] = 1.0
        elif edit_region_mode == "preserve_complement":
            if source_coords is None:
                raise RuntimeError(
                    "SS edit_region_mode='preserve_complement' requires source voxel coordinates."
                )
            source_coords_int = (
                source_coords[:, 1:].long() if source_coords.shape[1] == 4 else source_coords.long()
            )
            voxel_mask.fill_(1.0)
            if source_coords_int.shape[0] > 0:
                source_codes = coords_to_flat_indices(source_coords_int, resolution=resolution)
                mask_codes = coords_to_flat_indices(mask_coords_int, resolution=resolution)
                preserve_coords = source_coords_int[~torch.isin(source_codes, mask_codes)]
                if preserve_coords.shape[0] > 0:
                    voxel_mask[0, 0, preserve_coords[:, 0], preserve_coords[:, 1], preserve_coords[:, 2]] = 0.0
        else:
            raise ValueError(f"Unknown SS edit_region_mode: {edit_region_mode}")

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

    @staticmethod
    def _trace_sample_to_device(
        sample: Any,
        *,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> Any:
        if torch.is_tensor(sample):
            return sample.to(device=device, dtype=sample.dtype if dtype is None else dtype)
        if hasattr(sample, "to"):
            if dtype is None:
                return sample.to(device=device)
            return sample.to(device=device, dtype=dtype)
        raise TypeError(f"Unsupported source trace sample type: {type(sample)!r}")

    @staticmethod
    def _trace_sample_to_cpu(sample: Any) -> Any:
        if torch.is_tensor(sample):
            return sample.detach().cpu()
        if hasattr(sample, "detach"):
            sample = sample.detach()
        if hasattr(sample, "cpu"):
            return sample.cpu()
        if hasattr(sample, "to"):
            return sample.to(device="cpu")
        raise TypeError(f"Unsupported source trace sample type: {type(sample)!r}")

    @classmethod
    def _trace_eval_states_to_cpu(
        cls,
        eval_states: tuple[TraceEvalState, ...] | list[TraceEvalState],
    ) -> tuple[TraceEvalState, ...]:
        return tuple(
            TraceEvalState(
                actual_t=float(state.actual_t),
                logical_t=float(state.logical_t),
                eval_idx=int(state.eval_idx),
                sample=cls._trace_sample_to_cpu(state.sample),
            )
            for state in eval_states
        )

    def _prepare_step_source_snapshots(
        self,
        *,
        hook: Any | None,
        stage_name: str,
        model,
        source_trace: SourceTrace | None,
        source_cond_dict: Dict[str, Any] | None,
        step_ctx: StepContext,
        cfg_strength: float,
        cfg_interval: Tuple[float, float],
        solver_mode: str,
        source_bundle_mode: str,
    ) -> None:
        if hook is None or not hasattr(hook, "requires_step_source_snapshot"):
            return
        if source_trace is None or source_cond_dict is None:
            return
        if not hook.requires_step_source_snapshot(stage_name, step_ctx.logical_t):
            return

        source_entry = source_trace.get_entry(step_ctx.logical_t)
        if source_entry is None:
            return

        device = source_cond_dict["cond"].device
        predictor_eval_specs = self._build_predictor_eval_specs(
            step_ctx=step_ctx,
            solver_mode=solver_mode,
        )
        bundle_mode = self._validate_kv_source_bundle_mode(source_bundle_mode)
        hook.clear_step_source_snapshot(stage_name=stage_name)
        hook.begin_step_source_capture(stage_name=stage_name, logical_t=step_ctx.logical_t)
        try:
            if bundle_mode == "external_voxhammer":
                predictor_states = tuple(source_entry.predictor_states)
                if len(predictor_states) != len(predictor_eval_specs):
                    raise RuntimeError(
                        f"{stage_name} source entry logical_t={step_ctx.logical_t:.10f} has "
                        f"{len(predictor_states)} predictor states, expected {len(predictor_eval_specs)} "
                        f"for bundle mode '{bundle_mode}' and solver '{solver_mode}'."
                    )
                for expected_eval_spec, source_state in zip(predictor_eval_specs, predictor_states):
                    if int(source_state.eval_idx) != int(expected_eval_spec.eval_idx):
                        raise RuntimeError(
                            f"{stage_name} source predictor eval_idx mismatch at logical_t="
                            f"{step_ctx.logical_t:.10f}: got {source_state.eval_idx}, "
                            f"expected {expected_eval_spec.eval_idx}."
                        )
                    replay_sample = self._trace_sample_to_device(source_state.sample, device=device)
                    eval_spec = SolverEvalSpec(
                        actual_t=float(source_state.actual_t),
                        logical_t=float(step_ctx.logical_t),
                        eval_idx=int(expected_eval_spec.eval_idx),
                    )
                    self._predict_with_optional_cfg(
                        model,
                        replay_sample,
                        source_cond_dict,
                        cfg_strength=cfg_strength,
                        cfg_interval=cfg_interval,
                        eval_spec=eval_spec,
                        hook=hook,
                        phase="source_capture",
                    )
            else:
                source_sample = self._trace_sample_to_device(source_entry.sample, device=device)
                replay_sample = source_sample
                dt = float(step_ctx.t_next - step_ctx.t_curr)
                for eval_idx, eval_spec in enumerate(predictor_eval_specs):
                    pred_v = self._predict_with_optional_cfg(
                        model,
                        replay_sample,
                        source_cond_dict,
                        cfg_strength=cfg_strength,
                        cfg_interval=cfg_interval,
                        eval_spec=eval_spec,
                        hook=hook,
                        phase="source_capture",
                    )
                    if eval_idx == 0 and len(predictor_eval_specs) > 1:
                        replay_sample = source_sample + 0.5 * dt * pred_v
            snapshot = hook.end_step_source_capture(stage_name=stage_name, logical_t=step_ctx.logical_t)
        except Exception:
            hook.cancel_step_source_capture(stage_name=stage_name)
            raise
        hook.set_step_source_snapshot(
            stage_name=stage_name,
            logical_t=step_ctx.logical_t,
            snapshot=snapshot,
        )

    def _denoise_ss_with_step_blend(
        self,
        pipeline,
        edit_cond_dict: Dict[str, Any],
        config: Any,
        *,
        initial_sample: Optional[torch.Tensor] = None,
        source_trace: SourceTrace | None = None,
        source_cond_dict: Dict[str, Any] | None = None,
        latent_mask: Optional[torch.Tensor] = None,
        blend_strength: float = 1.0,
        solver_mode: str = "simple",
        verbose: bool = False,
        hook: Any | None = None,
    ) -> torch.Tensor:
        self._validate_solver_mode(solver_mode)

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
        source_bundle_mode = self._resolve_kv_source_bundle_mode("ss", extra)
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
                step_ctx = StepContext(
                    stage="ss",
                    phase="denoise",
                    step_index=i,
                    t_curr=float(t_curr),
                    t_next=float(t_next),
                    logical_t=self._resolve_solver_logical_t(
                        phase="denoise",
                        t_curr=t_curr,
                        t_next=t_next,
                    ),
                )
                if source_trace is not None and mask is not None:
                    source_latent = source_trace.get_sample(step_ctx.logical_t)
                    if source_latent is not None:
                        source_latent = source_latent.to(sample.device, sample.dtype)
                        sample = mask * sample + (1 - mask) * (
                            blend_strength * source_latent + (1 - blend_strength) * sample
                        )

                self._prepare_step_source_snapshots(
                    hook=hook,
                    stage_name="ss",
                    model=flow_model,
                    source_trace=source_trace,
                    source_cond_dict=source_cond_dict,
                    step_ctx=step_ctx,
                    cfg_strength=cfg_strength,
                    cfg_interval=cfg_interval,
                    solver_mode=solver_mode,
                    source_bundle_mode=source_bundle_mode,
                )
                try:
                    sample = self._sample_with_solver(
                        flow_model,
                        sample,
                        edit_cond_dict,
                        step_ctx=step_ctx,
                        cfg_strength=cfg_strength,
                        cfg_interval=cfg_interval,
                        solver_mode=solver_mode,
                        refinement_steps=0,
                        refinement_solver_mode=solver_mode,
                        hook=hook,
                    ).x_next
                finally:
                    if hook is not None and hasattr(hook, "clear_step_source_snapshot"):
                        hook.clear_step_source_snapshot(stage_name="ss")

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
        source_trace: SourceTrace | None = None,
        source_cond_dict: Dict[str, Any] | None = None,
        latent_mask: Optional[torch.Tensor | SparseLatentBlendMask] = None,
        solver_mode: str = "simple",
        verbose: bool = False,
        hook: Any | None = None,
    ):
        self._validate_solver_mode(solver_mode)
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
        source_bundle_mode = self._resolve_kv_source_bundle_mode("slat", extra)
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
                step_ctx = StepContext(
                    stage="slat",
                    phase="denoise",
                    step_index=i,
                    t_curr=float(t_curr),
                    t_next=float(t_next),
                    logical_t=self._resolve_solver_logical_t(
                        phase="denoise",
                        t_curr=t_curr,
                        t_next=t_next,
                    ),
                )
                if source_trace is not None and latent_mask is not None:
                    source_latent = source_trace.get_sample(step_ctx.logical_t)
                    if source_latent is not None:
                        sample, _ = blend_sparse_features(sample, source_latent, latent_mask)

                self._prepare_step_source_snapshots(
                    hook=hook,
                    stage_name="slat",
                    model=flow_model,
                    source_trace=source_trace,
                    source_cond_dict=source_cond_dict,
                    step_ctx=step_ctx,
                    cfg_strength=cfg_strength,
                    cfg_interval=cfg_interval,
                    solver_mode=solver_mode,
                    source_bundle_mode=source_bundle_mode,
                )
                try:
                    sample = self._sample_with_solver(
                        flow_model,
                        sample,
                        edit_cond_dict,
                        step_ctx=step_ctx,
                        cfg_strength=cfg_strength,
                        cfg_interval=cfg_interval,
                        solver_mode=solver_mode,
                        refinement_steps=0,
                        refinement_solver_mode=solver_mode,
                        hook=hook,
                    ).x_next
                finally:
                    if hook is not None and hasattr(hook, "clear_step_source_snapshot"):
                        hook.clear_step_source_snapshot(stage_name="slat")

                if i % 5 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return sample
