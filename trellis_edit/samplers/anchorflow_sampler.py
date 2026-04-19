from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tqdm import tqdm


def _reduction_dims(x: torch.Tensor) -> tuple[int, ...]:
    return tuple(range(1, x.ndim))


def _rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x.square().mean(dim=_reduction_dims(x), keepdim=True).add(eps).sqrt()


def _mean_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x * y).mean(dim=_reduction_dims(x), keepdim=True)


class AnchorFlowSampler:
    """Sparse-structure AnchorFlow sampler.

    AnchorFlow reuses the same no-inversion latent editing setup as FlowEdit,
    but augments the velocity delta with an anchor-alignment term in latent
    space. The implementation here mirrors the parent `../AnchorFlow` repo
    while staying compatible with TRELLIS's flow-euler sampler interface.
    """

    @staticmethod
    def _current_sigma(sampler: Any, t: float) -> float:
        sigma_min = float(getattr(sampler, "sigma_min", 0.0))
        return sigma_min + (1.0 - sigma_min) * float(t)

    @staticmethod
    def _latent_anchor(sample: torch.Tensor, velocity: torch.Tensor, sigma: float) -> torch.Tensor:
        return sample + (1.0 - float(sigma)) * velocity

    @staticmethod
    def _shared_center_residual_term(
        anchor_source: torch.Tensor,
        anchor_target: torch.Tensor,
        flow_delta: torch.Tensor,
        prev_center: torch.Tensor | None,
        prev_residual: torch.Tensor | None,
        *,
        center_weight: float,
        band_weight: float,
        ortho_weight: float,
        residual_weight: float,
        band_min_ratio: float,
        band_max_ratio: float,
        margin_scale: float,
        direction_gate_tau: float,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        center = 0.5 * (anchor_source + anchor_target)
        residual = anchor_target - anchor_source

        # Use the guided source/target flow gap as the semantic edit direction.
        flow_mag = _rms_norm(flow_delta.detach(), eps)
        flow_dir = flow_delta.detach() / flow_mag.clamp_min(eps)

        signed_parallel_coeff = _mean_dot(residual, flow_dir)
        parallel = signed_parallel_coeff * flow_dir
        orthogonal = residual - parallel

        margin = (margin_scale * flow_mag).clamp_min(eps)
        lower = band_min_ratio * margin
        upper = band_max_ratio * margin

        desired_parallel_coeff = signed_parallel_coeff.clamp_min(0.0)
        desired_parallel_coeff = torch.maximum(desired_parallel_coeff, lower)
        desired_parallel_coeff = torch.minimum(desired_parallel_coeff, upper)
        desired_parallel = desired_parallel_coeff * flow_dir

        band_correction = desired_parallel - parallel
        ortho_correction = -orthogonal

        if prev_center is None:
            center_correction = torch.zeros_like(center)
        else:
            center_correction = -(center - prev_center)

        if prev_residual is None:
            residual_correction = torch.zeros_like(residual)
        else:
            residual_correction = -(residual - prev_residual)

        if direction_gate_tau > 0.0:
            direction_gate = flow_mag / (flow_mag + direction_gate_tau)
        else:
            direction_gate = torch.ones_like(flow_mag)

        anchor_term = (
            residual
            + direction_gate * (band_weight * band_correction + ortho_weight * ortho_correction)
            + center_weight * center_correction
            + residual_weight * residual_correction
        )

        return anchor_term, center.detach(), residual.detach()

    @torch.no_grad()
    def sample(
        self,
        *,
        sampler: Any,
        model: Any,
        source_latent: torch.Tensor,
        source_cond: torch.Tensor,
        target_cond: torch.Tensor,
        neg_cond: torch.Tensor,
        steps: int,
        rescale_t: float,
        n_max: int,
        n_avg: int,
        source_cfg_strength: float,
        target_cfg_strength: float,
        cfg_interval: tuple[float, float],
        center_weight: float,
        band_weight: float,
        ortho_weight: float,
        residual_weight: float,
        band_min_ratio: float,
        band_max_ratio: float,
        margin_scale: float,
        direction_gate_tau: float,
        eps: float,
        anchor_noise: bool = False,
        verbose: bool = True,
    ) -> torch.Tensor:
        if source_latent.shape[0] != 1:
            raise RuntimeError(
                "AnchorFlowSampler currently expects a single source latent; "
                f"got batch size {source_latent.shape[0]}."
            )

        total_steps = int(steps)
        n_max = int(n_max)
        n_avg = int(n_avg)
        if total_steps <= 0:
            raise RuntimeError(f"AnchorFlow steps must be > 0, got {total_steps}")
        if n_max <= 0:
            raise RuntimeError(f"AnchorFlow n_max must be > 0, got {n_max}")
        if n_avg <= 0:
            raise RuntimeError(f"AnchorFlow n_avg must be > 0, got {n_avg}")

        output_dtype = source_latent.dtype
        source_latent = source_latent.to(dtype=torch.float32)
        edited_latent = source_latent.clone()

        t_seq = np.linspace(1, 0, total_steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((float(t_seq[i]), float(t_seq[i + 1])) for i in range(total_steps))
        start_step = max(0, total_steps - n_max)

        source_kwargs = {
            "neg_cond": neg_cond,
            "cfg_strength": float(source_cfg_strength),
            "cfg_interval": cfg_interval,
        }
        target_kwargs = {
            "neg_cond": neg_cond,
            "cfg_strength": float(target_cfg_strength),
            "cfg_interval": cfg_interval,
        }

        fixed_noise = torch.randn_like(source_latent) if anchor_noise else None
        prev_center: torch.Tensor | None = None
        prev_residual: torch.Tensor | None = None

        for step_idx, (t, t_prev) in enumerate(
            tqdm(t_pairs, desc="AnchorFlow SS", disable=not verbose)
        ):
            if step_idx < start_step:
                continue

            current_sigma = self._current_sigma(sampler, t)
            velocity_delta = torch.zeros_like(source_latent)
            step_center_sum: torch.Tensor | None = None
            step_residual_sum: torch.Tensor | None = None

            for _ in range(n_avg):
                forward_noise = fixed_noise if fixed_noise is not None else torch.randn_like(source_latent)
                latent_source_t = (1.0 - t) * source_latent + current_sigma * forward_noise
                latent_target_t = edited_latent + latent_source_t - source_latent

                _, _, velocity_source = sampler._get_model_prediction(
                    model,
                    latent_source_t,
                    t,
                    source_cond,
                    **source_kwargs,
                )
                _, _, velocity_target = sampler._get_model_prediction(
                    model,
                    latent_target_t,
                    t,
                    target_cond,
                    **target_kwargs,
                )

                flow_delta = velocity_target - velocity_source
                anchor_source = self._latent_anchor(latent_source_t, velocity_source, current_sigma)
                anchor_target = self._latent_anchor(latent_target_t, velocity_target, current_sigma)
                anchor_term, step_center, step_residual = self._shared_center_residual_term(
                    anchor_source,
                    anchor_target,
                    flow_delta,
                    prev_center=prev_center,
                    prev_residual=prev_residual,
                    center_weight=float(center_weight),
                    band_weight=float(band_weight),
                    ortho_weight=float(ortho_weight),
                    residual_weight=float(residual_weight),
                    band_min_ratio=float(band_min_ratio),
                    band_max_ratio=float(band_max_ratio),
                    margin_scale=float(margin_scale),
                    direction_gate_tau=float(direction_gate_tau),
                    eps=float(eps),
                )
                velocity_delta += 2.0 * flow_delta + (1.0 - current_sigma) * anchor_term

                if step_center_sum is None:
                    step_center_sum = step_center
                    step_residual_sum = step_residual
                else:
                    step_center_sum = step_center_sum + step_center
                    step_residual_sum = step_residual_sum + step_residual

            velocity_delta = velocity_delta / float(n_avg)
            if step_center_sum is not None and step_residual_sum is not None:
                prev_center = step_center_sum / float(n_avg)
                prev_residual = step_residual_sum / float(n_avg)

            edited_latent = edited_latent + (t_prev - t) * velocity_delta

        return edited_latent.to(dtype=output_dtype)
