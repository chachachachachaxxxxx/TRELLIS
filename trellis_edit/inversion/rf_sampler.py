"""RF-Solver: Second-order Rectified Flow sampler from VoxHammer.

Reference: VoxHammer paper, Section 3.1
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


def build_rescaled_t_seq(steps: int, rescale_t: float) -> np.ndarray:
    """Build the shared monotonic time grid used by inversion and denoising."""
    total_steps = int(steps)
    if total_steps <= 0:
        raise ValueError(f"steps must be > 0, got {total_steps}")

    rescale_t = float(rescale_t)
    t_seq = np.linspace(0.0, 1.0, total_steps + 1)
    return rescale_t * t_seq / (1.0 + (rescale_t - 1.0) * t_seq)


def resolve_inversion_steps(total_steps: int, inversion_steps: Optional[int]) -> int:
    """Resolve the effective inversion depth against the full schedule length."""
    total_steps = int(total_steps)
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}")

    if inversion_steps is None:
        return total_steps

    resolved = int(inversion_steps)
    if resolved <= 0:
        raise ValueError(f"inversion_steps must be > 0, got {resolved}")
    if resolved > total_steps:
        raise ValueError(
            f"inversion_steps must be <= total_steps, got {resolved} > {total_steps}"
        )
    return resolved


def build_inversion_t_pairs(
    steps: int,
    rescale_t: float,
    inversion_steps: Optional[int] = None,
) -> tuple[list[tuple[float, float]], int]:
    """Build forward data->noise timestep pairs up to the requested inversion depth."""
    total_steps = int(steps)
    stop_step = resolve_inversion_steps(total_steps, inversion_steps)
    t_seq = build_rescaled_t_seq(total_steps, rescale_t)
    t_pairs = [(float(t_seq[i]), float(t_seq[i + 1])) for i in range(stop_step)]
    return t_pairs, stop_step


def build_denoise_t_pairs(
    steps: int,
    rescale_t: float,
    start_step: Optional[int] = None,
) -> tuple[list[tuple[float, float]], int]:
    """Build reverse noise->data timestep pairs from the requested starting point."""
    total_steps = int(steps)
    resolved_start_step = resolve_inversion_steps(total_steps, start_step)
    t_seq = build_rescaled_t_seq(total_steps, rescale_t)
    t_pairs = [(float(t_seq[i]), float(t_seq[i - 1])) for i in range(resolved_start_step, 0, -1)]
    return t_pairs, resolved_start_step


class RFSolverSampler:
    """RF-Solver: Taylor-improved second-order flow sampler from VoxHammer.

    Implements the inversion and denoising scheme described in VoxHammer paper Section 3.1.
    Uses second-order Taylor expansion with midpoint correction for improved accuracy.

    Features:
    - Bidirectional sampling: inverse (data → noise) and forward (noise → data)
    - Second-order midpoint correction for better trajectory approximation
    - Late-time CFG: applies classifier-free guidance only in t ∈ [cfg_interval[0], cfg_interval[1]]
    """

    @torch.no_grad()
    def _run_model(self, model, sample, t_value: float, cond: Optional[torch.Tensor]):
        """Run model with given timestep and condition."""
        t = torch.tensor([1000.0 * t_value] * sample.shape[0], device=sample.device, dtype=torch.float32)
        if cond is not None and cond.shape[0] == 1 and sample.shape[0] > 1:
            cond = cond.repeat(sample.shape[0], *([1] * (cond.ndim - 1)))
        return model(sample, t, cond)

    @torch.no_grad()
    def _is_sparse_like(self, value) -> bool:
        return hasattr(value, "coords") and hasattr(value, "feats")

    @torch.no_grad()
    def _guided_prediction(
        self,
        model,
        sample,
        t_value: float,
        cond_dict: dict,
        cfg_strength: float,
        cfg_interval: Tuple[float, float],
    ):
        """Compute CFG-guided prediction with aggressive memory cleanup."""
        import gc

        if cfg_interval[0] <= t_value <= cfg_interval[1] and cfg_strength > 0.0:
            # Run positive prediction
            pred = self._run_model(model, sample, t_value, cond_dict["cond"])

            # Sparse flow predictions are not safe to shuttle through CPU between
            # CFG branches. Keep them on-device and combine them directly.
            if self._is_sparse_like(pred):
                neg_pred = self._run_model(model, sample, t_value, cond_dict["neg_cond"])
                result = (1.0 + cfg_strength) * pred - cfg_strength * neg_pred
                del pred, neg_pred
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return result

            # Move to CPU immediately to free GPU memory
            pred_cpu = pred.cpu()
            del pred
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Run negative prediction
            neg_pred = self._run_model(model, sample, t_value, cond_dict["neg_cond"])

            # Move back to GPU for computation
            pred = pred_cpu.to(sample.device)
            del pred_cpu

            # Compute guided prediction
            result = (1.0 + cfg_strength) * pred - cfg_strength * neg_pred

            # Clean up intermediate tensors
            del pred, neg_pred
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return result

        return self._run_model(model, sample, t_value, cond_dict["cond"])

    @torch.no_grad()
    def sample_once(
        self,
        model,
        sample,
        t_curr: float,
        t_next: float,
        cond_dict: dict,
        cfg_strength: float,
        cfg_interval: Tuple[float, float],
    ):
        """Single sampling step with second-order Taylor correction.

        Uses the second-order Taylor update:
            x_{t+h} = x_t + h·f(x_t, t) + ½·h²·∂_t f(x_t, t)

        where ∂_t f is approximated via midpoint finite difference.
        """
        import gc

        # First prediction at current point
        pred = self._guided_prediction(model, sample, t_curr, cond_dict, cfg_strength, cfg_interval)
        dt = t_next - t_curr

        # Compute midpoint sample
        sample_mid = sample + 0.5 * dt * pred
        t_mid = t_curr + 0.5 * dt

        # Clean up before midpoint prediction
        del pred
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Midpoint prediction
        pred_mid = self._guided_prediction(model, sample_mid, t_mid, cond_dict, cfg_strength, cfg_interval)

        # Recompute first prediction (needed for second-order correction)
        pred = self._guided_prediction(model, sample, t_curr, cond_dict, cfg_strength, cfg_interval)

        # With first_order := (pred_mid - pred) / (dt / 2), the Taylor
        # correction enters with a plus sign.
        first_order = (pred_mid - pred) / (0.5 * dt)
        result = sample + dt * pred + 0.5 * (dt**2) * first_order

        # Clean up intermediate tensors
        del pred, sample_mid, pred_mid, first_order
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    @torch.no_grad()
    def sample(
        self,
        model,
        sample,
        cond_dict: dict,
        steps: int,
        rescale_t: float,
        cfg_strength: float,
        cfg_interval: Tuple[float, float],
        inverse: bool,
        start_step: Optional[int] = None,
        verbose: bool = True,
    ):
        """Run full RF-Solver sampling process.

        Args:
            model: Flow model
            sample: Initial sample (data for inverse, noise for forward)
            cond_dict: Condition dictionary with 'cond' and 'neg_cond'
            steps: Number of sampling steps
            rescale_t: Time rescaling factor
            cfg_strength: CFG strength
            cfg_interval: CFG interval (start, end)
            inverse: If True, run inverse (data → noise), else forward (noise → data)
            start_step: When set, truncate inversion at this step index and start
                denoising from the same intermediate point instead of t=1.
            verbose: Show progress bar

        Returns:
            Final sample
        """
        if inverse:
            t_pairs, _ = build_inversion_t_pairs(
                steps=steps,
                rescale_t=rescale_t,
                inversion_steps=start_step,
            )
            desc = "RF-Solver inversion"
        else:
            t_pairs, _ = build_denoise_t_pairs(
                steps=steps,
                rescale_t=rescale_t,
                start_step=start_step,
            )
            desc = "RF-Solver denoise"
        for t_curr, t_next in tqdm(t_pairs, desc=desc, disable=not verbose):
            sample = self.sample_once(model, sample, t_curr, t_next, cond_dict, cfg_strength, cfg_interval)
        return sample
