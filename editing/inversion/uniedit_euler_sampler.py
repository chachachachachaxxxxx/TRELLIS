"""UniEdit with first-order Euler + predictor-corrector strategy.

Simplified version of UniEdit that uses:
- First-order Euler integration (no second-order correction)
- Predictor-corrector for improved accuracy
- Delayed inversion/editing with alpha parameter
- Source/target velocity fusion with omega parameter
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


def _dense_scalar_field(x: torch.Tensor) -> torch.Tensor:
    """Compute scalar field from dense tensor."""
    return x.abs().mean(dim=1, keepdim=True)


def _sparse_scalar_field(x) -> torch.Tensor:
    """Compute scalar field from sparse tensor."""
    return x.feats.abs().mean(dim=1, keepdim=True)


def _normalize_dense_map(scores: torch.Tensor, selector: Optional[torch.Tensor]) -> torch.Tensor:
    """Normalize dense score map to [0, 1] range."""
    bsz = scores.shape[0]
    scores_flat = scores.reshape(bsz, -1)
    out_flat = torch.zeros_like(scores_flat)
    if selector is None:
        selector_flat = torch.ones_like(scores_flat, dtype=torch.bool)
    else:
        selector_flat = selector.reshape(bsz, -1) > 0.5

    for batch_idx in range(bsz):
        active = selector_flat[batch_idx]
        if not torch.any(active):
            continue
        vals = scores_flat[batch_idx, active]
        min_val = vals.min()
        max_val = vals.max()
        denom = torch.clamp(max_val - min_val, min=1e-6)
        out_flat[batch_idx, active] = (scores_flat[batch_idx, active] - min_val) / denom
    return out_flat.reshape_as(scores)


def _normalize_sparse_map(scores: torch.Tensor, coords: torch.Tensor, selector: Optional[torch.Tensor]) -> torch.Tensor:
    """Normalize sparse score map to [0, 1] range."""
    out = torch.zeros_like(scores)
    batch_ids = coords[:, 0].long()
    active_selector = None if selector is None else (selector.reshape(-1) > 0.5)
    batch_size = int(batch_ids.max().item()) + 1 if batch_ids.numel() > 0 else 0

    for batch_idx in range(batch_size):
        batch_mask = batch_ids == batch_idx
        if active_selector is None:
            active = batch_mask
        else:
            active = batch_mask & active_selector
        if not torch.any(active):
            continue
        vals = scores[active]
        min_val = vals.min()
        max_val = vals.max()
        denom = torch.clamp(max_val - min_val, min=1e-6)
        out[active] = (scores[active] - min_val) / denom
    return out


def compute_uniedit_map(guidance, selector: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute UniEdit preservation map from guidance.

    Args:
        guidance: Dense tensor or SparseTensor containing guidance signal
        selector: Optional selector mask for normalization

    Returns:
        Normalized preservation map in [0, 1] range
    """
    # Check if it's a SparseTensor by checking for coords attribute
    if hasattr(guidance, 'coords') and hasattr(guidance, 'feats'):
        scores = _sparse_scalar_field(guidance)
        return _normalize_sparse_map(scores=scores, coords=guidance.coords, selector=selector)
    if torch.is_tensor(guidance):
        scores = _dense_scalar_field(guidance)
        return _normalize_dense_map(scores=scores, selector=selector)
    raise RuntimeError(f"Unsupported guidance type: {type(guidance)}")


class UniEditEulerSampler:
    """UniEdit with first-order Euler + predictor-corrector.

    Key features:
    - Delayed inversion: only invert first alpha fraction of steps
    - Delayed editing: only edit last alpha fraction of steps
    - Predictor-corrector: predict, step, predict again, use corrected prediction
    - Source/target fusion: blend velocities based on their difference
    """

    @torch.no_grad()
    def _run_model(self, model, sample, t_value: float, cond: Optional[torch.Tensor]):
        """Run model with given timestep and condition."""
        t = torch.tensor([1000.0 * t_value] * sample.shape[0], device=sample.device, dtype=torch.float32)
        if cond is not None and cond.shape[0] == 1 and sample.shape[0] > 1:
            cond = cond.repeat(sample.shape[0], *([1] * (cond.ndim - 1)))
        return model(sample, t, cond)

    @torch.no_grad()
    def _guided_prediction_for_cond(
        self,
        model,
        sample,
        t_value: float,
        cond: torch.Tensor,
        neg_cond: torch.Tensor,
        cfg_strength: float,
        cfg_interval: Tuple[float, float],
    ):
        """Compute CFG-guided prediction for a single condition."""
        if cfg_interval[0] <= t_value <= cfg_interval[1] and cfg_strength > 0.0:
            # Compute positive prediction
            pred = self._run_model(model, sample, t_value, cond)

            # Clean up before negative prediction
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Compute negative prediction
            neg_pred = self._run_model(model, sample, t_value, neg_cond)

            # Compute CFG result
            result = (1.0 + cfg_strength) * pred - cfg_strength * neg_pred

            # Clean up intermediate tensors
            del pred, neg_pred
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return result
        return self._run_model(model, sample, t_value, cond)

    @torch.no_grad()
    def invert_once_predictor_corrector(
        self,
        model,
        sample,
        t_curr: float,
        t_next: float,
        cond: torch.Tensor,
        neg_cond: torch.Tensor,
        cfg_strength: float,
        cfg_interval: Tuple[float, float],
        zero_init: bool = False,
    ):
        """Single inversion step with predictor-corrector.

        Args:
            model: Flow model
            sample: Current sample
            t_curr: Current timestep
            t_next: Next timestep
            cond: Condition (empty for inversion)
            neg_cond: Negative condition
            cfg_strength: CFG strength
            cfg_interval: CFG interval
            zero_init: If True, use zero velocity for first prediction

        Returns:
            Updated sample
        """
        import gc

        dt = t_next - t_curr

        # Predictor: predict velocity at current point
        if zero_init:
            pred = torch.zeros_like(sample) if torch.is_tensor(sample) else type(sample)(
                coords=sample.coords,
                feats=torch.zeros_like(sample.feats),
                shape=sample.shape,
            )
        else:
            pred = self._guided_prediction_for_cond(
                model=model,
                sample=sample,
                t_value=t_curr,
                cond=cond,
                neg_cond=neg_cond,
                cfg_strength=cfg_strength,
                cfg_interval=cfg_interval,
            )

        # Take a temporary step
        sample_next = sample + dt * pred

        # Clean up
        del pred
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Corrector: predict velocity at next point
        pred_next = self._guided_prediction_for_cond(
            model=model,
            sample=sample_next,
            t_value=t_next,
            cond=cond,
            neg_cond=neg_cond,
            cfg_strength=cfg_strength,
            cfg_interval=cfg_interval,
        )

        # Use corrected prediction for final update
        result = sample + dt * pred_next

        # Clean up
        del sample_next, pred_next
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    @torch.no_grad()
    def edit_once_predictor_corrector(
        self,
        model,
        sample,
        t_curr: float,
        t_next: float,
        source_cond: torch.Tensor,
        target_cond: torch.Tensor,
        neg_cond: torch.Tensor,
        cfg_strength: float,
        cfg_interval: Tuple[float, float],
        omega: float,
        selector: Optional[torch.Tensor] = None,
    ):
        """Single editing step with predictor-corrector and UniEdit fusion.

        Args:
            model: Flow model
            sample: Current sample
            t_curr: Current timestep
            t_next: Next timestep
            source_cond: Source condition
            target_cond: Target condition
            neg_cond: Negative condition
            cfg_strength: CFG strength
            cfg_interval: CFG interval
            omega: UniEdit omega parameter
            selector: Optional selector for region control

        Returns:
            Updated sample
        """
        import gc

        dt = t_next - t_curr

        # Predictor: compute source and target velocities
        pred_tgt = self._guided_prediction_for_cond(
            model=model,
            sample=sample,
            t_value=t_curr,
            cond=target_cond,
            neg_cond=neg_cond,
            cfg_strength=cfg_strength,
            cfg_interval=cfg_interval,
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pred_src = self._guided_prediction_for_cond(
            model=model,
            sample=sample,
            t_value=t_curr,
            cond=source_cond,
            neg_cond=neg_cond,
            cfg_strength=cfg_strength,
            cfg_interval=cfg_interval,
        )

        # Compute UniEdit fusion
        guidance = pred_tgt - pred_src
        save_map = compute_uniedit_map(guidance, selector=selector)
        fused = pred_tgt * save_map + pred_src * (1.0 - save_map)
        pred = fused + guidance * ((1.0 + save_map) * float(omega))

        # Take a temporary step
        sample_next = sample + dt * pred

        # Clean up
        del pred_tgt, pred_src, guidance, save_map, fused, pred
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Corrector: compute source and target velocities at next point
        pred_tgt_next = self._guided_prediction_for_cond(
            model=model,
            sample=sample_next,
            t_value=t_next,
            cond=target_cond,
            neg_cond=neg_cond,
            cfg_strength=cfg_strength,
            cfg_interval=cfg_interval,
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pred_src_next = self._guided_prediction_for_cond(
            model=model,
            sample=sample_next,
            t_value=t_next,
            cond=source_cond,
            neg_cond=neg_cond,
            cfg_strength=cfg_strength,
            cfg_interval=cfg_interval,
        )

        # Compute UniEdit fusion with corrected predictions
        guidance_next = pred_tgt_next - pred_src_next
        save_map_next = compute_uniedit_map(guidance_next, selector=selector)
        fused_next = pred_tgt_next * save_map_next + pred_src_next * (1.0 - save_map_next)
        pred_next = fused_next + guidance_next * ((1.0 + save_map_next) * float(omega))

        # Use corrected prediction for final update
        result = sample + dt * pred_next

        # Clean up
        del sample_next, pred_tgt_next, pred_src_next, guidance_next, save_map_next, fused_next, pred_next
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    @torch.no_grad()
    def invert(
        self,
        model,
        sample,
        cond: torch.Tensor,
        neg_cond: torch.Tensor,
        steps: int,
        rescale_t: float,
        cfg_strength: float,
        cfg_interval: Tuple[float, float],
        alpha: float = 0.5,
        zero_init: bool = False,
        verbose: bool = True,
    ):
        """Run delayed inversion with predictor-corrector.

        Only inverts the first alpha fraction of steps.

        Args:
            model: Flow model
            sample: Initial sample (data)
            cond: Condition (typically empty for inversion)
            neg_cond: Negative condition
            steps: Number of inversion steps
            rescale_t: Time rescaling factor
            cfg_strength: CFG strength
            cfg_interval: CFG interval
            alpha: Fraction of steps to invert (0.5 = invert first half)
            zero_init: If True, use zero velocity for first prediction
            verbose: Show progress bar

        Returns:
            Partially inverted sample
        """
        t_seq = np.linspace(1.0, 0.0, int(steps) + 1)
        t_seq = rescale_t * t_seq / (1.0 + (rescale_t - 1.0) * t_seq)
        t_seq = t_seq[::-1]  # Reverse for inversion
        t_pairs = [(float(t_seq[i]), float(t_seq[i + 1])) for i in range(len(t_seq) - 1)]

        # Compute step threshold
        num_pairs = len(t_pairs)
        step_threshold = round(alpha * num_pairs)

        for i, (t_curr, t_next) in enumerate(tqdm(t_pairs, desc="UniEdit Euler inversion", disable=not verbose)):
            if i < step_threshold:
                # Only invert first alpha fraction
                sample = self.invert_once_predictor_corrector(
                    model=model,
                    sample=sample,
                    t_curr=t_curr,
                    t_next=t_next,
                    cond=cond,
                    neg_cond=neg_cond,
                    cfg_strength=cfg_strength,
                    cfg_interval=cfg_interval,
                    zero_init=(zero_init and i == 0),
                )
            # else: skip remaining steps (delayed inversion)

        return sample

    @torch.no_grad()
    def edit(
        self,
        model,
        sample,
        source_cond: torch.Tensor,
        target_cond: torch.Tensor,
        neg_cond: torch.Tensor,
        steps: int,
        rescale_t: float,
        cfg_strength: float,
        cfg_interval: Tuple[float, float],
        omega: float,
        alpha: float = 0.5,
        selector: Optional[torch.Tensor] = None,
        verbose: bool = True,
    ):
        """Run delayed editing with predictor-corrector and UniEdit fusion.

        Only edits the last alpha fraction of steps.

        Args:
            model: Flow model
            sample: Initial sample (partially inverted)
            source_cond: Source condition
            target_cond: Target condition
            neg_cond: Negative condition
            steps: Number of editing steps
            rescale_t: Time rescaling factor
            cfg_strength: CFG strength
            cfg_interval: CFG interval
            omega: UniEdit omega parameter
            alpha: Fraction of steps to edit (0.5 = edit last half)
            selector: Optional selector for region control
            verbose: Show progress bar

        Returns:
            Edited sample
        """
        t_seq = np.linspace(1.0, 0.0, int(steps) + 1)
        t_seq = rescale_t * t_seq / (1.0 + (rescale_t - 1.0) * t_seq)
        t_pairs = [(float(t_seq[i]), float(t_seq[i + 1])) for i in range(len(t_seq) - 1)]

        # Compute step threshold
        num_pairs = len(t_pairs)
        step_threshold = round(alpha * num_pairs)

        for i, (t_curr, t_next) in enumerate(tqdm(t_pairs, desc="UniEdit Euler editing", disable=not verbose)):
            if i >= num_pairs - step_threshold:
                # Only edit last alpha fraction
                sample = self.edit_once_predictor_corrector(
                    model=model,
                    sample=sample,
                    t_curr=t_curr,
                    t_next=t_next,
                    source_cond=source_cond,
                    target_cond=target_cond,
                    neg_cond=neg_cond,
                    cfg_strength=cfg_strength,
                    cfg_interval=cfg_interval,
                    omega=omega,
                    selector=selector,
                )
            # else: skip early steps (delayed editing)

        return sample
