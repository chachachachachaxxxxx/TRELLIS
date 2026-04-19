"""UniEdit RF-Solver for two-stage editing with source/target fusion."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from tqdm import tqdm

from .rf_sampler import RFSolverSampler, build_denoise_t_pairs


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


class UniEditRFSolver(RFSolverSampler):
    """RF-Solver with UniEdit source/target velocity fusion.

    Supports three modes:
    - full_uniedit: Complete UniEdit fusion with omega guidance
    - preserve_overlap: Preserve overlapping regions, free generation for new regions
    - target_only: Only use target branch (no fusion)
    """

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
    def _merged_prediction(
        self,
        model,
        sample,
        t_value: float,
        source_cond: torch.Tensor,
        target_cond: torch.Tensor,
        neg_cond: torch.Tensor,
        cfg_strength: float,
        cfg_interval: Tuple[float, float],
        omega: float,
        selector: Optional[torch.Tensor],
        mode: str,
    ):
        """Compute merged prediction with source/target fusion.

        Args:
            model: Flow model
            sample: Current sample
            t_value: Current timestep
            source_cond: Source condition
            target_cond: Target condition
            neg_cond: Negative condition
            cfg_strength: CFG strength
            cfg_interval: CFG interval (start, end)
            omega: UniEdit omega parameter
            selector: Optional selector for preserve_overlap mode
            mode: Fusion mode (full_uniedit, preserve_overlap, target_only)

        Returns:
            Merged velocity prediction
        """
        # Compute target prediction
        pred_tgt = self._guided_prediction_for_cond(
            model=model,
            sample=sample,
            t_value=t_value,
            cond=target_cond,
            neg_cond=neg_cond,
            cfg_strength=cfg_strength,
            cfg_interval=cfg_interval,
        )
        if mode == "target_only":
            return pred_tgt

        # Clean up before source prediction
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Compute source prediction
        pred_src = self._guided_prediction_for_cond(
            model=model,
            sample=sample,
            t_value=t_value,
            cond=source_cond,
            neg_cond=neg_cond,
            cfg_strength=cfg_strength,
            cfg_interval=cfg_interval,
        )

        # Compute guidance and fusion
        guidance = pred_tgt - pred_src
        save_map = compute_uniedit_map(guidance, selector=selector if mode == "preserve_overlap" else None)
        fused = pred_tgt * save_map + pred_src * (1.0 - save_map)
        pred = fused + guidance * ((1.0 + save_map) * float(omega))
        if mode == "preserve_overlap" and selector is not None:
            pred = pred * selector + pred_tgt * (1.0 - selector)

        # Clean up intermediate tensors
        del pred_src, pred_tgt, guidance, save_map, fused
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return pred

    @torch.no_grad()
    def sample_once(
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
        selector: Optional[torch.Tensor],
        mode: str,
    ):
        """Perform one UniEdit sampling step with second-order integration."""
        # First prediction
        pred = self._merged_prediction(
            model=model,
            sample=sample,
            t_value=t_curr,
            source_cond=source_cond,
            target_cond=target_cond,
            neg_cond=neg_cond,
            cfg_strength=cfg_strength,
            cfg_interval=cfg_interval,
            omega=omega,
            selector=selector,
            mode=mode,
        )

        # Compute midpoint
        dt = t_next - t_curr
        sample_mid = sample + 0.5 * dt * pred
        t_mid = t_curr + 0.5 * dt

        # Clean up before midpoint prediction
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Midpoint prediction
        pred_mid = self._merged_prediction(
            model=model,
            sample=sample_mid,
            t_value=t_mid,
            source_cond=source_cond,
            target_cond=target_cond,
            neg_cond=neg_cond,
            cfg_strength=cfg_strength,
            cfg_interval=cfg_interval,
            omega=omega,
            selector=selector,
            mode=mode,
        )

        # Keep the same first_order convention as the shared RF solver:
        # first_order := (pred_mid - pred) / (dt / 2), so the Taylor
        # correction uses a plus sign.
        first_order = (pred_mid - pred) / (0.5 * dt)
        result = sample + dt * pred + 0.5 * (dt ** 2) * first_order

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
        source_cond: torch.Tensor,
        target_cond: torch.Tensor,
        neg_cond: torch.Tensor,
        steps: int,
        rescale_t: float,
        cfg_strength: float,
        cfg_interval: Tuple[float, float],
        omega: float,
        start_step: Optional[int] = None,
        selector: Optional[torch.Tensor] = None,
        mode: str = "full_uniedit",
        verbose: bool = True,
    ):
        """Run complete UniEdit sampling.

        Args:
            model: Flow model
            sample: Initial sample (terminal noise)
            source_cond: Source condition
            target_cond: Target condition
            neg_cond: Negative condition
            steps: Number of sampling steps
            rescale_t: Time rescaling factor
            cfg_strength: CFG strength
            cfg_interval: CFG interval (start, end)
            omega: UniEdit omega parameter
            start_step: Optional intermediate timestep index to start denoising from
            selector: Optional selector for preserve_overlap mode
            mode: Fusion mode (full_uniedit, preserve_overlap, target_only)
            verbose: Show progress bar

        Returns:
            Denoised sample
        """
        t_pairs, _ = build_denoise_t_pairs(
            steps=steps,
            rescale_t=rescale_t,
            start_step=start_step,
        )

        desc_map = {
            "full_uniedit": "UniEdit RF-Solver denoise",
            "preserve_overlap": "UniEdit RF-Solver denoise (preserve overlap)",
            "target_only": "Target-only RF-Solver denoise",
        }

        for t_curr, t_next in tqdm(t_pairs, desc=desc_map.get(mode, "UniEdit RF-Solver denoise"), disable=not verbose):
            sample = self.sample_once(
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
                    mode=mode,
                )
        return sample
