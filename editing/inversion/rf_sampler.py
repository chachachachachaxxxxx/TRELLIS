"""Second-order Rectified Flow sampler for RF inversion."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


class SecondOrderRFSampler:
    """Taylor-improved second-order flow sampler inspired by VoxHammer's RF-Solver.

    This version intentionally keeps the original TRELLIS model forward signature
    and only adds:
    - inverse sampling (data -> terminal noise)
    - second-order midpoint correction
    - late-time CFG on t in [cfg_interval[0], cfg_interval[1]]
    """

    def _run_model(self, model, sample, t_value: float, cond: Optional[torch.Tensor]):
        """Run model with given timestep and condition."""
        t = torch.tensor([1000.0 * t_value] * sample.shape[0], device=sample.device, dtype=torch.float32)
        if cond is not None and cond.shape[0] == 1 and sample.shape[0] > 1:
            cond = cond.repeat(sample.shape[0], *([1] * (cond.ndim - 1)))
        return model(sample, t, cond)

    def _guided_prediction(
        self,
        model,
        sample,
        t_value: float,
        cond_dict: dict,
        cfg_strength: float,
        cfg_interval: Tuple[float, float],
    ):
        """Compute CFG-guided prediction with memory optimization."""
        import gc

        if cfg_interval[0] <= t_value <= cfg_interval[1] and cfg_strength > 0.0:
            # Run positive prediction
            pred = self._run_model(model, sample, t_value, cond_dict["cond"])

            # Clean up before negative prediction
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Run negative prediction
            neg_pred = self._run_model(model, sample, t_value, cond_dict["neg_cond"])

            # Compute guided prediction
            result = (1.0 + cfg_strength) * pred - cfg_strength * neg_pred

            # Clean up intermediate tensors
            del pred, neg_pred
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return result

        return self._run_model(model, sample, t_value, cond_dict["cond"])

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
        """Single sampling step with second-order correction and memory optimization."""
        import gc

        # First prediction
        pred = self._guided_prediction(model, sample, t_curr, cond_dict, cfg_strength, cfg_interval)
        dt = t_next - t_curr

        # Compute midpoint sample
        sample_mid = sample + 0.5 * dt * pred
        t_mid = t_curr + 0.5 * dt

        # Clean up before midpoint prediction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Midpoint prediction
        pred_mid = self._guided_prediction(model, sample_mid, t_mid, cond_dict, cfg_strength, cfg_interval)

        # Compute second-order correction
        first_order = (pred_mid - pred) / (0.5 * dt)
        result = sample + dt * pred - 0.5 * (dt**2) * first_order

        # Clean up intermediate tensors
        del pred, sample_mid, pred_mid, first_order
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

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
        verbose: bool = True,
    ):
        """Run full sampling process.

        Args:
            model: Flow model
            sample: Initial sample (data for inverse, noise for forward)
            cond_dict: Condition dictionary with 'cond' and 'neg_cond'
            steps: Number of sampling steps
            rescale_t: Time rescaling factor
            cfg_strength: CFG strength
            cfg_interval: CFG interval (start, end)
            inverse: If True, run inverse (data -> noise), else forward (noise -> data)
            verbose: Show progress bar

        Returns:
            Final sample
        """
        # RF inversion/denoising is inference-only, no gradients needed
        with torch.no_grad():
            t_seq = np.linspace(1.0, 0.0, int(steps) + 1)
            t_seq = rescale_t * t_seq / (1.0 + (rescale_t - 1.0) * t_seq)
            if inverse:
                t_seq = t_seq[::-1]
                desc = "RF inversion"
            else:
                desc = "RF denoise"
            t_pairs = [(float(t_seq[i]), float(t_seq[i + 1])) for i in range(len(t_seq) - 1)]
            for t_curr, t_next in tqdm(t_pairs, desc=desc, disable=not verbose):
                sample = self.sample_once(model, sample, t_curr, t_next, cond_dict, cfg_strength, cfg_interval)
        return sample
