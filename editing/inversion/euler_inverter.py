"""First-order Euler sampler for inversion testing."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .base_inverter import BaseInverter, InversionResult, InversionTrajectory


class EulerSampler:
    """First-order Euler sampler (baseline for comparison)."""

    @torch.no_grad()
    def _run_model(self, model, sample, t_value: float, cond: Optional[torch.Tensor]):
        """Run model with given timestep and condition."""
        t = torch.tensor([1000.0 * t_value] * sample.shape[0], device=sample.device, dtype=torch.float32)
        if cond is not None and cond.shape[0] == 1 and sample.shape[0] > 1:
            cond = cond.repeat(sample.shape[0], *([1] * (cond.ndim - 1)))
        return model(sample, t, cond)

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
        """Compute CFG-guided prediction."""
        import gc

        if cfg_interval[0] <= t_value <= cfg_interval[1] and cfg_strength > 0.0:
            pred = self._run_model(model, sample, t_value, cond_dict["cond"])
            neg_pred = self._run_model(model, sample, t_value, cond_dict["neg_cond"])
            result = (1.0 + cfg_strength) * pred - cfg_strength * neg_pred
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
        """Single first-order Euler step: x_{t+h} = x_t + h·f(x_t, t)."""
        pred = self._guided_prediction(model, sample, t_curr, cond_dict, cfg_strength, cfg_interval)
        dt = t_next - t_curr
        return sample + dt * pred

    @torch.no_grad()
    def sample_with_trajectory(
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
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[float]]:
        """Run Euler sampling with trajectory recording.

        Returns:
            Tuple of (final_sample, trajectory, timesteps)
        """
        t_seq = np.linspace(1.0, 0.0, int(steps) + 1)
        t_seq = rescale_t * t_seq / (1.0 + (rescale_t - 1.0) * t_seq)
        if inverse:
            t_seq = t_seq[::-1]
            desc = "Euler inversion"
        else:
            desc = "Euler denoise"

        t_pairs = [(float(t_seq[i]), float(t_seq[i + 1])) for i in range(len(t_seq) - 1)]

        trajectory = [sample.clone()]
        timesteps = [t_seq[0]]

        for t_curr, t_next in tqdm(t_pairs, desc=desc, disable=not verbose):
            sample = self.sample_once(model, sample, t_curr, t_next, cond_dict, cfg_strength, cfg_interval)
            trajectory.append(sample.clone())
            timesteps.append(t_next)

        return sample, trajectory, timesteps


class EulerInverter(BaseInverter):
    """First-order Euler inverter with optional corrector."""

    def __init__(self, corrector_steps: int = 0):
        """Initialize Euler inverter.

        Args:
            corrector_steps: Number of corrector iterations per step (0 = no correction)
        """
        super().__init__(f"euler_first_order_c{corrector_steps}")
        self.sampler = EulerSampler()
        self.corrector_steps = corrector_steps

    @torch.no_grad()
    def _corrector_step(
        self,
        model,
        sample,
        t_value: float,
        cond_dict: dict,
        cfg_strength: float,
        cfg_interval: Tuple[float, float],
        noise_scale: float = 0.01,
    ):
        """Corrector step: Langevin-like correction."""
        pred = self.sampler._guided_prediction(model, sample, t_value, cond_dict, cfg_strength, cfg_interval)
        correction = noise_scale * pred
        return sample + correction

    @torch.no_grad()
    def _sample_once_with_correction(
        self,
        model,
        sample,
        t_curr: float,
        t_next: float,
        cond_dict: dict,
        cfg_strength: float,
        cfg_interval: Tuple[float, float],
    ):
        """Single step with predictor-corrector."""
        # Predictor (Euler step)
        sample = self.sampler.sample_once(model, sample, t_curr, t_next, cond_dict, cfg_strength, cfg_interval)

        # Corrector iterations
        for _ in range(self.corrector_steps):
            sample = self._corrector_step(model, sample, t_next, cond_dict, cfg_strength, cfg_interval)

        return sample

    @torch.no_grad()
    def _sample_with_trajectory(
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
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[float]]:
        """Run Euler sampling with trajectory recording."""
        t_seq = np.linspace(1.0, 0.0, int(steps) + 1)
        t_seq = rescale_t * t_seq / (1.0 + (rescale_t - 1.0) * t_seq)
        if inverse:
            t_seq = t_seq[::-1]
            desc = f"Euler inversion (C={self.corrector_steps})"
        else:
            desc = f"Euler denoise (C={self.corrector_steps})"

        t_pairs = [(float(t_seq[i]), float(t_seq[i + 1])) for i in range(len(t_seq) - 1)]

        trajectory = [sample.clone()]
        timesteps = [t_seq[0]]

        for t_curr, t_next in tqdm(t_pairs, desc=desc, disable=not verbose):
            if self.corrector_steps > 0:
                sample = self._sample_once_with_correction(
                    model, sample, t_curr, t_next, cond_dict, cfg_strength, cfg_interval
                )
            else:
                sample = self.sampler.sample_once(
                    model, sample, t_curr, t_next, cond_dict, cfg_strength, cfg_interval
                )
            trajectory.append(sample.clone())
            timesteps.append(t_next)

        return sample, trajectory, timesteps

    def invert(
        self,
        model,
        sample,
        cond_dict: dict,
        steps: int,
        rescale_t: float,
        cfg_strength: float,
        cfg_interval: Tuple[float, float],
        verbose: bool = True,
    ) -> InversionResult:
        """Invert using first-order Euler method."""
        # Inverse: data → noise
        terminal_noise, inv_traj, inv_timesteps = self._sample_with_trajectory(
            model=model,
            sample=sample,
            cond_dict=cond_dict,
            steps=steps,
            rescale_t=rescale_t,
            cfg_strength=cfg_strength,
            cfg_interval=cfg_interval,
            inverse=True,
            verbose=verbose,
        )

        # Forward: noise → data
        reconstructed, fwd_traj, fwd_timesteps = self._sample_with_trajectory(
            model=model,
            sample=terminal_noise,
            cond_dict=cond_dict,
            steps=steps,
            rescale_t=rescale_t,
            cfg_strength=cfg_strength,
            cfg_interval=cfg_interval,
            inverse=False,
            verbose=verbose,
        )

        return InversionResult(
            terminal_noise=terminal_noise,
            inverse_trajectory=InversionTrajectory(
                latents=inv_traj,
                timesteps=inv_timesteps,
                metadata={"method": self.name, "steps": steps, "corrector_steps": self.corrector_steps},
            ),
            forward_trajectory=InversionTrajectory(
                latents=fwd_traj,
                timesteps=fwd_timesteps,
                metadata={"method": self.name, "steps": steps, "corrector_steps": self.corrector_steps},
            ),
        )

    def get_config(self) -> Dict:
        """Get configuration."""
        return {
            "name": self.name,
            "order": 1,
            "corrector_steps": self.corrector_steps,
            "description": f"First-order Euler method with {self.corrector_steps} corrector steps",
        }
