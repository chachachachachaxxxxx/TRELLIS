from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tqdm import tqdm


class FlowEditSampler:
    """Sparse-structure FlowEdit sampler.

    This follows Nano3D's stage-1 idea: start from the source sparse-structure
    latent, estimate a source/target velocity delta at each timestep, and apply
    that delta directly in latent space without inversion.
    """

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
        start_step: int,
        n_avg: int,
        source_cfg_strength: float,
        target_cfg_strength: float,
        cfg_interval: tuple[float, float],
        verbose: bool = True,
    ) -> torch.Tensor:
        if source_latent.shape[0] != 1:
            raise RuntimeError(
                "FlowEditSampler currently expects a single source latent; "
                f"got batch size {source_latent.shape[0]}."
            )

        total_steps = int(steps)
        start_step = int(start_step)
        if total_steps <= 0:
            raise RuntimeError(f"FlowEdit steps must be > 0, got {total_steps}")
        if start_step < 0 or start_step > total_steps:
            raise RuntimeError(f"FlowEdit start_step must be in [0, {total_steps}], got {start_step}")
        if n_avg <= 0:
            raise RuntimeError(f"FlowEdit n_avg must be > 0, got {n_avg}")

        output_dtype = source_latent.dtype
        source_latent = source_latent.to(dtype=torch.float32)
        edited_latent = source_latent.clone()

        t_seq = np.linspace(1, 0, total_steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((float(t_seq[i]), float(t_seq[i + 1])) for i in range(total_steps))

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

        for step_idx, (t, t_prev) in enumerate(tqdm(t_pairs, desc="FlowEdit SS", disable=not verbose)):
            if step_idx < start_step:
                continue

            velocity_delta = torch.zeros_like(source_latent)
            for _ in range(n_avg):
                forward_noise = torch.randn_like(source_latent)
                latent_source_t = (1.0 - t) * source_latent + t * forward_noise
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
                velocity_delta += (velocity_target - velocity_source) / float(n_avg)

            edited_latent = edited_latent + (t_prev - t) * velocity_delta

        return edited_latent.to(dtype=output_dtype)
