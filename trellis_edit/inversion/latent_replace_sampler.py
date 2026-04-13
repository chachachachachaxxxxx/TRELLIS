"""Sparse Latent Replacement RF-Solver for latent_replace_union ablation."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .rf_sampler import RFSolverSampler


def _time_key(t_value: float) -> str:
    """Convert time value to cache key."""
    return repr(float(t_value))


def build_rf_t_pairs(steps: int, rescale_t: float, inverse: bool) -> List[Tuple[float, float]]:
    """Build time pairs for RF sampling.

    Args:
        steps: Number of sampling steps
        rescale_t: Time rescaling factor
        inverse: If True, reverse the sequence (for inversion)

    Returns:
        List of (t_curr, t_next) pairs
    """
    t_seq = np.linspace(1.0, 0.0, int(steps) + 1)
    t_seq = rescale_t * t_seq / (1.0 + (rescale_t - 1.0) * t_seq)
    if inverse:
        t_seq = t_seq[::-1]
    return [tuple(map(float, (t_seq[i], t_seq[i + 1]))) for i in range(len(t_seq) - 1)]


@torch.no_grad()
def apply_sparse_latent_replacement(
    sample,
    cached_latent,
    target_indices: torch.Tensor,
    source_indices: torch.Tensor,
):
    """Replace latent features at target indices with cached features.

    Args:
        sample: Current sample (SparseTensor)
        cached_latent: Cached latent from inversion (SparseTensor, on CPU)
        target_indices: Indices in target to replace
        source_indices: Corresponding indices in cached latent

    Returns:
        Sample with replaced features
    """
    if target_indices.numel() == 0:
        return sample
    feats = sample.feats.clone()
    # Load cached features from CPU to GPU
    cached_feats = cached_latent.feats[source_indices].to(device=feats.device, dtype=feats.dtype)
    feats[target_indices] = cached_feats
    return sample.replace(feats)


class SparseLatentReplaceRFSolver(RFSolverSampler):
    """RF-Solver with sparse latent replacement for latent_replace_union ablation.

    This sampler caches intermediate latents during inversion (on CPU to save GPU memory),
    then replaces specific latent features during denoising to preserve source regions.

    All caching is done on CPU following VoxHammer's implementation strategy.
    """

    @torch.no_grad()
    def invert_with_cache(
        self,
        model,
        sample,
        cond_dict: dict,
        steps: int,
        rescale_t: float,
        cfg_strength: float,
        cfg_interval: Tuple[float, float],
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Invert sample and cache intermediate latents.

        Args:
            model: Flow model
            sample: Sample to invert (SparseTensor)
            cond_dict: Condition dictionary with 'cond' and 'neg_cond'
            steps: Number of inversion steps
            rescale_t: Time rescaling factor
            cfg_strength: CFG strength
            cfg_interval: CFG interval (start, end)
            verbose: Show progress bar

        Returns:
            Tuple of (terminal_noise, latent_cache)
            - terminal_noise: Inverted noise at t=1
            - latent_cache: Dict mapping time keys to cached latents (on CPU)
        """
        import gc

        latent_cache = {}
        t_pairs = build_rf_t_pairs(steps=steps, rescale_t=rescale_t, inverse=True)

        for step_idx, (t_curr, t_next) in enumerate(tqdm(t_pairs, desc="RF-Solver inversion (cache stage2 latents)", disable=not verbose)):
            # Perform one inversion step
            sample = self.sample_once(model, sample, t_curr, t_next, cond_dict, cfg_strength, cfg_interval)

            # Cache to CPU immediately (following VoxHammer's strategy)
            cached = sample.detach()
            latent_cache[_time_key(t_next)] = cached.cpu()
            del cached

            # Aggressive GPU memory cleanup every step
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Python GC every 5 steps
            if (step_idx + 1) % 5 == 0:
                gc.collect()

        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return sample, latent_cache

    @torch.no_grad()
    def sample_with_replacement(
        self,
        model,
        sample,
        cond_dict: dict,
        steps: int,
        rescale_t: float,
        cfg_strength: float,
        cfg_interval: Tuple[float, float],
        latent_cache: Dict[str, torch.Tensor],
        replace_target_indices: torch.Tensor,
        replace_source_indices: torch.Tensor,
        verbose: bool = True,
    ):
        """Sample with latent replacement at each step.

        Args:
            model: Flow model
            sample: Initial sample (terminal noise)
            cond_dict: Condition dictionary with 'cond' and 'neg_cond'
            steps: Number of sampling steps
            rescale_t: Time rescaling factor
            cfg_strength: CFG strength
            cfg_interval: CFG interval (start, end)
            latent_cache: Cached latents from inversion
            replace_target_indices: Indices in target to replace
            replace_source_indices: Corresponding indices in cached latents
            verbose: Show progress bar

        Returns:
            Denoised sample
        """
        import gc

        t_pairs = build_rf_t_pairs(steps=steps, rescale_t=rescale_t, inverse=False)

        for step_idx, (t_curr, t_next) in enumerate(tqdm(t_pairs, desc="RF-Solver denoise (latent replacement)", disable=not verbose)):
            # Replace latents before sampling (preserve source regions)
            if replace_target_indices.numel() > 0:
                cached_latent = latent_cache.get(_time_key(t_curr))
                # Load from CPU and replace
                sample = apply_sparse_latent_replacement(
                    sample=sample,
                    cached_latent=cached_latent,  # On CPU
                    target_indices=replace_target_indices,
                    source_indices=replace_source_indices,
                )

            # Perform one sampling step
            sample = self.sample_once(model, sample, t_curr, t_next, cond_dict, cfg_strength, cfg_interval)

            # Aggressive GPU memory cleanup every step
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Python GC every 5 steps
            if (step_idx + 1) % 5 == 0:
                gc.collect()

        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return sample
