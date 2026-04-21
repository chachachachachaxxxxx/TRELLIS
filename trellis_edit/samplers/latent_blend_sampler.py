"""Custom samplers with per-step latent blending (VoxHammer-style).

Implements latent blending at each denoising step:
- Before calling the model, blend current latent with cached source latent
- Supports both SS stage (dense latents) and SLAT stage (sparse features)
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from easydict import EasyDict as edict

from trellis.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler
from trellis_edit.composable.runtime import SourceTrace


@dataclass(frozen=True)
class SparseLatentBlendMask:
    coords: torch.Tensor
    edit_weights: Optional[torch.Tensor] = None


def _resolve_sparse_latent_mask(
    latent_mask: torch.Tensor | SparseLatentBlendMask,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(latent_mask, SparseLatentBlendMask):
        return latent_mask.coords, latent_mask.edit_weights
    return latent_mask, None


def blend_sparse_features(
    sample,
    source_latent,
    latent_mask: torch.Tensor | SparseLatentBlendMask,
) -> tuple[Any, dict[str, int]]:
    preserve_coords, edit_weights = _resolve_sparse_latent_mask(latent_mask)
    if preserve_coords.shape[0] == 0:
        return sample, {
            "preserve_coords": 0,
            "sample_matches": 0,
            "source_matches": 0,
            "valid_matches": 0,
        }

    source_latent = source_latent.to(sample.coords.device)
    preserve_coords = preserve_coords.to(sample.coords.device)

    match_sample = (sample.coords.unsqueeze(1) == preserve_coords.unsqueeze(0)).all(dim=-1)
    match_source = (source_latent.coords.unsqueeze(1) == preserve_coords.unsqueeze(0)).all(dim=-1)

    preserve_in_sample = match_sample.any(dim=0)
    preserve_in_source = match_source.any(dim=0)
    valid_matches = preserve_in_sample & preserve_in_source
    if not valid_matches.any():
        return sample, {
            "preserve_coords": int(preserve_coords.shape[0]),
            "sample_matches": int(preserve_in_sample.sum().item()),
            "source_matches": int(preserve_in_source.sum().item()),
            "valid_matches": 0,
        }

    idx_sample = match_sample[:, valid_matches].float().argmax(0)
    idx_source = match_source[:, valid_matches].float().argmax(0)

    feats = sample.feats.clone()
    source_feats = source_latent.feats.to(feats.device, feats.dtype)[idx_source]

    if edit_weights is None:
        feats[idx_sample] = source_feats
    else:
        blend_weights = edit_weights.to(device=feats.device, dtype=feats.dtype)
        if blend_weights.ndim == 1:
            blend_weights = blend_weights.unsqueeze(1)
        blend_weights = blend_weights[valid_matches]
        current_feats = feats[idx_sample].clone()
        feats[idx_sample] = current_feats * blend_weights + source_feats * (1 - blend_weights)

    return sample.replace(feats), {
        "preserve_coords": int(preserve_coords.shape[0]),
        "sample_matches": int(preserve_in_sample.sum().item()),
        "source_matches": int(preserve_in_source.sum().item()),
        "valid_matches": int(valid_matches.sum().item()),
    }


class LatentBlendFlowEulerSampler(FlowEulerGuidanceIntervalSampler):
    """Flow Euler sampler with per-step latent blending.

    At each denoising step, before calling the model:
    1. Blend current latent with source latent based on mask
    2. Then proceed with normal Euler sampling

    This implements VoxHammer's latent replacement strategy (Eq. 4 and 5).
    """

    def __init__(self, sigma_min: float):
        super().__init__(sigma_min)

        # Blending state
        self.source_trace: Optional[SourceTrace] = None  # Canonical source trace at each logical timestep
        self.latent_mask: Optional[torch.Tensor | SparseLatentBlendMask] = None  # Blending mask
        self.blend_enabled: bool = False
        self.is_sparse: bool = False  # True for SLAT stage, False for SS stage

    def set_blend_source(
        self,
        source_trace: SourceTrace,
        latent_mask: torch.Tensor | SparseLatentBlendMask,
        is_sparse: bool = False,
    ):
        """Set source latent cache and mask for blending.

        Args:
            source_trace: Canonical source trace keyed by logical timestep
            latent_mask: Blending mask
                - SS stage: Tensor [B, C, D, H, W], 0=preserve source, 1=use edit
                - SLAT stage: Tensor [N_preserve, 4] or SparseLatentBlendMask
            is_sparse: True for SLAT stage, False for SS stage
        """
        self.source_trace = source_trace
        self.latent_mask = latent_mask
        self.blend_enabled = True
        self.is_sparse = is_sparse

        # Initialize statistics
        self.stats = {
            'blend_calls': 0,
            'last_sample_voxels': 0,
            'last_source_voxels': 0,
            'last_preserve_coords': 0,
            'last_sample_match': 0,
            'last_source_match': 0,
            'last_valid_match': 0,
        }

    def disable_blend(self):
        """Disable blending."""
        self.blend_enabled = False
        self.source_trace = None
        self.latent_mask = None

    def _blend_latent(self, sample: Any, t_norm: float) -> Any:
        """Blend current latent with cached source latent.

        Args:
            sample: Current latent at timestep t
            t_norm: Normalized timestep (0-1)

        Returns:
            Blended latent
        """
        if not self.blend_enabled or self.source_trace is None:
            return sample

        source_latent = self.source_trace.get_sample(t_norm)
        if source_latent is None:
            return sample

        if self.is_sparse:
            # SLAT stage: Sparse feature blending (Eq. 5)
            return self._blend_sparse(sample, source_latent)
        else:
            # SS stage: Dense latent blending (Eq. 4)
            return self._blend_dense(sample, source_latent)

    def _blend_dense(self, sample: torch.Tensor, source_latent: torch.Tensor) -> torch.Tensor:
        """Blend dense latents (SS stage).

        Implements VoxHammer Eq. 4:
            z_t ← M ⊙ z_t + (1-M) ⊙ ẑ_t

        where:
            z_t: current latent (edit)
            ẑ_t: source latent (cached from inversion)
            M: mask (1=edit, 0=preserve)
        """
        # sample: [B, C, D, H, W]
        # source_latent: [B, C, D, H, W] (from cache)
        # latent_mask: [B, C, D, H, W]

        source = source_latent.to(sample.device, sample.dtype)
        mask = self.latent_mask.to(sample.device, sample.dtype)

        # Blend: edit * mask + source * (1 - mask)
        blended = sample * mask + source * (1 - mask)

        return blended

    def _blend_sparse(self, sample, source_latent) -> Any:
        """Blend sparse features (SLAT stage).

        Implements VoxHammer Eq. 5:
            ∀u ∈ Ωkeep: z_t[u] ← ẑ_t[u]

        where:
            Ωkeep: preserve region (coords in latent_mask)
            z_t[u]: current feature at voxel u
            ẑ_t[u]: source feature at voxel u (from cache)
        """
        # sample: SparseTensor
        # source_latent: SparseTensor (from cache, on CPU)
        # latent_mask: Tensor [N_preserve, 4] (coords of preserve region)

        preserve_coords, _ = _resolve_sparse_latent_mask(self.latent_mask)
        if preserve_coords.shape[0] == 0:
            return sample

        sample, blend_stats = blend_sparse_features(sample, source_latent, self.latent_mask)

        # Update statistics
        if hasattr(self, "stats"):
            self.stats["blend_calls"] += 1
            self.stats["last_sample_voxels"] = sample.coords.shape[0]
            self.stats["last_source_voxels"] = source_latent.coords.shape[0]
            self.stats["last_preserve_coords"] = blend_stats["preserve_coords"]
            self.stats["last_sample_match"] = blend_stats["sample_matches"]
            self.stats["last_source_match"] = blend_stats["source_matches"]
            self.stats["last_valid_match"] = blend_stats["valid_matches"]

        return sample

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        """Single sampling step with latent blending.

        Key difference from base class:
        1. Blend x_t with source latent BEFORE calling model
        2. Then proceed with normal Euler sampling
        """
        # Blend current latent with source (VoxHammer Eq. 4/5)
        t_norm = t  # t is already normalized (0-1)
        x_t = self._blend_latent(x_t, t_norm)

        # Normal Euler sampling
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v

        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        """Generate samples with latent blending.

        At each step:
        1. Blend current latent with source
        2. Call model to get velocity
        3. Update latent
        """
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))

        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})

        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)

        ret.samples = sample
        return ret


class LatentBlendFlowEulerCfgSampler(LatentBlendFlowEulerSampler):
    """Latent blend sampler with CFG support."""

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        """Sample with CFG."""
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))

        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})

        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            # Blend latent
            t_norm = t
            sample = self._blend_latent(sample, t_norm)

            # CFG
            pred_x_0_cond, pred_eps_cond, pred_v_cond = self._get_model_prediction(
                model, sample, t, cond, **kwargs
            )
            pred_x_0_neg, pred_eps_neg, pred_v_neg = self._get_model_prediction(
                model, sample, t, neg_cond, **kwargs
            )

            # Apply CFG
            pred_v = pred_v_cond + cfg_strength * (pred_v_cond - pred_v_neg)
            pred_x_0 = pred_x_0_cond + cfg_strength * (pred_x_0_cond - pred_x_0_neg)

            # Update
            sample = sample - (t - t_prev) * pred_v
            ret.pred_x_t.append(sample)
            ret.pred_x_0.append(pred_x_0)

        ret.samples = sample
        return ret


class LatentBlendFlowEulerGuidanceIntervalSampler(LatentBlendFlowEulerCfgSampler):
    """Latent blend sampler with guidance interval (late-time CFG)."""

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """Sample with guidance interval."""
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))

        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})

        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            # Blend latent
            t_norm = t
            sample = self._blend_latent(sample, t_norm)

            # Check if in CFG interval
            if cfg_interval[0] <= t <= cfg_interval[1]:
                # Apply CFG
                pred_x_0_cond, pred_eps_cond, pred_v_cond = self._get_model_prediction(
                    model, sample, t, cond, neg_cond=neg_cond,
                    cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs
                )
                pred_x_0_neg, pred_eps_neg, pred_v_neg = self._get_model_prediction(
                    model, sample, t, neg_cond, neg_cond=neg_cond,
                    cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs
                )
                pred_v = pred_v_cond + cfg_strength * (pred_v_cond - pred_v_neg)
                pred_x_0 = pred_x_0_cond + cfg_strength * (pred_x_0_cond - pred_x_0_neg)
            else:
                # No CFG
                pred_x_0, pred_eps, pred_v = self._get_model_prediction(
                    model, sample, t, cond, neg_cond=neg_cond,
                    cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs
                )

            # Update
            sample = sample - (t - t_prev) * pred_v
            ret.pred_x_t.append(sample)
            ret.pred_x_0.append(pred_x_0)

        ret.samples = sample
        return ret
