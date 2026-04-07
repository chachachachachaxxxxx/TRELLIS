"""RF (Rectified Flow) inversion utilities for TRELLIS editing."""
from __future__ import annotations

from typing import Tuple

import torch


def get_slat_norm_tensors(pipeline, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get SLAT normalization tensors (mean and std).

    Args:
        pipeline: TRELLIS pipeline
        device: Target device
        dtype: Target dtype

    Returns:
        Tuple of (mean, std) tensors
    """
    mean = pipeline.slat_normalization["mean"].to(device=device, dtype=dtype)
    std = pipeline.slat_normalization["std"].to(device=device, dtype=dtype)
    return mean, std


def invert_sparse_structure(
    pipeline,
    cond_src: dict,
    voxel_src: torch.Tensor,
    params: dict,
    cfg_interval: Tuple[float, float],
    verbose: bool = False,
) -> torch.Tensor:
    """Invert sparse structure to terminal noise.

    Args:
        pipeline: TRELLIS pipeline
        cond_src: Source condition dictionary
        voxel_src: Source voxel tensor
        params: Sampler parameters (steps, rescale_t, cfg_strength)
        cfg_interval: CFG interval (start, end)
        verbose: Whether to show progress

    Returns:
        Terminal noise tensor
    """
    from trellis.pipelines.samplers import SecondOrderRFSampler

    encoder = pipeline.models["sparse_structure_encoder"]
    flow_model = pipeline.models["sparse_structure_flow_model"]
    z_src = encoder(voxel_src)
    sampler = SecondOrderRFSampler()
    return sampler.sample(
        model=flow_model,
        sample=z_src,
        cond_dict=cond_src,
        steps=params["steps"],
        rescale_t=params["rescale_t"],
        cfg_strength=params["cfg_strength"],
        cfg_interval=cfg_interval,
        inverse=True,
        verbose=verbose,
    )


def denoise_sparse_structure(
    pipeline,
    cond_edit: dict,
    terminal_noise: torch.Tensor,
    params: dict,
    cfg_interval: Tuple[float, float],
    verbose: bool = False,
) -> torch.Tensor:
    """Denoise terminal noise to sparse structure.

    Args:
        pipeline: TRELLIS pipeline
        cond_edit: Edit condition dictionary
        terminal_noise: Terminal noise tensor
        params: Sampler parameters
        cfg_interval: CFG interval
        verbose: Whether to show progress

    Returns:
        Sparse structure coordinates
    """
    from trellis.pipelines.samplers import SecondOrderRFSampler

    flow_model = pipeline.models["sparse_structure_flow_model"]
    decoder = pipeline.models["sparse_structure_decoder"]
    sampler = SecondOrderRFSampler()
    z_tgt = sampler.sample(
        model=flow_model,
        sample=terminal_noise,
        cond_dict=cond_edit,
        steps=params["steps"],
        rescale_t=params["rescale_t"],
        cfg_strength=params["cfg_strength"],
        cfg_interval=cfg_interval,
        inverse=False,
        verbose=verbose,
    )
    voxel = decoder(z_tgt)
    coords = torch.argwhere(voxel > 0)[:, [0, 2, 3, 4]].int()
    if coords.shape[0] == 0:
        raise RuntimeError("Denoised sparse structure is empty")
    return coords


def invert_slat(
    pipeline,
    cond_src: dict,
    slat_src,
    params: dict,
    cfg_interval: Tuple[float, float],
    verbose: bool = False,
):
    """Invert SLAT to terminal noise.

    Args:
        pipeline: TRELLIS pipeline
        cond_src: Source condition dictionary
        slat_src: Source SLAT tensor
        params: Sampler parameters
        cfg_interval: CFG interval
        verbose: Whether to show progress

    Returns:
        Terminal noise tensor
    """
    from trellis.pipelines.samplers import SecondOrderRFSampler

    flow_model = pipeline.models["slat_flow_model"]
    mean, std = get_slat_norm_tensors(pipeline, slat_src.device, slat_src.feats.dtype)
    slat_normalized = (slat_src - mean) / std
    sampler = SecondOrderRFSampler()
    return sampler.sample(
        model=flow_model,
        sample=slat_normalized,
        cond_dict=cond_src,
        steps=params["steps"],
        rescale_t=params["rescale_t"],
        cfg_strength=params["cfg_strength"],
        cfg_interval=cfg_interval,
        inverse=True,
        verbose=verbose,
    )


def denoise_slat(
    pipeline,
    cond_edit: dict,
    terminal_noise,
    params: dict,
    cfg_interval: Tuple[float, float],
    verbose: bool = False,
):
    """Denoise terminal noise to SLAT.

    Args:
        pipeline: TRELLIS pipeline
        cond_edit: Edit condition dictionary
        terminal_noise: Terminal noise tensor
        params: Sampler parameters
        cfg_interval: CFG interval
        verbose: Whether to show progress

    Returns:
        Denoised SLAT tensor
    """
    from trellis.pipelines.samplers import SecondOrderRFSampler

    flow_model = pipeline.models["slat_flow_model"]
    sampler = SecondOrderRFSampler()
    slat_normalized = sampler.sample(
        model=flow_model,
        sample=terminal_noise,
        cond_dict=cond_edit,
        steps=params["steps"],
        rescale_t=params["rescale_t"],
        cfg_strength=params["cfg_strength"],
        cfg_interval=cfg_interval,
        inverse=False,
        verbose=verbose,
    )
    mean, std = get_slat_norm_tensors(pipeline, slat_normalized.device, slat_normalized.feats.dtype)
    return slat_normalized * std + mean
