"""RF-Solver inversion utilities for TRELLIS editing."""
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
    norm = pipeline.slat_normalization

    # Handle different formats
    if isinstance(norm, dict):
        # Dict format: {"mean": tensor/list, "std": tensor/list}
        mean_val = norm["mean"]
        std_val = norm["std"]

        # Convert to tensors if needed
        if isinstance(mean_val, (list, tuple)):
            mean = torch.tensor(mean_val, device=device, dtype=dtype)
        else:
            mean = mean_val.to(device=device, dtype=dtype)

        if isinstance(std_val, (list, tuple)):
            std = torch.tensor(std_val, device=device, dtype=dtype)
        else:
            std = std_val.to(device=device, dtype=dtype)

    elif isinstance(norm, (list, tuple)):
        # List format: [mean, std]
        mean = torch.tensor(norm[0], device=device, dtype=dtype)
        std = torch.tensor(norm[1], device=device, dtype=dtype)
    else:
        raise TypeError(f"Unexpected slat_normalization type: {type(norm)}")

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
    from editing.inversion.rf_sampler import RFSolverSampler

    encoder = pipeline.models["sparse_structure_encoder"]
    flow_model = pipeline.models["sparse_structure_flow_model"]
    z_src = encoder(voxel_src)
    sampler = RFSolverSampler()
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
    from editing.inversion.rf_sampler import RFSolverSampler

    flow_model = pipeline.models["sparse_structure_flow_model"]
    decoder = pipeline.models["sparse_structure_decoder"]
    sampler = RFSolverSampler()
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
    from editing.inversion.rf_sampler import RFSolverSampler

    flow_model = pipeline.models["slat_flow_model"]
    mean, std = get_slat_norm_tensors(pipeline, slat_src.device, slat_src.feats.dtype)
    slat_normalized = (slat_src - mean) / std
    sampler = RFSolverSampler()
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
    from editing.inversion.rf_sampler import RFSolverSampler

    flow_model = pipeline.models["slat_flow_model"]
    sampler = RFSolverSampler()
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
