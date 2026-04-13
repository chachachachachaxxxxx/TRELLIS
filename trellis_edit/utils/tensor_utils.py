from __future__ import annotations

import torch


def tensor_signature(tensor: torch.Tensor, rows: int = 3, cols: int = 8) -> torch.Tensor:
    """Extract a small signature from a tensor for comparison.

    Args:
        tensor: Input tensor
        rows: Number of rows to extract
        cols: Number of columns to extract

    Returns:
        Signature tensor
    """
    return tensor[:1, :rows, :cols].detach().float().cpu()


def signature_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute distance between two tensor signatures.

    Args:
        a: First signature
        b: Second signature

    Returns:
        Maximum absolute difference
    """
    return float(torch.max(torch.abs(a - b)).item())
