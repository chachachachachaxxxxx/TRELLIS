"""Trajectory similarity metrics for inversion evaluation."""
from __future__ import annotations

from typing import Dict, List

import torch


def compute_trajectory_similarity(
    inverse_trajectory: List[torch.Tensor],
    forward_trajectory: List[torch.Tensor],
    timesteps: List[float],
) -> Dict[str, float]:
    """Compute similarity metrics between inverse and forward trajectories.

    Args:
        inverse_trajectory: Latents from data → noise
        forward_trajectory: Latents from noise → data
        timesteps: Timestep values

    Returns:
        Dictionary of similarity metrics
    """
    if len(inverse_trajectory) != len(forward_trajectory):
        raise ValueError("Trajectories must have same length")

    metrics = {}

    # L2 distance at each timestep
    l2_distances = []
    for inv_latent, fwd_latent in zip(inverse_trajectory, forward_trajectory):
        if hasattr(inv_latent, 'feats'):
            # Sparse tensor
            # For simplicity, compare feature norms
            inv_norm = inv_latent.feats.norm(dim=1).mean().item()
            fwd_norm = fwd_latent.feats.norm(dim=1).mean().item()
            l2_distances.append(abs(inv_norm - fwd_norm))
        else:
            # Dense tensor
            dist = (inv_latent - fwd_latent).norm().item()
            l2_distances.append(dist)

    metrics["l2_mean"] = sum(l2_distances) / len(l2_distances)
    metrics["l2_max"] = max(l2_distances)
    metrics["l2_final"] = l2_distances[-1]  # At t=0 (data)

    # Cosine similarity at final step
    inv_final = inverse_trajectory[-1]
    fwd_final = forward_trajectory[-1]

    if hasattr(inv_final, 'feats'):
        # Sparse tensor - compare feature vectors
        inv_flat = inv_final.feats.flatten()
        fwd_flat = fwd_final.feats.flatten()
    else:
        inv_flat = inv_final.flatten()
        fwd_flat = fwd_final.flatten()

    cos_sim = torch.nn.functional.cosine_similarity(
        inv_flat.unsqueeze(0),
        fwd_flat.unsqueeze(0),
    ).item()
    metrics["cosine_similarity_final"] = cos_sim

    return metrics


def compute_sparse_trajectory_similarity(
    inverse_trajectory: List,  # List of SparseTensor
    forward_trajectory: List,  # List of SparseTensor
    timesteps: List[float],
) -> Dict[str, float]:
    """Compute similarity metrics for sparse tensor trajectories.

    Args:
        inverse_trajectory: Sparse latents from data → noise
        forward_trajectory: Sparse latents from noise → data
        timesteps: Timestep values

    Returns:
        Dictionary of similarity metrics
    """
    metrics = {}

    # Feature-level metrics
    feat_l2_distances = []
    coord_overlap_ratios = []

    for inv_st, fwd_st in zip(inverse_trajectory, forward_trajectory):
        # Feature L2 distance
        inv_feats = inv_st.feats
        fwd_feats = fwd_st.feats

        # Compare feature norms
        inv_norm = inv_feats.norm(dim=1).mean().item()
        fwd_norm = fwd_feats.norm(dim=1).mean().item()
        feat_l2_distances.append(abs(inv_norm - fwd_norm))

        # Coordinate overlap
        inv_coords_set = set(map(tuple, inv_st.coords.cpu().numpy()))
        fwd_coords_set = set(map(tuple, fwd_st.coords.cpu().numpy()))
        overlap = len(inv_coords_set & fwd_coords_set)
        total = len(inv_coords_set | fwd_coords_set)
        coord_overlap_ratios.append(overlap / total if total > 0 else 0.0)

    metrics["feat_l2_mean"] = sum(feat_l2_distances) / len(feat_l2_distances)
    metrics["feat_l2_max"] = max(feat_l2_distances)
    metrics["feat_l2_final"] = feat_l2_distances[-1]

    metrics["coord_overlap_mean"] = sum(coord_overlap_ratios) / len(coord_overlap_ratios)
    metrics["coord_overlap_final"] = coord_overlap_ratios[-1]

    return metrics
