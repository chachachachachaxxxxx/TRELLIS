"""Inversion utilities for TRELLIS editing."""
from __future__ import annotations

from .base_inverter import BaseInverter, InversionResult, InversionTrajectory
from .euler_inverter import EulerInverter
from .latent_replace_sampler import SparseLatentReplaceRFSolver
from .rf_inversion import (
    denoise_slat,
    denoise_sparse_structure,
    get_slat_norm_tensors,
    invert_slat,
    invert_sparse_structure,
)
from .rf_sampler import RFSolverSampler
from .rf_solver_inverter import RFSolverInverter
from .trajectory_metrics import compute_sparse_trajectory_similarity, compute_trajectory_similarity

__all__ = [
    # Legacy samplers
    "RFSolverSampler",
    "SparseLatentReplaceRFSolver",
    "denoise_slat",
    "denoise_sparse_structure",
    "get_slat_norm_tensors",
    "invert_slat",
    "invert_sparse_structure",
    # New inversion testing framework
    "BaseInverter",
    "InversionResult",
    "InversionTrajectory",
    "EulerInverter",
    "RFSolverInverter",
    "compute_trajectory_similarity",
    "compute_sparse_trajectory_similarity",
]
