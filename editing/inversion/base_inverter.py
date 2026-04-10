"""Base class for inversion methods."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class InversionTrajectory:
    """Stores a complete inversion trajectory.

    Attributes:
        latents: List of latent states at each timestep
        timesteps: List of timestep values
        metadata: Additional metadata (e.g., method name, params)
    """
    latents: List[torch.Tensor]
    timesteps: List[float]
    metadata: Dict


@dataclass
class InversionResult:
    """Result of an inversion operation.

    Attributes:
        terminal_noise: Final noise at t=1
        forward_trajectory: Trajectory from noise back to data
        inverse_trajectory: Trajectory from data to noise
        similarity_metrics: Metrics comparing trajectories
    """
    terminal_noise: torch.Tensor
    forward_trajectory: InversionTrajectory
    inverse_trajectory: InversionTrajectory
    similarity_metrics: Optional[Dict] = None


class BaseInverter(ABC):
    """Abstract base class for inversion methods."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
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
        """Invert sample to terminal noise and back.

        Args:
            model: Flow model
            sample: Initial sample (data)
            cond_dict: Condition dictionary
            steps: Number of steps
            rescale_t: Time rescaling
            cfg_strength: CFG strength
            cfg_interval: CFG interval
            verbose: Show progress

        Returns:
            InversionResult with trajectories and metrics
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict:
        """Get inverter configuration."""
        pass
