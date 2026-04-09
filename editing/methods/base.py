from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image


@dataclass
class EditMethodConfig:
    """Configuration for an edit method."""
    method_name: str
    seed: int
    num_samples: int = 1
    sparse_structure_sampler_params: Optional[Dict[str, Any]] = None
    slat_sampler_params: Optional[Dict[str, Any]] = None
    extra_params: Optional[Dict[str, Any]] = None


@dataclass
class EditMethodInputs:
    """Inputs for an edit method."""
    source_image: Optional[Image.Image] = None
    edit_image: Optional[Image.Image] = None
    mask_image: Optional[Image.Image] = None
    mask_glb_path: Optional[Path] = None
    source_model_path: Optional[Path] = None
    asset_dir: Optional[Path] = None
    source_voxels_path: Optional[Path] = None
    source_features_path: Optional[Path] = None
    extra_inputs: Optional[Dict[str, Any]] = None


@dataclass
class EditMethodOutputs:
    """Outputs from an edit method."""
    outputs: Any  # TRELLIS pipeline outputs
    source_outputs: Optional[Any] = None
    intermediate_results: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class EditMethod(ABC):
    """Base class for edit methods.

    An edit method encapsulates:
    1. Preprocessing logic specific to the method
    2. Pipeline execution with method-specific hooks/patches
    3. Artifact saving logic

    Subclasses should implement:
    - prepare(): Set up method-specific state (hooks, patches, etc.)
    - run(): Execute the editing pipeline
    - save_artifacts(): Save method-specific outputs
    """

    def __init__(self, method_name: str):
        self.method_name = method_name

    @abstractmethod
    def prepare(
        self,
        pipeline,
        inputs: EditMethodInputs,
        config: EditMethodConfig,
    ) -> Dict[str, Any]:
        """Prepare method-specific state before running.

        This is where you should:
        - Set up attention hooks/patches
        - Prepare inversion state
        - Build method-specific conditions

        Args:
            pipeline: TRELLIS pipeline instance
            inputs: Preprocessed inputs
            config: Method configuration

        Returns:
            Dictionary of prepared state (conditions, hooks, etc.)
        """
        pass

    @abstractmethod
    def run(
        self,
        pipeline,
        prepared_state: Dict[str, Any],
        config: EditMethodConfig,
    ) -> EditMethodOutputs:
        """Execute the editing pipeline.

        Args:
            pipeline: TRELLIS pipeline instance
            prepared_state: State from prepare()
            config: Method configuration

        Returns:
            EditMethodOutputs with results
        """
        pass

    @abstractmethod
    def save_artifacts(
        self,
        outputs: EditMethodOutputs,
        out_dir: Path,
        config: EditMethodConfig,
    ) -> Dict[str, str]:
        """Save method-specific artifacts.

        Args:
            outputs: Results from run()
            out_dir: Output directory
            config: Method configuration

        Returns:
            Dictionary mapping artifact names to file paths
        """
        pass

    def cleanup(self):
        """Clean up method state (restore patches, etc.).

        Called after run() completes or fails.
        """
        pass

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for this method.

        Returns:
            Dictionary of default parameters
        """
        return {}


class SimpleEditMethod(EditMethod):
    """Simple edit method that doesn't need hooks/patches.

    For methods that just run the pipeline with different conditions.
    """

    def prepare(
        self,
        pipeline,
        inputs: EditMethodInputs,
        config: EditMethodConfig,
    ) -> Dict[str, Any]:
        """Default prepare: just return empty state."""
        return {}

    def cleanup(self):
        """Default cleanup: nothing to do."""
        pass
