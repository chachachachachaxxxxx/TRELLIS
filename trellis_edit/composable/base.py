from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from trellis_edit.common import ExperimentOutputLayout

from .artifacts import PreprocessArtifact, SLATArtifact, SSArtifact
from .config import (
    ExperimentConfig,
    P2PLatentBlendSLATConfig,
    P2PLatentBlendSSConfig,
    PreprocessConfig,
    UniEditSLATConfig,
    UniEditSSConfig,
)


@dataclass(frozen=True)
class LoadedInputs:
    source_image: Image.Image
    edit_image: Image.Image
    mask_image: Image.Image | None


@dataclass(frozen=True)
class ExperimentContext:
    config: ExperimentConfig
    pipeline: Any
    layout: ExperimentOutputLayout
    loaded_inputs: LoadedInputs

    @property
    def preprocess_dir(self) -> Path:
        return self.layout.edit_dir / "preprocess"

    @property
    def ss_dir(self) -> Path:
        return self.layout.edit_dir / "ss"

    @property
    def slat_dir(self) -> Path:
        return self.layout.edit_dir / "slat"


class PreprocessPlugin(ABC):
    name: str

    @abstractmethod
    def run(
        self,
        context: ExperimentContext,
        config: PreprocessConfig,
    ) -> PreprocessArtifact:
        raise NotImplementedError


class SSStagePlugin(ABC):
    name: str

    @abstractmethod
    def run(
        self,
        context: ExperimentContext,
        preprocess: PreprocessArtifact,
        config: UniEditSSConfig | P2PLatentBlendSSConfig,
    ) -> SSArtifact:
        raise NotImplementedError

    @abstractmethod
    def save(
        self,
        artifact: SSArtifact,
        out_dir: Path,
        config: UniEditSSConfig | P2PLatentBlendSSConfig,
    ) -> dict[str, str]:
        raise NotImplementedError

    def cleanup(self) -> None:
        pass


class SLATStagePlugin(ABC):
    name: str

    @abstractmethod
    def run(
        self,
        context: ExperimentContext,
        preprocess: PreprocessArtifact,
        config: UniEditSLATConfig | P2PLatentBlendSLATConfig,
        ss_artifact: SSArtifact | None,
    ) -> SLATArtifact:
        raise NotImplementedError

    @abstractmethod
    def save(
        self,
        artifact: SLATArtifact,
        out_dir: Path,
        config: UniEditSLATConfig | P2PLatentBlendSLATConfig,
    ) -> dict[str, str]:
        raise NotImplementedError

    def cleanup(self) -> None:
        pass
