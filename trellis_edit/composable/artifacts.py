from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trellis_edit.preprocess.image_alignment import PreparedInputs


@dataclass
class PreprocessArtifact:
    prepared_inputs: PreparedInputs
    artifact_paths: dict[str, str] = field(default_factory=dict)


@dataclass
class SSArtifact:
    plugin_name: str
    coords: Any
    voxel_mesh: Any | None
    metadata: dict[str, Any]
    artifact_paths: dict[str, str] = field(default_factory=dict)
    primary_coords_path: Path | None = None


@dataclass
class SLATArtifact:
    plugin_name: str
    outputs: Any
    source_outputs: Any | None
    metadata: dict[str, Any]
    artifact_paths: dict[str, str] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    preprocess: PreprocessArtifact
    ss: SSArtifact | None = None
    slat: SLATArtifact | None = None
