from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_OUTPUTS_ROOT = Path("outputs")


@dataclass(frozen=True)
class ExperimentOutputLayout:
    method_name: str
    case_name: str
    root_dir: Path
    method_dir: Path
    case_dir: Path
    edit_dir: Path
    source_original_dir: Path
    artifacts_dir: Path
    logs_dir: Path


def build_experiment_output_layout(
    method_name: str,
    case_name: str,
    outputs_root: Path | str = DEFAULT_OUTPUTS_ROOT,
) -> ExperimentOutputLayout:
    root_dir = Path(outputs_root)
    method_dir = root_dir / method_name
    case_dir = method_dir / case_name
    return ExperimentOutputLayout(
        method_name=method_name,
        case_name=case_name,
        root_dir=root_dir,
        method_dir=method_dir,
        case_dir=case_dir,
        edit_dir=case_dir / "edit",
        source_original_dir=case_dir / "source_original",
        artifacts_dir=case_dir / "artifacts",
        logs_dir=case_dir / "artifacts" / "logs",
    )
