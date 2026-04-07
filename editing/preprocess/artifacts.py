from __future__ import annotations

from pathlib import Path

from editing.common import ensure_dir, write_json
from .image_alignment import PreparedEditCondition, PreparedInputs


def save_preprocessed_inputs(
    out_dir: Path | str,
    prepared: PreparedInputs,
    *,
    include_mask: bool = True,
) -> dict[str, str]:
    out_dir = ensure_dir(Path(out_dir))
    artifacts = {
        "metadata": str(write_json(out_dir / "input_preprocess.json", prepared.meta)),
        "source": str(out_dir / "source_preprocessed.png"),
        "edit": str(out_dir / "edit_preprocessed.png"),
    }
    prepared.source.save(out_dir / "source_preprocessed.png")
    prepared.edit.save(out_dir / "edit_preprocessed.png")
    if include_mask:
        artifacts["mask"] = str(out_dir / "mask_preprocessed.png")
        prepared.mask.save(out_dir / "mask_preprocessed.png")
    return artifacts


def save_edit_condition_artifacts(
    out_dir: Path | str,
    prepared: PreparedEditCondition,
) -> dict[str, str]:
    out_dir = ensure_dir(Path(out_dir))
    artifacts = {
        "metadata": str(write_json(out_dir / "input_preprocess.json", prepared.meta)),
        "edit": str(out_dir / "edit_condition_input.png"),
    }
    prepared.edit.save(out_dir / "edit_condition_input.png")
    return artifacts
