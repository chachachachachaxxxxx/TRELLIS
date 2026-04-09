from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from editing.common.save_utils import read_json
from .path_resolver import display_path, first_existing, resolve_existing_path


SOURCE_IMAGE_CANDIDATES = (
    "2d_render.png",
    "render.png",
    "source_render.png",
    "source.png",
    "input.png",
)
EDIT_IMAGE_CANDIDATES = (
    "2d_edit.png",
    "edit.png",
    "target.png",
)
MASK_IMAGE_CANDIDATES = (
    "2d_mask.png",
    "mask.png",
)
MASK_GLB_CANDIDATES = (
    "mask.glb",
    "mask.gltf",
)
INPUT_MODEL_CANDIDATES = (
    "model.glb",
    "model.gltf",
    "sample.glb",
    "sample.gltf",
)


def _dict_like(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _string_or_empty(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _has_rf_assets(directory: Path | None) -> bool:
    if directory is None:
        return False
    return (directory / "voxels.ply").is_file() and (directory / "features.npz").is_file()


@dataclass(frozen=True)
class EditingCase:
    case_name: str
    base_dir: Path
    case_dir: Path | None
    manifest_path: Path | None
    asset_dir: Path | None
    render_dir: Path | None
    source_model: Path | None
    source_image: Path | None
    edit_image: Path | None
    mask_image: Path | None
    mask_glb: Path | None
    source_prompt: str | None
    edit_prompt: str | None
    seed: int | None
    notes: str
    defaults: dict[str, Any]
    raw_manifest: dict[str, Any]
    inferred_fields: tuple[str, ...]

    def method_defaults(self, method_name: str) -> dict[str, Any]:
        methods = _dict_like(self.raw_manifest.get("methods"))
        return _dict_like(methods.get(method_name))

    def snapshot(self) -> dict[str, Any]:
        return {
            "case_name": self.case_name,
            "base_dir": str(self.base_dir),
            "case_dir": str(self.case_dir) if self.case_dir is not None else None,
            "manifest_path": str(self.manifest_path) if self.manifest_path is not None else None,
            "asset_dir": str(self.asset_dir) if self.asset_dir is not None else None,
            "render_dir": str(self.render_dir) if self.render_dir is not None else None,
            "source_model": str(self.source_model) if self.source_model is not None else None,
            "source_image": str(self.source_image) if self.source_image is not None else None,
            "edit_image": str(self.edit_image) if self.edit_image is not None else None,
            "mask_image": str(self.mask_image) if self.mask_image is not None else None,
            "mask_glb": str(self.mask_glb) if self.mask_glb is not None else None,
            "source_prompt": self.source_prompt,
            "edit_prompt": self.edit_prompt,
            "seed": self.seed,
            "notes": self.notes,
            "defaults": self.defaults,
            "inferred_fields": list(self.inferred_fields),
        }


def _load_case_manifest(case_ref: str | Path | None) -> tuple[Path, Path | None, Path | None, dict[str, Any], bool]:
    if case_ref is None or not str(case_ref).strip():
        cwd = Path.cwd().resolve()
        return cwd, None, None, {}, False

    resolved = Path(case_ref).expanduser().resolve()

    if resolved.is_file():
        manifest_path = resolved
        case_dir = resolved.parent
        raw_manifest = read_json(manifest_path)
        return case_dir, case_dir, manifest_path, raw_manifest, True

    manifest_path = resolved / "manifest.json"
    raw_manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    return resolved, resolved, (manifest_path if manifest_path.is_file() else None), raw_manifest, True


def _candidate_paths(directory: Path | None, names: tuple[str, ...]) -> list[Path | None]:
    if directory is None:
        return []
    return [directory / name for name in names]


def load_case(case_ref: str | Path | None = None) -> EditingCase:
    base_dir, case_dir, manifest_path, raw_manifest, allow_standard_layout = _load_case_manifest(case_ref)

    source_section = _dict_like(raw_manifest.get("source"))
    edit_section = _dict_like(raw_manifest.get("edit"))
    defaults = _dict_like(raw_manifest.get("defaults"))

    inferred_fields: list[str] = []

    source_dir = resolve_existing_path(
        base_dir,
        source_section.get("dir") or raw_manifest.get("source_dir"),
        label="source.dir",
        kind="dir",
    )
    if source_dir is None and allow_standard_layout:
        source_dir = first_existing([base_dir / "source"], kind="dir")

    edit_dir = resolve_existing_path(
        base_dir,
        edit_section.get("dir") or raw_manifest.get("edit_dir"),
        label="edit.dir",
        kind="dir",
    )
    if edit_dir is None and allow_standard_layout:
        edit_dir = first_existing([base_dir / "edit"], kind="dir")

    asset_dir = resolve_existing_path(
        base_dir,
        source_section.get("asset_dir")
        or raw_manifest.get("asset_dir"),
        label="asset_dir",
    )
    if asset_dir is None and _has_rf_assets(source_dir):
        asset_dir = source_dir
        inferred_fields.append("asset_dir")

    render_dir = resolve_existing_path(
        base_dir,
        source_section.get("render_dir") or raw_manifest.get("render_dir"),
        label="render_dir",
        kind="dir",
    )
    if render_dir is None and _has_rf_assets(source_dir):
        render_dir = source_dir
        inferred_fields.append("render_dir")

    source_model = resolve_existing_path(
        base_dir,
        source_section.get("source_model") or raw_manifest.get("source_model"),
        label="source_model",
        kind="file",
    )
    if source_model is None:
        source_model = first_existing(_candidate_paths(source_dir, INPUT_MODEL_CANDIDATES), kind="file")
        if source_model is not None:
            inferred_fields.append("source_model")

    source_image = resolve_existing_path(
        base_dir,
        source_section.get("image") or raw_manifest.get("source_image"),
        label="source_image",
        kind="file",
    )
    if source_image is None:
        source_image = first_existing(
            _candidate_paths(source_dir, SOURCE_IMAGE_CANDIDATES)
            + _candidate_paths(case_dir, SOURCE_IMAGE_CANDIDATES),
            kind="file",
        )
        if source_image is not None:
            inferred_fields.append("source_image")

    edit_image = resolve_existing_path(
        base_dir,
        edit_section.get("image") or raw_manifest.get("edit_image"),
        label="edit_image",
        kind="file",
    )
    if edit_image is None:
        edit_image = first_existing(
            _candidate_paths(edit_dir, EDIT_IMAGE_CANDIDATES)
            + _candidate_paths(case_dir, EDIT_IMAGE_CANDIDATES),
            kind="file",
        )
        if edit_image is not None:
            inferred_fields.append("edit_image")

    mask_image = resolve_existing_path(
        base_dir,
        edit_section.get("mask_image") or raw_manifest.get("mask_image"),
        label="mask_image",
        kind="file",
    )
    if mask_image is None:
        mask_image = first_existing(
            _candidate_paths(edit_dir, MASK_IMAGE_CANDIDATES)
            + _candidate_paths(case_dir, MASK_IMAGE_CANDIDATES),
            kind="file",
        )
        if mask_image is not None:
            inferred_fields.append("mask_image")

    mask_glb = resolve_existing_path(
        base_dir,
        edit_section.get("mask_glb") or raw_manifest.get("mask_glb"),
        label="mask_glb",
        kind="file",
    )
    if mask_glb is None:
        mask_glb = first_existing(
            _candidate_paths(edit_dir, MASK_GLB_CANDIDATES)
            + _candidate_paths(case_dir, MASK_GLB_CANDIDATES),
            kind="file",
        )
        if mask_glb is not None:
            inferred_fields.append("mask_glb")

    # Load text prompts
    source_prompt = _string_or_empty(source_section.get("prompt") or raw_manifest.get("source_prompt")).strip() or None
    edit_prompt = _string_or_empty(edit_section.get("prompt") or raw_manifest.get("edit_prompt")).strip() or None

    case_name = _string_or_empty(raw_manifest.get("case_name") or raw_manifest.get("name")).strip()
    if not case_name:
        case_name = base_dir.name if case_dir is not None else "adhoc_case"

    return EditingCase(
        case_name=case_name,
        base_dir=base_dir,
        case_dir=case_dir,
        manifest_path=manifest_path,
        asset_dir=asset_dir,
        render_dir=render_dir,
        source_model=source_model,
        source_image=source_image,
        edit_image=edit_image,
        mask_image=mask_image,
        mask_glb=mask_glb,
        source_prompt=source_prompt,
        edit_prompt=edit_prompt,
        seed=_int_or_none(raw_manifest.get("seed") or defaults.get("seed")),
        notes=_string_or_empty(raw_manifest.get("notes")).strip(),
        defaults=defaults,
        raw_manifest=raw_manifest,
        inferred_fields=tuple(inferred_fields),
    )


def apply_case_overrides(
    case: EditingCase,
    *,
    case_name: str = "",
    seed: int | None = None,
    asset_dir: str = "",
    render_dir: str = "",
    source_model: str = "",
    source_image: str = "",
    edit_image: str = "",
    mask_image: str = "",
    mask_glb: str = "",
    source_prompt: str = "",
    edit_prompt: str = "",
) -> EditingCase:
    cwd = Path.cwd().resolve()
    updates: dict[str, Any] = {}

    if case_name.strip():
        updates["case_name"] = case_name.strip()
    if seed is not None:
        updates["seed"] = int(seed)
    if source_prompt.strip():
        updates["source_prompt"] = source_prompt.strip()
    if edit_prompt.strip():
        updates["edit_prompt"] = edit_prompt.strip()

    path_overrides = {
        "asset_dir": (asset_dir, "asset_dir", "any"),
        "render_dir": (render_dir, "render_dir", "dir"),
        "source_model": (source_model, "source_model", "file"),
        "source_image": (source_image, "source_image", "file"),
        "edit_image": (edit_image, "edit_image", "file"),
        "mask_image": (mask_image, "mask_image", "file"),
        "mask_glb": (mask_glb, "mask_glb", "file"),
    }
    for field_name, (value, label, kind) in path_overrides.items():
        if not str(value).strip():
            continue
        updates[field_name] = resolve_existing_path(cwd, value, label=label, kind=kind)

    if not updates:
        return case

    inferred_fields = tuple(field for field in case.inferred_fields if field not in updates)
    updated = replace(case, inferred_fields=inferred_fields, **updates)
    return updated


def summarize_case(case: EditingCase) -> dict[str, Any]:
    base = case.case_dir or case.base_dir
    return {
        "case_name": case.case_name,
        "case_dir": display_path(case.case_dir, base),
        "manifest_path": display_path(case.manifest_path, base),
        "asset_dir": display_path(case.asset_dir, base),
        "render_dir": display_path(case.render_dir, base),
        "source_model": display_path(case.source_model, base),
        "source_image": display_path(case.source_image, base),
        "edit_image": display_path(case.edit_image, base),
        "mask_image": display_path(case.mask_image, base),
        "mask_glb": display_path(case.mask_glb, base),
        "seed": case.seed,
        "inferred_fields": list(case.inferred_fields),
    }
