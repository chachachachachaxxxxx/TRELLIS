from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple


SOURCE_RENDER_CANDIDATES = (
    "2d_render.png",
    "render.png",
    "source_render.png",
    "source.png",
    "input.png",
)


def resolve_path(base_dir: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def resolve_existing_path(
    base_dir: Path,
    value: str | Path | None,
    *,
    label: str,
    kind: str = "any",
) -> Path | None:
    path = resolve_path(base_dir, value)
    if path is None:
        return None
    if not path.exists():
        raise RuntimeError(f"{label} does not exist: {path}")
    if kind == "file" and not path.is_file():
        raise RuntimeError(f"{label} must be a file: {path}")
    if kind == "dir" and not path.is_dir():
        raise RuntimeError(f"{label} must be a directory: {path}")
    return path


def first_existing(paths: Iterable[Path | None], *, kind: str = "any") -> Path | None:
    for path in paths:
        if path is None:
            continue
        if not path.exists():
            continue
        if kind == "file" and not path.is_file():
            continue
        if kind == "dir" and not path.is_dir():
            continue
        return path.resolve()
    return None


def display_path(path: Path | None, relative_to: Path | None = None) -> str | None:
    if path is None:
        return None
    if relative_to is not None:
        try:
            return str(path.relative_to(relative_to))
        except ValueError:
            pass
    return str(path)


def ensure_path_exists(path: Path, label: str) -> Path:
    """Validate that a path exists.

    Args:
        path: Path to validate
        label: Human-readable label for error messages

    Returns:
        The validated path

    Raises:
        RuntimeError: If path does not exist
    """
    if not path.exists():
        raise RuntimeError(f"{label} does not exist: {path}")
    return path


def resolve_asset_dir(source_model: str) -> Path:
    """Resolve asset directory from source model path.

    If source_model is a directory, returns it.
    If source_model is a file, returns its parent directory.

    Args:
        source_model: Path to source 3D model (file or directory)

    Returns:
        Asset directory path
    """
    model_path = ensure_path_exists(Path(source_model).expanduser().resolve(), "source-model")
    if model_path.is_dir():
        return model_path
    return model_path.parent


def resolve_source_image_path(asset_dir: Path, explicit_path: str) -> Path:
    """Resolve source render image path.

    If explicit_path is provided, validates and returns it.
    Otherwise, searches for common source render filenames in asset_dir.

    Args:
        asset_dir: Directory to search in
        explicit_path: Optional explicit path to source image

    Returns:
        Path to source render image

    Raises:
        RuntimeError: If no source render image found
    """
    if explicit_path:
        return ensure_path_exists(Path(explicit_path).expanduser().resolve(), "source-image")

    # Try direct children first
    for name in SOURCE_RENDER_CANDIDATES:
        candidate = asset_dir / name
        if candidate.is_file():
            return candidate

    # Try recursive search
    for name in SOURCE_RENDER_CANDIDATES:
        matches = sorted(asset_dir.rglob(name))
        if matches:
            return matches[0]

    raise RuntimeError(
        "Could not find a source render image automatically. "
        f"Expected one of {list(SOURCE_RENDER_CANDIDATES)} inside {asset_dir}. "
        "Please pass --source-image explicitly."
    )


def validate_required_asset_files(asset_dir: Path) -> Tuple[Path, Path]:
    """Validate that required RF inversion asset files exist.

    Args:
        asset_dir: Directory containing assets

    Returns:
        Tuple of (voxels_path, features_path)

    Raises:
        RuntimeError: If required files are missing
    """
    voxels_path = asset_dir / "voxels.ply"
    features_path = asset_dir / "features.npz"
    missing = [str(path.name) for path in (voxels_path, features_path) if not path.is_file()]
    if missing:
        raise RuntimeError(
            "source-model must point to an asset directory (or a file inside it) containing "
            f"{missing}. A plain TRELLIS export like sample.ply/sample.glb is not enough for RF inversion."
        )
    return voxels_path, features_path


def candidate_file(path: Path) -> Optional[Path]:
    """Check if path is a file, return it or None.

    Args:
        path: Path to check

    Returns:
        Path if it's a file, None otherwise
    """
    return path if path.is_file() else None


def resolve_image_dir(image_dir: str) -> Path:
    """Resolve and validate image directory path.

    Args:
        image_dir: Path to image directory

    Returns:
        Validated image directory path
    """
    return ensure_path_exists(Path(image_dir).expanduser().resolve(), "image_dir")
