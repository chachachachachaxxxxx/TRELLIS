from __future__ import annotations

from pathlib import Path
from typing import Iterable


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
