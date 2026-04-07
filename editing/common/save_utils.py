from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def ensure_dir(path: Path | str) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def write_json(path: Path | str, payload: Any) -> Path:
    resolved = Path(path)
    ensure_dir(resolved.parent)
    resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return resolved


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
