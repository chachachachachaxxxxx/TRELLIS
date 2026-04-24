#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trellis_edit.benchmark_pages import DEFAULT_BENCHMARK_ROOT, build_focus_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove a run from benchmark focus.")
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=DEFAULT_BENCHMARK_ROOT,
        help="Benchmark root directory.",
    )
    parser.add_argument("--focus-id", required=True, help="Focus entry id to remove.")
    return parser.parse_args()


def load_manifest(path: Path) -> dict:
    if not path.is_file():
        return {"runs": []}
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    benchmark_root = args.benchmark_root.resolve()
    focus_root = benchmark_root / "focus"
    manifest_path = focus_root / "collection_manifest.json"
    payload = load_manifest(manifest_path)

    payload["runs"] = [item for item in payload.get("runs", []) if item.get("id") != args.focus_id]
    if payload.get("baseline_run_id") == args.focus_id:
        payload["baseline_run_id"] = payload["runs"][0]["id"] if payload["runs"] else None
    write_manifest(manifest_path, payload)

    link_path = focus_root / args.focus_id
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink() or link_path.is_file():
            link_path.unlink()
        elif link_path.is_dir():
            raise SystemExit(f"Refuse to remove non-symlink directory: {link_path}")

    output_path = build_focus_index(benchmark_root)
    print(f"removed focus id: {args.focus_id}")
    print(f"focus manifest: {manifest_path}")
    if output_path is not None:
        print(f"focus index: {output_path}")


if __name__ == "__main__":
    main()
