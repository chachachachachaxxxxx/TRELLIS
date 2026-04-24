#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trellis_edit.benchmark_pages import DEFAULT_BENCHMARK_ROOT, build_focus_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add or update a run in benchmark focus.")
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=DEFAULT_BENCHMARK_ROOT,
        help="Benchmark root directory.",
    )
    parser.add_argument(
        "--daily-run",
        required=True,
        help="Daily run id under benchmark/daily, or an absolute path to a daily bundle.",
    )
    parser.add_argument("--focus-id", required=True, help="Focus entry id and link name.")
    parser.add_argument("--label", help="Display label. Defaults to focus id.")
    parser.add_argument("--group", help="Display group. Defaults to the daily run manifest group.")
    parser.add_argument(
        "--as-baseline",
        action="store_true",
        help="Also set this focus id as baseline_run_id.",
    )
    return parser.parse_args()


def resolve_daily_run(benchmark_root: Path, daily_run: str) -> Path:
    candidate = Path(daily_run)
    if candidate.is_absolute():
        return candidate.resolve()
    return (benchmark_root / "daily" / daily_run).resolve()


def load_manifest(path: Path) -> dict:
    if not path.is_file():
        return {"runs": []}
    return json.loads(path.read_text(encoding="utf-8"))


def infer_focus_group(
    explicit_group: str | None,
    daily_manifest: dict,
    daily_summary: dict,
    focus_id: str,
) -> str:
    if explicit_group:
        return explicit_group

    manifest_group = daily_manifest.get("group")
    if manifest_group:
        inferred_group = str(manifest_group)
    else:
        config_name = str(daily_manifest.get("config_name") or focus_id)
        inferred_group = "baseline" if config_name.startswith("baseline") else "compare"

    total_cases = ((daily_summary or {}).get("totals") or {}).get("cases")
    if inferred_group == "baseline" and isinstance(total_cases, int) and total_cases <= 12:
        return "compare"
    return inferred_group


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    benchmark_root = args.benchmark_root.resolve()
    focus_root = benchmark_root / "focus"
    focus_root.mkdir(parents=True, exist_ok=True)

    daily_root = resolve_daily_run(benchmark_root, args.daily_run)
    if not daily_root.is_dir():
        raise SystemExit(f"Daily run not found: {daily_root}")
    daily_manifest = load_manifest(daily_root / "manifest.json")
    daily_summary = load_manifest(daily_root / "summary.json")
    run_group = infer_focus_group(args.group, daily_manifest, daily_summary, args.focus_id)

    link_path = focus_root / args.focus_id
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_dir() and not link_path.is_symlink():
            raise SystemExit(f"Refuse to replace non-symlink directory: {link_path}")
        link_path.unlink()
    target = os.path.relpath(daily_root, start=focus_root)
    os.symlink(target, link_path, target_is_directory=True)

    manifest_path = focus_root / "collection_manifest.json"
    payload = load_manifest(manifest_path)
    runs = [item for item in payload.get("runs", []) if item.get("id") != args.focus_id]
    runs.append(
        {
            "id": args.focus_id,
            "path": args.focus_id,
            "label": args.label or args.focus_id,
            "group": run_group,
        }
    )
    payload["runs"] = runs
    if args.as_baseline:
        payload["baseline_run_id"] = args.focus_id
    write_manifest(manifest_path, payload)

    output_path = build_focus_index(benchmark_root)
    print(f"focus link: {link_path} -> {target}")
    print(f"focus manifest: {manifest_path}")
    if output_path is not None:
        print(f"focus index: {output_path}")


if __name__ == "__main__":
    main()
