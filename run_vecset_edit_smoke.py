#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from trellis_edit.vecset_benchmark import (
    DEFAULT_GT_ROOT,
    load_benchmark_metadata,
    select_first_prompt_case,
    select_prompt_cases,
)


DEFAULT_OUTPUT_ROOT = Path("/cache/wangxinxing/data/trellis_edit_benchmark/pred")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a single VecSet-Edit smoke case on Edit3D-Bench.",
    )
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--object-name", type=str, default="")
    parser.add_argument("--prompt-id", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--with-eval", action="store_true", help="Run render/eval instead of generation-only smoke.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _select_case(args: argparse.Namespace) -> tuple[str, str, int]:
    metadata = load_benchmark_metadata(args.gt_root)
    if args.dataset or args.object_name or int(args.prompt_id) > 0:
        cases = select_prompt_cases(
            metadata,
            gt_root=args.gt_root,
            limit=1,
            dataset=args.dataset or None,
            object_name=args.object_name or None,
            prompt_id=int(args.prompt_id) if int(args.prompt_id) > 0 else None,
            require_source_glb=True,
            require_edit_image=True,
            require_render_image=True,
            require_mask_image=True,
        )
        if not cases:
            raise RuntimeError("No matching smoke case found for the requested filters.")
        return cases[0]
    return select_first_prompt_case(
        metadata,
        gt_root=args.gt_root,
        require_render_image=True,
        require_mask_image=True,
    )


def main() -> None:
    args = build_parser().parse_args()
    args.gt_root = args.gt_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    dataset, object_name, prompt_id = _select_case(args)

    script_path = Path(__file__).resolve().parent / "run_vecset_edit_benchmark.py"
    smoke_output_root = args.output_root / "vecset_edit_local_smoke"
    command = [
        sys.executable,
        str(script_path),
        "--gt-root",
        str(args.gt_root),
        "--output-root",
        str(smoke_output_root),
        "--dataset",
        dataset,
        "--object-name",
        object_name,
        "--prompt-id",
        str(prompt_id),
        "--limit",
        "1",
        "--device",
        args.device,
        "--config-name",
        "vecset_edit_smoke_local",
    ]
    if not args.with_eval:
        command.append("--generate-only")
    if args.dry_run:
        command.append("--dry-run")

    print(f"[Smoke] {dataset}/{object_name}/prompt_{prompt_id}")
    print(f"[Smoke] Output root: {smoke_output_root}")
    print(f"[CMD] {' '.join(command)}")
    result = subprocess.run(command, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
