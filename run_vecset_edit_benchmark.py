#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from run_batch_edit_and_eval import (
    render_all_results,
    run_evaluation,
    save_results,
)
from trellis_edit.common import ensure_dir, utc_now_iso, write_json
from trellis_edit.vecset_benchmark import (
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_GT_ROOT,
    DEFAULT_METRICS,
    ensure_vecset_edit_root,
    load_benchmark_metadata,
    parse_render_gpu_ids,
    select_prompt_cases,
)


DEFAULT_OUTPUT_ROOT = Path("/cache/wangxinxing/data/trellis_edit_benchmark/pred/vecset_edit_local")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the local VecSet-Edit benchmark generation, then optionally render/evaluate/package the outputs."
        ),
    )
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--render-gpus", type=str, default="", help="Comma-separated GPU ids for benchmark rendering.")
    parser.add_argument("--eval-device", type=str, default="", help="Evaluation device, defaults to --device.")
    parser.add_argument("--limit", type=int, default=300, help="Number of prompt-level cases to include.")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--object-name", type=str, default="")
    parser.add_argument("--prompt-id", type=int, default=0)
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--config-name", type=str, default="vecset_edit_local")
    parser.add_argument("--run-group", type=str, default="editing_methods")
    parser.add_argument("--resume", action="store_true", help="Skip prompt outputs that already exist.")
    parser.add_argument(
        "--case-shard-count",
        type=int,
        default=1,
        help="Split prompt-level generation into this many disjoint shards.",
    )
    parser.add_argument(
        "--case-shard-index",
        type=int,
        default=0,
        help="Zero-based case shard index to run.",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate prompt-level edit.glb outputs. Skip render/eval/daily packaging.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the VecSet command and selected cases without executing.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--azimuth", type=float, default=0.0)
    parser.add_argument("--elevation", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--attentive-2d", type=int, default=8)
    parser.add_argument("--cut-off-p", type=float, default=0.5)
    parser.add_argument("--topk-percent-2d", type=float, default=0.2)
    parser.add_argument("--threshold-percent-2d", type=float, default=0.1)
    parser.add_argument("--step-pruning", type=int, default=5)
    parser.add_argument("--edit-strength", type=float, default=0.9)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument(
        "--texture-render-method",
        type=str,
        default="nvdiffrast",
        choices=("nvdiffrast", "bpy"),
    )
    parser.add_argument("--texture-diff-threshold", type=float, default=0.005)
    return parser


def _select_case_shard(
    cases: list[tuple[str, str, int]],
    *,
    shard_count: int,
    shard_index: int,
) -> list[tuple[str, str, int]]:
    if shard_count <= 1:
        return cases
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"Invalid shard index {shard_index} for shard count {shard_count}.")
    return [case for idx, case in enumerate(cases) if idx % shard_count == shard_index]


def _build_vecset_command(args: argparse.Namespace) -> list[str]:
    repo_root = ensure_vecset_edit_root()
    command = [
        sys.executable,
        str((repo_root / "run_edit3d_benchmark.py").resolve()),
        "--gt-root",
        str(args.gt_root),
        "--pred-base",
        str(args.output_root.parent),
        "--run-name",
        args.output_root.name,
        "--max-cases",
        str(int(args.limit)),
        "--case-shard-count",
        str(int(args.case_shard_count)),
        "--case-shard-index",
        str(int(args.case_shard_index)),
        "--device",
        args.device,
        "--skip-render",
        "--skip-eval",
        "--seed",
        str(int(args.seed)),
        "--num-inference-steps",
        str(int(args.num_inference_steps)),
        "--azimuth",
        str(float(args.azimuth)),
        "--elevation",
        str(float(args.elevation)),
        "--scale",
        str(float(args.scale)),
        "--attentive-2d",
        str(int(args.attentive_2d)),
        "--cut-off-p",
        str(float(args.cut_off_p)),
        "--topk-percent-2d",
        str(float(args.topk_percent_2d)),
        "--threshold-percent-2d",
        str(float(args.threshold_percent_2d)),
        "--step-pruning",
        str(int(args.step_pruning)),
        "--edit-strength",
        str(float(args.edit_strength)),
        "--guidance-scale",
        str(float(args.guidance_scale)),
        "--texture-render-method",
        args.texture_render_method,
        "--texture-diff-threshold",
        str(float(args.texture_diff_threshold)),
        "--metrics",
        *list(args.metrics),
    ]
    if args.resume:
        command.append("--skip-existing")
    if args.dataset:
        command.extend(["--dataset", args.dataset])
    if args.object_name:
        command.extend(["--object-name", args.object_name])
    if int(args.prompt_id) > 0:
        command.extend(["--prompt-id", str(int(args.prompt_id))])
    return command


def _count_existing_outputs(output_root: Path, cases: list[tuple[str, str, int]]) -> int:
    count = 0
    for dataset, object_name, prompt_id in cases:
        output_glb = output_root / dataset / object_name / f"prompt_{prompt_id}" / "edit.glb"
        if output_glb.is_file():
            count += 1
    return count


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def _write_summary(
    *,
    output_root: Path,
    args: argparse.Namespace,
    cases: list[tuple[str, str, int]],
    total_seconds: float,
) -> dict[str, Any]:
    manifest = _load_manifest(output_root / "manifest.json")
    failed_count = int(manifest.get("failed_count") or 0)
    success_count = int(manifest.get("success_count") or 0)
    skipped_count = int(manifest.get("skipped_count") or 0)
    generated_count = _count_existing_outputs(output_root, cases)
    payload = {
        "config_name": args.config_name,
        "run_group": args.run_group,
        "run_name": output_root.name,
        "method": "vecset_edit",
        "device": args.device,
        "eval_device": args.eval_device or args.device,
        "render_gpu_ids": parse_render_gpu_ids(args.render_gpus, device=args.device),
        "num_selected_cases": len(cases),
        "case_shard_count": int(args.case_shard_count),
        "case_shard_index": int(args.case_shard_index),
        "num_generated_cases": generated_count,
        "num_success_cases": success_count,
        "num_skipped_cases": skipped_count,
        "num_failed_cases": failed_count,
        "resume": bool(args.resume),
        "generate_only": bool(args.generate_only),
        "metrics": list(args.metrics),
        "dataset": args.dataset or None,
        "object_name": args.object_name or None,
        "prompt_id": int(args.prompt_id) if int(args.prompt_id) > 0 else None,
        "seed": int(args.seed),
        "total_seconds": round(float(total_seconds), 3),
        "created_at": utc_now_iso(),
    }
    write_json(output_root / "summary.json", payload)
    return payload


def _run_vecset_generation(args: argparse.Namespace) -> None:
    repo_root = ensure_vecset_edit_root()
    command = _build_vecset_command(args)
    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", str(int(args.seed)))
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    print(f"[VecSet] cwd={repo_root}")
    print(f"[VecSet] command={' '.join(command)}")
    if args.dry_run:
        return
    result = subprocess.run(command, cwd=repo_root, env=env, text=True)
    if result.returncode not in (0, 2):
        raise RuntimeError(f"VecSet generation failed with exit code {result.returncode}.")
    if result.returncode == 2:
        raise RuntimeError(
            "VecSet generation finished with failed cases. Inspect the output manifest before render/eval."
        )


def main() -> None:
    args = build_parser().parse_args()
    args.gt_root = args.gt_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.benchmark_root = args.benchmark_root.expanduser().resolve()
    if args.case_shard_count < 1:
        raise ValueError("--case-shard-count must be >= 1.")
    if args.case_shard_count > 1 and not args.generate_only:
        raise RuntimeError(
            "Sharded VecSet generation must use --generate-only. "
            "Run a final non-sharded --resume pass for render/eval."
        )

    metadata = load_benchmark_metadata(args.gt_root)
    cases = select_prompt_cases(
        metadata,
        gt_root=args.gt_root,
        limit=int(args.limit),
        dataset=args.dataset or None,
        object_name=args.object_name or None,
        prompt_id=int(args.prompt_id) if int(args.prompt_id) > 0 else None,
        require_source_glb=True,
        require_edit_image=True,
        require_render_image=True,
        require_mask_image=True,
    )
    cases = _select_case_shard(
        cases,
        shard_count=args.case_shard_count,
        shard_index=args.case_shard_index,
    )
    if not cases:
        raise RuntimeError("No benchmark cases selected for VecSet edit.")

    if args.output_root.exists() and not args.resume and not args.dry_run:
        shutil.rmtree(args.output_root)
    if not args.dry_run:
        ensure_dir(args.output_root)

    print(f"[Info] Selected {len(cases)} prompt cases")
    if args.dry_run:
        for dataset, object_name, prompt_id in cases[:8]:
            print(f"[Case] {dataset}/{object_name}/prompt_{prompt_id}")

    start_time = time.time()
    _run_vecset_generation(args)
    if args.dry_run:
        print("[Done] Dry-run only; no files were generated.")
        return

    summary = _write_summary(
        output_root=args.output_root,
        args=args,
        cases=cases,
        total_seconds=time.time() - start_time,
    )
    if args.generate_only:
        print(f"[Done] VecSet edit generation prepared: {args.output_root}")
        return

    render_gpu_ids = parse_render_gpu_ids(args.render_gpus, device=args.device)
    eval_device = args.eval_device or args.device

    print("[Render] Rendering benchmark views...")
    if not render_all_results(args.output_root, gpu_ids=render_gpu_ids, metrics=list(args.metrics)):
        raise RuntimeError("Benchmark rendering failed.")

    print("[Eval] Running benchmark evaluation...")
    eval_output_dir = ensure_dir(args.output_root / "evaluation_output")
    ok, results = run_evaluation(
        gt_root=args.gt_root,
        pred_root=args.output_root,
        metrics=list(args.metrics),
        output_dir=eval_output_dir,
        device=eval_device,
    )
    if not ok or results is None:
        raise RuntimeError("Benchmark evaluation failed.")

    total_time = time.time() - start_time
    save_results(
        output_root=args.output_root,
        entrypoint_name="vecset_edit",
        config_name=args.config_name,
        run_group=args.run_group,
        gt_root=args.gt_root,
        cases=cases,
        requested_metrics=list(args.metrics),
        benchmark_root=args.benchmark_root,
        skip_benchmark_render=False,
        results=results,
        total_time=total_time,
    )
    summary["total_seconds"] = round(float(total_time), 3)
    summary["generate_only"] = False
    write_json(args.output_root / "summary.json", summary)
    print(f"[Done] VecSet edit benchmark finished in {total_time:.1f}s")
    print(f"[Done] Pred root: {args.output_root}")


if __name__ == "__main__":
    main()
