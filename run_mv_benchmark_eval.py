#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from trellis_edit.benchmarks import (
    BenchmarkContext,
    BenchmarkRunner,
    RunResult,
    create_daily_bundle_from_run_result,
    infer_run_labels_from_pred_root,
    rebuild_benchmark_indexes,
    update_daily_bundle_from_run_result,
)
from trellis_edit.benchmarks.tasks import (
    DEFAULT_DAILY_ROOT,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MV_DATASET_ROOT,
    MV_METRIC_NAMES,
    MultiViewEditTask,
    iter_daily_run_roots,
)
from trellis_edit.common import ensure_dir, write_json


MV_BUNDLE_ARTIFACT_FILES = {
    "mv_metrics": "mv_metrics.json",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _peer_pred_roots_from_run_roots(peer_run_roots: list[Path]) -> list[Path]:
    pred_roots: list[Path] = []
    for peer_run_root in peer_run_roots:
        peer_manifest_path = peer_run_root / "manifest.json"
        if not peer_manifest_path.is_file():
            continue
        peer_manifest_payload = _read_json(peer_manifest_path)
        peer_pred_root = peer_manifest_payload.get("pred_root")
        if peer_pred_root:
            pred_roots.append(Path(str(peer_pred_root)).expanduser().resolve())
    return pred_roots


def _evaluate_pred_root(
    *,
    pred_root: Path,
    dataset_root: Path,
    image_size: int,
    device: str,
    overwrite_render: bool,
    quiet_blender: bool,
    cycles_backend: str,
    peer_pred_roots: list[Path],
) -> RunResult:
    task = MultiViewEditTask(dataset_root)
    runner = BenchmarkRunner()
    return runner.run(
        task,
        BenchmarkContext(
            pred_root=pred_root.expanduser().resolve(),
            requested_metrics=tuple(MV_METRIC_NAMES),
            device=device,
            task_config={
                "image_size": image_size,
                "overwrite_render": overwrite_render,
                "quiet_blender": quiet_blender,
                "cycles_backend": cycles_backend,
                "peer_pred_roots": [str(path) for path in peer_pred_roots],
            },
        ),
    )


def _evaluate_run(
    *,
    run_root: Path,
    dataset_root: Path,
    image_size: int,
    device: str,
    overwrite_render: bool,
    quiet_blender: bool,
    cycles_backend: str,
    peer_run_roots: list[Path],
) -> dict[str, Any]:
    manifest_payload = _read_json(run_root / "manifest.json")
    pred_root = Path(str(manifest_payload["pred_root"])).expanduser().resolve()
    run_result = _evaluate_pred_root(
        pred_root=pred_root,
        dataset_root=dataset_root,
        image_size=image_size,
        device=device,
        overwrite_render=overwrite_render,
        quiet_blender=quiet_blender,
        cycles_backend=cycles_backend,
        peer_pred_roots=_peer_pred_roots_from_run_roots(peer_run_roots),
    )
    write_json(pred_root / "mv_metrics.json", run_result.run_artifacts.get("mv_metrics", {}))
    update_daily_bundle_from_run_result(
        run_root=run_root,
        run_result=run_result,
        manifest_extra={"dataset_root": str(dataset_root)},
        bundle_artifact_files=MV_BUNDLE_ARTIFACT_FILES,
    )
    return {
        "run_root": str(run_root),
        "pred_root": str(pred_root),
        "summary": run_result.summary_metrics,
        "case_count": len(run_result.case_results),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate multi-view editing metrics and either augment or create benchmark daily bundles."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_MV_DATASET_ROOT,
        help="Multi-view benchmark dataset root.",
    )
    parser.add_argument(
        "--daily-root",
        type=Path,
        default=DEFAULT_DAILY_ROOT,
        help="Benchmark daily root.",
    )
    parser.add_argument(
        "--pred-root",
        type=Path,
        action="append",
        help="Raw multi-view prediction root(s), e.g. pred_mv/<run_name>.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        action="append",
        help="Specific daily run root(s). Defaults to all runs under --daily-root.",
    )
    parser.add_argument(
        "--peer-run-root",
        type=Path,
        action="append",
        help="Optional peer run root(s) for cross-seed novel-view LPIPS.",
    )
    parser.add_argument(
        "--peer-pred-root",
        type=Path,
        action="append",
        help="Optional peer raw prediction root(s) for cross-seed novel-view LPIPS.",
    )
    parser.add_argument(
        "--run-group",
        type=str,
        default=None,
        help="Optional run group override when creating a new daily bundle from --pred-root.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--cycles-backend", type=str, default="AUTO")
    parser.add_argument("--overwrite-render", action="store_true")
    parser.add_argument("--quiet-blender", action="store_true", default=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    daily_root = args.daily_root.expanduser().resolve()
    if not dataset_root.exists():
        raise RuntimeError(f"Dataset root does not exist: {dataset_root}")
    ensure_dir(daily_root)

    pred_roots = [path.expanduser().resolve() for path in (args.pred_root or [])]
    peer_pred_roots = [path.expanduser().resolve() for path in (args.peer_pred_root or [])]
    peer_run_roots = [path.expanduser().resolve() for path in (args.peer_run_root or [])]
    run_roots: list[Path] = []
    if args.run_root:
        run_roots = [path.expanduser().resolve() for path in args.run_root]
    elif not pred_roots:
        run_roots = iter_daily_run_roots(daily_root)

    results = []
    for pred_root in pred_roots:
        if not pred_root.exists():
            raise RuntimeError(f"Prediction root does not exist: {pred_root}")
        run_result = _evaluate_pred_root(
            pred_root=pred_root,
            dataset_root=dataset_root,
            image_size=int(args.image_size),
            device=args.device,
            overwrite_render=bool(args.overwrite_render),
            quiet_blender=bool(args.quiet_blender),
            cycles_backend=args.cycles_backend,
            peer_pred_roots=peer_pred_roots if len(pred_roots) == 1 else [],
        )
        write_json(pred_root / "mv_metrics.json", run_result.run_artifacts.get("mv_metrics", {}))

        run_labels = infer_run_labels_from_pred_root(
            pred_root,
            fallback_entrypoint=run_result.task_name,
        )
        bundle_root = create_daily_bundle_from_run_result(
            daily_root=daily_root,
            pred_root=pred_root,
            entrypoint_name=run_labels["entrypoint_name"],
            config_name=run_labels["config_name"],
            run_group=args.run_group if args.run_group is not None else run_labels["group"],
            total_time_seconds=float(run_labels["total_time_seconds"]),
            run_result=run_result,
            manifest_extra={"dataset_root": str(dataset_root)},
            bundle_artifact_files=MV_BUNDLE_ARTIFACT_FILES,
        )
        results.append(
            {
                "mode": "pred_root",
                "run_root": str(bundle_root),
                "pred_root": str(pred_root),
                "summary": run_result.summary_metrics,
                "case_count": len(run_result.case_results),
            }
        )

    for run_root in run_roots:
        if not run_root.exists():
            raise RuntimeError(f"Daily run root does not exist: {run_root}")
        results.append(
            _evaluate_run(
                run_root=run_root,
                dataset_root=dataset_root,
                image_size=int(args.image_size),
                device=args.device,
                overwrite_render=bool(args.overwrite_render),
                quiet_blender=bool(args.quiet_blender),
                cycles_backend=args.cycles_backend,
                peer_run_roots=peer_run_roots if len(run_roots) == 1 else [],
            )
        )

    rebuild_benchmark_indexes(
        benchmark_root=daily_root.parent,
        suppress_focus_errors=True,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
