#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from trellis_edit.benchmarks import (
    BenchmarkContext,
    BenchmarkRunner,
    CaseResult,
    RunResult,
    create_daily_bundle_from_run_result,
    infer_run_labels_from_daily_history,
    infer_run_labels_from_pred_root,
    rebuild_benchmark_indexes,
    update_daily_bundle_from_run_result,
)
from trellis_edit.benchmarks.shards import (
    load_run_result,
    merge_sharded_run_results,
    read_json,
    select_case_shard,
    serialize_run_result,
    shard_result_path,
    shard_state_dir,
)
from trellis_edit.benchmarks.tasks import (
    DEFAULT_DAILY_ROOT,
    DEFAULT_GEOMETRY_DATASET_ROOT,
    DEFAULT_GEOMETRY_IMAGE_SIZE,
    GEOMETRY_METRIC_NAMES,
    GeometryEditTask,
    iter_daily_run_roots,
)
from trellis_edit.common import ensure_dir, write_json


GEOMETRY_BUNDLE_ARTIFACT_FILES = {
    "geometry_metrics": "geometry_metrics.json",
}


def _summarize_metrics(
    case_results: list[CaseResult],
    *,
    metric_names: tuple[str, ...],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric_name in metric_names:
        values = [
            float(metric_value)
            for metric_value in (case_result.metrics.get(metric_name) for case_result in case_results)
            if metric_value is not None
        ]
        if not values:
            continue
        count = len(values)
        mean = float(sum(values) / count)
        variance = float(sum((value - mean) ** 2 for value in values) / count)
        summary[metric_name] = {
            "mean": mean,
            "std": variance ** 0.5,
            "count": count,
        }
    return summary


def _discover_case_tuples(*, gt_root: Path, pred_root: Path) -> list[tuple[str, str, int]]:
    task = GeometryEditTask(gt_root)
    cases = task.discover_cases(BenchmarkContext(pred_root=pred_root))
    return [
        (
            str(case.identity.dataset or ""),
            str(case.identity.object_name or ""),
            int(case.identity.prompt_id),
        )
        for case in cases
        if case.identity.dataset and case.identity.object_name and case.identity.prompt_id is not None
    ]


GEOMETRY_SHARD_STATE_DIR = "_geometry_eval_shards"
GEOMETRY_SHARD_RESULT_PREFIX = "geometry_metrics"
GEOMETRY_MERGED_FILENAME = "geometry_metrics.json"


def _merge_shard_run_result(
    *,
    gt_root: Path,
    pred_root: Path,
    shard_count: int,
) -> RunResult:
    if shard_count <= 1:
        raise ValueError("Shard merge requires --case-shard-count > 1.")

    shard_runs: list[RunResult] = []
    manifest_payload: dict[str, Any] = {}
    task_name = GeometryEditTask.name

    for shard_index in range(shard_count):
        shard_path = shard_result_path(
            pred_root,
            shard_count=shard_count,
            shard_index=shard_index,
            state_dir_name=GEOMETRY_SHARD_STATE_DIR,
            result_prefix=GEOMETRY_SHARD_RESULT_PREFIX,
            merged_filename=GEOMETRY_MERGED_FILENAME,
        )
        if not shard_path.is_file():
            raise RuntimeError(f"Missing shard result: {shard_path}")
        shard_run = load_run_result(shard_path)
        task_name = shard_run.task_name or task_name
        shard_runs.append(shard_run)
        if not manifest_payload:
            manifest_payload = dict(shard_run.manifest)

    case_order = {
        f"{dataset}/{object_name}/prompt_{prompt_id}": index
        for index, (dataset, object_name, prompt_id) in enumerate(
            _discover_case_tuples(gt_root=gt_root, pred_root=pred_root)
        )
    }
    def case_sort_key(case_result: CaseResult) -> tuple[int, str]:
        return (
            case_order.get(case_result.identity.case_id, 10**9),
            case_result.identity.case_id,
        )

    merged_case_results = [
        case_result
        for shard_run in shard_runs
        for case_result in shard_run.case_results
    ]
    merged_case_results.sort(key=case_sort_key)

    requested_metrics_tuple = (
        tuple(shard_runs[0].requested_metrics)
        if shard_runs and shard_runs[0].requested_metrics
        else tuple(GEOMETRY_METRIC_NAMES)
    )
    summary_metrics = _summarize_metrics(
        merged_case_results,
        metric_names=requested_metrics_tuple,
    )
    geometry_metrics_payload = {
        "task": task_name,
        "metrics": summary_metrics,
        "cases": {
            case_result.identity.case_id: {
                "metrics": dict(case_result.metrics),
                "task_meta": case_result.task_meta,
                "status": case_result.status,
            }
            for case_result in merged_case_results
        },
    }
    manifest_payload.update(
        {
            "gt_root": str(gt_root),
            "case_count": len(merged_case_results),
            "case_shard_count": shard_count,
            "case_shard_index": None,
        }
    )
    return merge_sharded_run_results(
        shard_runs,
        task_name=task_name,
        requested_metrics=list(requested_metrics_tuple),
        summary_metrics=summary_metrics,
        manifest=manifest_payload,
        run_artifacts={"geometry_metrics": geometry_metrics_payload},
        case_sort_key=case_sort_key,
    )


def _evaluate_pred_root(
    *,
    pred_root: Path,
    gt_root: Path,
    image_size: int,
    device: str,
    render_samples: int,
    overwrite_render: bool,
    quiet_blender: bool,
    cycles_backend: str,
    outside_points: int,
    outside_initial_samples: int,
    uni3d_model: str,
    cases: list[tuple[str, str, int]] | None = None,
) -> RunResult:
    task = GeometryEditTask(gt_root)
    runner = BenchmarkRunner()
    return runner.run(
        task,
        BenchmarkContext(
            pred_root=pred_root.expanduser().resolve(),
            requested_metrics=tuple(GEOMETRY_METRIC_NAMES),
            device=device,
            task_config={
                "image_size": image_size,
                "render_samples": render_samples,
                "overwrite_render": overwrite_render,
                "quiet_blender": quiet_blender,
                "cycles_backend": cycles_backend,
                "outside_points": outside_points,
                "outside_initial_samples": outside_initial_samples,
                "uni3d_model": uni3d_model,
                "cases": cases,
            },
        ),
    )


def _evaluate_run(
    *,
    run_root: Path,
    gt_root: Path,
    image_size: int,
    device: str,
    render_samples: int,
    overwrite_render: bool,
    quiet_blender: bool,
    cycles_backend: str,
    outside_points: int,
    outside_initial_samples: int,
    uni3d_model: str,
) -> dict[str, Any]:
    manifest_payload = read_json(run_root / "manifest.json")
    pred_root = Path(str(manifest_payload["pred_root"])).expanduser().resolve()
    run_result = _evaluate_pred_root(
        pred_root=pred_root,
        gt_root=gt_root,
        image_size=image_size,
        device=device,
        render_samples=render_samples,
        overwrite_render=overwrite_render,
        quiet_blender=quiet_blender,
        cycles_backend=cycles_backend,
        outside_points=outside_points,
        outside_initial_samples=outside_initial_samples,
        uni3d_model=uni3d_model,
    )
    write_json(pred_root / "geometry_metrics.json", run_result.run_artifacts.get("geometry_metrics", {}))
    update_daily_bundle_from_run_result(
        run_root=run_root,
        run_result=run_result,
        manifest_extra={"gt_root": str(gt_root)},
        bundle_artifact_files=GEOMETRY_BUNDLE_ARTIFACT_FILES,
    )
    return {
        "run_root": str(run_root),
        "pred_root": str(pred_root),
        "summary": run_result.summary_metrics,
        "case_count": len(run_result.case_results),
    }


def _resolved_geometry_run_labels(
    *,
    pred_root: Path,
    daily_root: Path,
    fallback_entrypoint: str,
) -> dict[str, Any]:
    run_labels = infer_run_labels_from_pred_root(
        pred_root,
        fallback_entrypoint=fallback_entrypoint,
    )
    history_labels = infer_run_labels_from_daily_history(daily_root, pred_root)
    if history_labels is not None:
        if history_labels.get("entrypoint_name"):
            run_labels["entrypoint_name"] = str(history_labels["entrypoint_name"])
        if history_labels.get("config_name"):
            run_labels["config_name"] = str(history_labels["config_name"])
    return run_labels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the geometry-only single-view editing benchmark and write benchmark bundles."
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=DEFAULT_GEOMETRY_DATASET_ROOT,
        help="Single-view benchmark dataset root.",
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
        help="Prediction root(s), e.g. pred/<run_name>.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        action="append",
        help="Specific daily run root(s). Defaults to all runs under --daily-root when --pred-root is omitted.",
    )
    parser.add_argument(
        "--run-group",
        type=str,
        default=None,
        help="Optional run group override when creating a new daily bundle from --pred-root.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--image-size", type=int, default=DEFAULT_GEOMETRY_IMAGE_SIZE)
    parser.add_argument("--render-samples", type=int, default=DEFAULT_GEOMETRY_IMAGE_SIZE // 16)
    parser.add_argument("--cycles-backend", type=str, default="AUTO")
    parser.add_argument("--outside-points", type=int, default=10000)
    parser.add_argument("--outside-initial-samples", type=int, default=40000)
    parser.add_argument("--uni3d-model", type=str, default="uni3d-g")
    parser.add_argument("--overwrite-render", action="store_true")
    parser.add_argument("--quiet-blender", action="store_true", default=True)
    parser.add_argument(
        "--case-shard-count",
        type=int,
        default=1,
        help="Split the discovered cases into this many disjoint evaluation shards.",
    )
    parser.add_argument(
        "--case-shard-index",
        type=int,
        default=0,
        help="Zero-based shard index to evaluate when --case-shard-count > 1.",
    )
    parser.add_argument(
        "--merge-shards",
        action="store_true",
        help="Merge previously written shard json files into geometry_metrics.json and the daily bundle.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    gt_root = args.gt_root.expanduser().resolve()
    daily_root = args.daily_root.expanduser().resolve()
    if not gt_root.exists():
        raise RuntimeError(f"GT root does not exist: {gt_root}")
    ensure_dir(daily_root)
    if int(args.case_shard_count) < 1:
        raise ValueError("--case-shard-count must be >= 1.")
    if int(args.case_shard_count) == 1 and int(args.case_shard_index) != 0:
        raise ValueError("--case-shard-index must be 0 when --case-shard-count == 1.")

    pred_roots = [path.expanduser().resolve() for path in (args.pred_root or [])]
    run_roots: list[Path] = []
    if args.run_root:
        run_roots = [path.expanduser().resolve() for path in args.run_root]
    elif not pred_roots:
        run_roots = iter_daily_run_roots(daily_root)
    if (int(args.case_shard_count) > 1 or bool(args.merge_shards)) and run_roots:
        raise RuntimeError("Case sharding and shard merging currently support --pred-root only.")

    results = []
    for pred_root in pred_roots:
        if not pred_root.exists():
            raise RuntimeError(f"Prediction root does not exist: {pred_root}")
        shard_count = int(args.case_shard_count)
        shard_index = int(args.case_shard_index)
        if args.merge_shards:
            run_result = _merge_shard_run_result(
                gt_root=gt_root,
                pred_root=pred_root,
                shard_count=shard_count,
            )
            write_json(pred_root / "geometry_metrics.json", run_result.run_artifacts.get("geometry_metrics", {}))

            run_labels = _resolved_geometry_run_labels(
                pred_root=pred_root,
                daily_root=daily_root,
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
                manifest_extra={"gt_root": str(gt_root)},
                bundle_artifact_files=GEOMETRY_BUNDLE_ARTIFACT_FILES,
            )
            results.append(
                {
                    "mode": "merge_shards",
                    "run_root": str(bundle_root),
                    "pred_root": str(pred_root),
                    "summary": run_result.summary_metrics,
                    "case_count": len(run_result.case_results),
                }
            )
            continue

        shard_cases: list[tuple[str, str, int]] | None = None
        if shard_count > 1:
            discovered_cases = _discover_case_tuples(gt_root=gt_root, pred_root=pred_root)
            shard_cases = select_case_shard(
                discovered_cases,
                shard_count=shard_count,
                shard_index=shard_index,
            )
            if not shard_cases:
                raise RuntimeError(
                    f"No cases selected for shard {shard_index}/{shard_count} under {pred_root}."
                )

        run_result = _evaluate_pred_root(
            pred_root=pred_root,
            gt_root=gt_root,
            image_size=int(args.image_size),
            device=args.device,
            render_samples=int(args.render_samples),
            overwrite_render=bool(args.overwrite_render),
            quiet_blender=bool(args.quiet_blender),
            cycles_backend=args.cycles_backend,
            outside_points=int(args.outside_points),
            outside_initial_samples=int(args.outside_initial_samples),
            uni3d_model=str(args.uni3d_model),
            cases=shard_cases,
        )

        if shard_count > 1:
            shard_dir = ensure_dir(shard_state_dir(pred_root, state_dir_name=GEOMETRY_SHARD_STATE_DIR))
            shard_path = shard_result_path(
                pred_root,
                shard_count=shard_count,
                shard_index=shard_index,
                state_dir_name=GEOMETRY_SHARD_STATE_DIR,
                result_prefix=GEOMETRY_SHARD_RESULT_PREFIX,
                merged_filename=GEOMETRY_MERGED_FILENAME,
            )
            write_json(shard_path, serialize_run_result(run_result))
            results.append(
                {
                    "mode": "pred_root_shard",
                    "pred_root": str(pred_root),
                    "shard_index": shard_index,
                    "shard_count": shard_count,
                    "case_count": len(run_result.case_results),
                    "summary": run_result.summary_metrics,
                    "shard_result": str(shard_path),
                }
            )
            continue

        write_json(pred_root / "geometry_metrics.json", run_result.run_artifacts.get("geometry_metrics", {}))

        run_labels = _resolved_geometry_run_labels(
            pred_root=pred_root,
            daily_root=daily_root,
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
            manifest_extra={"gt_root": str(gt_root)},
            bundle_artifact_files=GEOMETRY_BUNDLE_ARTIFACT_FILES,
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
                gt_root=gt_root,
                image_size=int(args.image_size),
                device=args.device,
                render_samples=int(args.render_samples),
                overwrite_render=bool(args.overwrite_render),
                quiet_blender=bool(args.quiet_blender),
                cycles_backend=args.cycles_backend,
                outside_points=int(args.outside_points),
                outside_initial_samples=int(args.outside_initial_samples),
                uni3d_model=str(args.uni3d_model),
            )
        )

    if not (int(args.case_shard_count) > 1 and not bool(args.merge_shards)):
        rebuild_benchmark_indexes(
            benchmark_root=daily_root.parent,
            suppress_focus_errors=True,
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
