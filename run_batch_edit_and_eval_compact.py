#!/usr/bin/env python3
"""
Compact batch runner for composable editing + Edit3D-Bench evaluation.

Compared to run_batch_edit_and_eval.py, this version keeps one long-lived
worker process per GPU and reuses a single loaded pipeline inside each worker.
That avoids reloading model weights and reinitializing CUDA context for every
single case.
"""

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

import yaml

from run_batch_edit_and_eval import (
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_GT_ROOT,
    DEFAULT_METRICS,
    DEFAULT_PRED_DIRNAME,
    estimate_remaining_time,
    format_time,
    load_edit3d_metadata,
    parse_gpu_list,
    prepare_case_run,
    run_evaluation,
    run_single_edit,
    save_results,
)
from trellis_edit.benchmarks import RunResult
from trellis_edit.common import ensure_dir
from trellis_edit.composable import has_entrypoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="批量运行 composable 编辑实验并评测（compact shared-pipeline 版本）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, help="结构化 YAML 配置文件路径（推荐）")
    parser.add_argument("--entrypoint", type=str, help="Composable entrypoint 名称")
    parser.add_argument("--gt-root", type=str, help="Edit3D-Bench GT 数据根目录")
    parser.add_argument("--config-name", type=str, help="本次批量评测的配置标识")
    parser.add_argument("--assets-root", type=str, help="源 3D 资产根目录，通常包含 voxels.ply 和 features.npz")
    parser.add_argument("--pred-root", type=str, help="批量评测输出根目录")
    parser.add_argument("--dataset", type=str, help="数据集过滤，例如 GSO")
    parser.add_argument("--object", type=str, help="物体名称过滤")
    parser.add_argument("--prompt-id", type=int, choices=[1, 2, 3], help="提示 ID 过滤")
    parser.add_argument("--max-cases", type=int, help="限制处理的案例数量")
    parser.add_argument("--model", type=str, help="覆盖 runtime.model")
    parser.add_argument("--seed", type=int, help="覆盖 runtime.seed")
    parser.add_argument("--device", type=str, help="覆盖 runtime.device 和评测 device")
    parser.add_argument("--gpus", type=str, help="编辑阶段使用的 GPU 列表，例如 '1,2'")
    parser.add_argument("--group", type=str, help="写入 daily/focus 的运行分组名")
    parser.add_argument("--metrics", nargs="+", help="覆盖评测指标")
    parser.add_argument("--skip-render", action="store_true", help="跳过 Edit3D-Bench 渲染步骤")
    parser.add_argument("--benchmark-root", type=str, help="benchmark daily/focus 根目录")
    skip_exists_group = parser.add_mutually_exclusive_group()
    skip_exists_group.add_argument("--skip-exists", dest="skip_exists", action="store_true", help="跳过已存在的 edit.glb")
    skip_exists_group.add_argument(
        "--no-skip-exists",
        dest="skip_exists",
        action="store_false",
        help="不跳过已存在的文件，重新生成所有结果",
    )
    parser.set_defaults(skip_exists=None)
    keep_outputs_group = parser.add_mutually_exclusive_group()
    keep_outputs_group.add_argument(
        "--keep-case-outputs",
        dest="keep_case_outputs",
        action="store_true",
        help="保留 _runs/<entrypoint>/<case>/ 下的完整单 case 输出",
    )
    keep_outputs_group.add_argument(
        "--drop-case-outputs",
        dest="keep_case_outputs",
        action="store_false",
        help="复制最终 edit.glb 后删除单 case 完整输出，仅保留评测所需文件",
    )
    parser.set_defaults(keep_case_outputs=None)
    parser.add_argument("--dry-run", action="store_true", help="只打印首个 case 解析后的 run_edit_experiment 配置")
    return parser


def _load_batch_config(args) -> tuple[dict[str, Any], dict[str, Any]]:
    config_data: dict[str, Any] = {}
    batch_config: dict[str, Any] = {}
    if not args.config:
        return config_data, batch_config

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] 配置文件不存在: {config_path}")
        raise SystemExit(1)

    with open(config_path, "r", encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle) or {}
    if not isinstance(config_data, dict):
        raise RuntimeError("批量配置文件必须是 YAML mapping")
    batch_config = dict(config_data.get("batch") or {})
    legacy_keys = [key for key in ("method_name", "method_args") if key in config_data]
    if legacy_keys:
        raise RuntimeError(
            "Legacy batch config keys are no longer supported: "
            + ", ".join(legacy_keys)
            + ". Use entrypoint + runtime/inputs/preprocess/ss/slat + batch."
        )
    return config_data, batch_config


def _select_cases(
    *,
    metadata: list[dict[str, Any]],
    dataset: str | None,
    object_name: str | None,
    prompt_id: int | None,
    max_cases: int | None,
) -> list[tuple[str, str, int]]:
    cases: list[tuple[str, str, int]] = []
    for entry in metadata:
        entry_dataset = entry.get("dataset")
        entry_object_name = entry.get("source_model")
        if not entry_dataset or not entry_object_name:
            continue
        if dataset and entry_dataset != dataset:
            continue
        if object_name and entry_object_name != object_name:
            continue
        for candidate_prompt_id in (1, 2, 3):
            if prompt_id and candidate_prompt_id != prompt_id:
                continue
            prompt_key = f"prompt_{candidate_prompt_id}"
            if entry.get(prompt_key):
                cases.append((entry_dataset, entry_object_name, candidate_prompt_id))

    if max_cases:
        cases = cases[: int(max_cases)]
    return cases


def _serialize_job(job) -> dict[str, str]:
    return {
        "case_name": job.case_name,
        "config_path": str(job.config_path),
        "expected_glb": str(job.expected_glb),
        "final_glb": str(job.final_glb),
        "case_output_dir": str(job.case_output_dir),
        "log_path": str(job.log_path),
    }


def _prepare_jobs(
    *,
    gt_root: Path,
    pred_root: Path,
    entrypoint_name: str,
    base_config: dict[str, Any],
    cases: list[tuple[str, str, int]],
    assets_root: Path | None,
    skip_exists: bool,
    model_override: str | None,
    seed_override: int | None,
) -> tuple[list[Any], int]:
    pending_jobs: list[Any] = []
    skip_count = 0
    for dataset, object_name, prompt_id in cases:
        prepared, skipped = prepare_case_run(
            gt_root=gt_root,
            pred_root=pred_root,
            entrypoint_name=entrypoint_name,
            dataset=dataset,
            object_name=object_name,
            prompt_id=prompt_id,
            base_config=base_config,
            assets_root=assets_root,
            skip_exists=skip_exists,
            model_override=model_override,
            seed_override=seed_override,
            device_override="cuda:0",
        )
        if skipped:
            skip_count += 1
            continue
        if prepared is not None:
            pending_jobs.append(prepared)
    return pending_jobs, skip_count


def _partition_jobs(jobs: list[Any], gpu_ids: list[str]) -> list[tuple[str, list[Any]]]:
    assignments: list[tuple[str, list[Any]]] = [(gpu_id, []) for gpu_id in gpu_ids]
    if not assignments:
        return assignments
    for index, job in enumerate(jobs):
        assignments[index % len(assignments)][1].append(job)
    return assignments


def _launch_compact_workers(
    *,
    pred_root: Path,
    assignments: list[tuple[str, list[Any]]],
    keep_case_outputs: bool,
) -> list[dict[str, Any]]:
    worker_manifest_dir = ensure_dir(pred_root / "_worker_manifests")
    worker_results_dir = ensure_dir(pred_root / "_worker_results")
    worker_logs_dir = ensure_dir(pred_root / "_worker_logs")

    workers: list[dict[str, Any]] = []
    repo_root = Path(__file__).resolve().parent
    worker_script = repo_root / "run_edit_experiment_worker.py"

    for worker_index, (gpu_id, jobs) in enumerate(assignments):
        if not jobs:
            continue

        worker_id = f"worker_gpu_{gpu_id}"
        worker_log_path = worker_logs_dir / f"{worker_id}.log"
        worker_result_path = worker_results_dir / f"{worker_id}.jsonl"
        manifest_path = worker_manifest_dir / f"{worker_id}.json"

        manifest = {
            "worker_id": worker_id,
            "gpu_id": gpu_id,
            "keep_case_outputs": keep_case_outputs,
            "worker_log_path": str(worker_log_path),
            "result_path": str(worker_result_path),
            "jobs": [_serialize_job(job) for job in jobs],
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env.setdefault("PYTHONUNBUFFERED", "1")
        log_handle = open(worker_log_path, "w", encoding="utf-8")
        cmd = [sys.executable, str(worker_script), "--manifest", str(manifest_path)]
        process = subprocess.Popen(
            cmd,
            cwd=repo_root,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        workers.append(
            {
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "jobs": jobs,
                "process": process,
                "log_handle": log_handle,
                "manifest_path": manifest_path,
                "result_path": worker_result_path,
                "worker_log_path": worker_log_path,
                "result_offset": 0,
                "completed_results": 0,
                "closed": False,
            }
        )
        print(
            f"[LAUNCH] {worker_id} -> cuda:{gpu_id} | "
            f"cases={len(jobs)} | log={worker_log_path}"
        )

    return workers


def _consume_worker_results(
    worker: dict[str, Any],
    *,
    counters: dict[str, int],
    total_cases: int,
    start_time: float,
) -> None:
    result_path = worker["result_path"]
    if not result_path.exists():
        return

    with open(result_path, "r", encoding="utf-8") as handle:
        handle.seek(worker["result_offset"])
        lines = handle.readlines()
        worker["result_offset"] = handle.tell()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        worker["completed_results"] += 1
        counters["finished"] += 1
        if payload.get("success"):
            counters["success"] += 1
            state = "SUCCESS"
        else:
            counters["failure"] += 1
            state = "FAIL"

        elapsed = time.time() - start_time
        eta = estimate_remaining_time(
            elapsed_seconds=elapsed,
            finished_cases=counters["finished"],
            total_cases=total_cases,
        )
        print(
            f"[{state}] {payload['case_name']} | gpu=cuda:{payload['gpu_id']} | "
            f"case_time={format_time(float(payload['duration_seconds']))} | "
            f"done={counters['finished']}/{total_cases} success={counters['success']} "
            f"failed={counters['failure']} skipped={counters['skip']} | "
            f"active_workers={counters['active_workers']} remaining={total_cases - counters['finished']} | "
            f"elapsed={format_time(elapsed)} ETA={eta}"
        )
        if not payload.get("success"):
            print(f"[FAIL] case log: {payload['case_log_path']}")
            print(f"[FAIL] worker log: {payload['worker_log_path']}")


def _monitor_compact_workers(
    *,
    workers: list[dict[str, Any]],
    skip_count: int,
    total_cases: int,
) -> tuple[int, int, int]:
    counters = {
        "finished": skip_count,
        "success": 0,
        "failure": 0,
        "skip": skip_count,
        "active_workers": len(workers),
    }
    start_time = time.time()
    last_status_time = 0.0

    while True:
        active_workers = 0
        for worker in workers:
            _consume_worker_results(
                worker,
                counters=counters,
                total_cases=total_cases,
                start_time=start_time,
            )
            returncode = worker["process"].poll()
            if returncode is None:
                active_workers += 1
                continue
            if worker["closed"]:
                continue

            worker["log_handle"].close()
            worker["closed"] = True

            missing = max(len(worker["jobs"]) - int(worker["completed_results"]), 0)
            if returncode != 0:
                if missing > 0:
                    counters["failure"] += missing
                    counters["finished"] += missing
                print(
                    f"[WORKER-FAIL] {worker['worker_id']} exited with code {returncode} "
                    f"(completed={worker['completed_results']}/{len(worker['jobs'])}) -> {worker['worker_log_path']}"
                )
            elif missing > 0:
                counters["failure"] += missing
                counters["finished"] += missing
                print(
                    f"[WORKER-INCOMPLETE] {worker['worker_id']} exited cleanly but only completed "
                    f"{worker['completed_results']}/{len(worker['jobs'])} jobs -> {worker['worker_log_path']}"
                )

        counters["active_workers"] = active_workers
        if all(worker["closed"] for worker in workers):
            break

        now = time.time()
        if now - last_status_time >= 60.0:
            elapsed = now - start_time
            eta = estimate_remaining_time(
                elapsed_seconds=elapsed,
                finished_cases=counters["finished"],
                total_cases=total_cases,
            )
            print(
                f"[STATUS] done={counters['finished']}/{total_cases} success={counters['success']} "
                f"failed={counters['failure']} skipped={counters['skip']} | "
                f"active_workers={active_workers} remaining={total_cases - counters['finished']} | "
                f"elapsed={format_time(elapsed)} ETA={eta}"
            )
            last_status_time = now
        time.sleep(5.0)

    return counters["success"], counters["skip"], counters["failure"]


def run_editing_and_eval_compact(
    *,
    gt_root: Path,
    pred_root: Path,
    entrypoint_name: str,
    config_name: str,
    base_config: dict[str, Any],
    cases: list[tuple[str, str, int]],
    assets_root: Path | None,
    metrics: list[str],
    device: str,
    gpu_ids: list[str],
    skip_benchmark_render: bool,
    skip_exists: bool,
    keep_case_outputs: bool,
    model_override: str | None,
    seed_override: int | None,
    dry_run: bool,
) -> tuple[bool, dict[str, Any] | None, RunResult | None]:
    if not dry_run and not skip_exists and pred_root.exists():
        print(f"[INFO] 清理旧的输出目录: {pred_root}")
        shutil.rmtree(pred_root)
    ensure_dir(pred_root)

    print("\n" + "=" * 80)
    print(f"输出目录: {pred_root}")
    print(f"Entrypoint: {entrypoint_name}")
    print(f"配置: {config_name}")
    print(f"案例数量: {len(cases)}")
    print(f"编辑使用 GPU: {', '.join(f'cuda:{gpu_id}' for gpu_id in gpu_ids)}")
    print(f"跳过已存在: {'是' if skip_exists else '否'}")
    print(f"保留单 case 输出: {'是' if keep_case_outputs else '否'}")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("步骤 1/4: 批量运行 composable 编辑实验（compact worker）")
    print("=" * 80)

    if dry_run:
        dataset, object_name, prompt_id = cases[0]
        success, _ = run_single_edit(
            gt_root=gt_root,
            pred_root=pred_root,
            entrypoint_name=entrypoint_name,
            dataset=dataset,
            object_name=object_name,
            prompt_id=prompt_id,
            base_config=base_config,
            assets_root=assets_root,
            skip_exists=skip_exists,
            keep_case_outputs=keep_case_outputs,
            model_override=model_override,
            seed_override=seed_override,
            device_override="cuda:0",
            dry_run=True,
        )
        if success:
            print("[DRY-RUN] 仅展示首个 case 的解析配置，提前结束。")
            return True, None, None
        return False, None, None

    pending_jobs, skip_count = _prepare_jobs(
        gt_root=gt_root,
        pred_root=pred_root,
        entrypoint_name=entrypoint_name,
        base_config=base_config,
        cases=cases,
        assets_root=assets_root,
        skip_exists=skip_exists,
        model_override=model_override,
        seed_override=seed_override,
    )
    if not pending_jobs:
        print("[INFO] 没有新的待处理 case。")
        success_count = 0
        failure_count = 0
    else:
        assignments = _partition_jobs(pending_jobs, gpu_ids)
        workers = _launch_compact_workers(
            pred_root=pred_root,
            assignments=assignments,
            keep_case_outputs=keep_case_outputs,
        )
        success_count, skip_count, failure_count = _monitor_compact_workers(
            workers=workers,
            skip_count=skip_count,
            total_cases=len(cases),
        )

    print(f"\n[INFO] 编辑完成: {success_count}/{len(cases)} 成功")
    if skip_count > 0:
        print(f"[INFO] 跳过已存在: {skip_count} 个")
    if failure_count > 0:
        print(f"[WARNING] 失败案例: {failure_count} 个")
    if success_count == 0:
        print("[ERROR] 没有成功的编辑结果")
        return False, None, None

    print("\n" + "=" * 80)
    print("步骤 2/4: 运行单图 benchmark 渲染 + 评测")
    print("=" * 80)
    eval_output_dir = pred_root / "evaluation_output"
    success, results, run_result = run_evaluation(
        gt_root=gt_root,
        pred_root=pred_root,
        metrics=metrics,
        output_dir=eval_output_dir,
        device=device,
        render_gpu_ids=gpu_ids,
        skip_render=skip_benchmark_render,
        cases=cases,
        batch_size=32,
        num_workers=4,
    )
    if not success:
        print("[WARNING] 评测失败")
        return True, None, None
    return True, results, run_result


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config_data, batch_config = _load_batch_config(args)
    entrypoint_name = args.entrypoint or config_data.get("entrypoint")
    if not entrypoint_name:
        print("[ERROR] 必须提供 entrypoint（CLI 或 YAML 顶层 entrypoint）")
        return 1
    if not has_entrypoint(entrypoint_name):
        print(f"[ERROR] Unknown composable entrypoint: {entrypoint_name}")
        return 1

    gt_root = Path(args.gt_root or batch_config.get("gt_root") or DEFAULT_GT_ROOT).expanduser().resolve()
    config_name = args.config_name or batch_config.get("config_name") or entrypoint_name
    run_group = args.group or batch_config.get("group")
    dataset = args.dataset or batch_config.get("dataset")
    object_name = args.object or batch_config.get("object")
    prompt_id = args.prompt_id or batch_config.get("prompt_id")
    max_cases = args.max_cases if args.max_cases is not None else batch_config.get("max_cases")
    benchmark_root_text = args.benchmark_root or batch_config.get("benchmark_root")
    benchmark_root = Path(benchmark_root_text).expanduser().resolve() if benchmark_root_text else DEFAULT_BENCHMARK_ROOT
    assets_root_text = args.assets_root or batch_config.get("assets_root")
    assets_root = Path(assets_root_text).expanduser().resolve() if assets_root_text else None
    pred_root_text = args.pred_root or batch_config.get("pred_root")
    pred_root = Path(pred_root_text).expanduser().resolve() if pred_root_text else (
        benchmark_root / DEFAULT_PRED_DIRNAME / f"{entrypoint_name}_{config_name}"
    ).resolve()
    metrics = args.metrics or batch_config.get("metrics") or DEFAULT_METRICS
    device = args.device or config_data.get("runtime", {}).get("device") or batch_config.get("device") or "cuda:0"
    gpu_ids = parse_gpu_list(args.gpus or batch_config.get("gpus"), fallback_device=device)
    skip_benchmark_render = args.skip_render or bool(batch_config.get("skip_benchmark_render", False))
    skip_exists = args.skip_exists
    if skip_exists is None:
        skip_exists = bool(batch_config.get("skip_exists", True))
    keep_case_outputs = args.keep_case_outputs
    if keep_case_outputs is None:
        keep_case_outputs = bool(batch_config.get("keep_case_outputs", False))

    if not gt_root.exists():
        print(f"[ERROR] GT 数据根目录不存在: {gt_root}")
        return 1

    runtime_defaults = dict(config_data.get("runtime") or {})
    model_override = args.model or runtime_defaults.get("model")
    seed_override = args.seed if args.seed is not None else runtime_defaults.get("seed")
    if seed_override is not None:
        seed_override = int(seed_override)

    print("=" * 80)
    print("批量运行 composable 编辑实验并评测（compact shared-pipeline 版本）")
    print("=" * 80)
    print(f"Entrypoint: {entrypoint_name}")
    print(f"配置: {config_name}")
    if run_group:
        print(f"运行分组: {run_group}")
    print(f"GT 数据: {gt_root}")
    print(f"预测输出: {pred_root}")
    print(f"benchmark 根目录: {benchmark_root}")
    if assets_root is not None:
        print(f"源资产: {assets_root}")
    if dataset:
        print(f"数据集过滤: {dataset}")
    if object_name:
        print(f"物体过滤: {object_name}")
    if prompt_id:
        print(f"提示 ID 过滤: {prompt_id}")
    if max_cases:
        print(f"案例限制: {max_cases}")
    print(f"评测指标: {metrics}")
    print(f"编辑 GPU 池: {', '.join(f'cuda:{gpu_id}' for gpu_id in gpu_ids)}")
    print(f"评测 GPU: {device}")
    print("=" * 80)

    metadata = load_edit3d_metadata(gt_root)
    cases = _select_cases(
        metadata=metadata,
        dataset=dataset,
        object_name=object_name,
        prompt_id=prompt_id,
        max_cases=max_cases,
    )
    print(f"[INFO] 找到 {len(cases)} 个案例")
    if not cases:
        print("[ERROR] 没有匹配的案例")
        return 1

    ensure_dir(pred_root)
    (pred_root / "_batch_source_config.yaml").write_text(
        yaml.safe_dump(
            config_data or {"entrypoint": entrypoint_name, "batch": batch_config},
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    start_time = time.time()
    success, results, run_result = run_editing_and_eval_compact(
        gt_root=gt_root,
        pred_root=pred_root,
        entrypoint_name=entrypoint_name,
        config_name=config_name,
        base_config=config_data or {"entrypoint": entrypoint_name},
        cases=cases,
        assets_root=assets_root,
        metrics=list(metrics),
        device=device,
        gpu_ids=gpu_ids,
        skip_benchmark_render=skip_benchmark_render,
        skip_exists=skip_exists,
        keep_case_outputs=keep_case_outputs,
        model_override=model_override,
        seed_override=seed_override,
        dry_run=args.dry_run,
    )
    if not success:
        print("[ERROR] 处理失败")
        return 1

    total_time = time.time() - start_time
    if results is not None:
        save_results(
            output_root=pred_root,
            entrypoint_name=entrypoint_name,
            config_name=config_name,
            run_group=run_group,
            benchmark_root=benchmark_root,
            run_result=run_result,
            total_time=total_time,
        )

    print("\n" + "=" * 80)
    print("批量处理完成")
    print("=" * 80)
    print(f"总耗时: {format_time(total_time)}")
    print(f"输出目录: {pred_root}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
