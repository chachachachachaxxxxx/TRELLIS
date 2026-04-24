from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from trellis_edit.benchmarks.single_view_common import (
    build_single_view_case_id,
    build_single_view_per_case_metrics,
    build_single_view_prompt_text_map,
    load_optional_json,
    load_single_view_metadata,
    requested_single_view_case_metrics,
)
from trellis_edit.common import ensure_dir, write_json
from trellis_edit.metrics.floaters import (
    compute_local_mesh_metric_payloads,
    split_local_mesh_metrics,
)

from ..base import BenchmarkContext, BenchmarkTask, CaseIdentity, CaseResult
from ..runner import BenchmarkRunner
from ..vendors import evaluate_single_view_predictions, get_edit3d_bench_render_script


DEFAULT_RENDER_TIMEOUT_SECONDS = 7200


@dataclass(frozen=True)
class SingleViewCase:
    identity: CaseIdentity
    prompt_dir: Path
    pred_dir: Path
    prompt_text: str | None


def render_script_path() -> Path:
    return get_edit3d_bench_render_script()


def _compact_metric_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload.get(key)
        for key in ("mean", "std", "count")
        if key in payload
    }


def _default_render_gpu_ids(*, device: str) -> list[str]:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        gpu_list = [item.strip() for item in visible_devices.split(",") if item.strip()]
        if gpu_list:
            return gpu_list
    if device.startswith("cuda:"):
        return [device.split(":", 1)[1]]
    return []


def _metrics_require_video(metrics: Sequence[str]) -> bool:
    requested = {str(metric).strip().lower() for metric in metrics}
    return "fvd" in requested


def _render_single_view_predictions(
    *,
    pred_root: Path,
    gpu_ids: list[str],
    metrics: Sequence[str],
    timeout_seconds: int = DEFAULT_RENDER_TIMEOUT_SECONDS,
) -> bool:
    render_script = render_script_path()
    if not render_script.is_file():
        raise RuntimeError(f"Cannot locate vendored render script: {render_script}")
    if not gpu_ids:
        raise RuntimeError("Single-view render requires at least one GPU id.")

    skip_video = not _metrics_require_video(metrics)
    render_logs_dir = ensure_dir(pred_root / "_render_logs")
    shard_count = len(gpu_ids)
    processes: list[tuple[str, subprocess.Popen[str], Any, Path]] = []

    for shard_id, gpu_id in enumerate(gpu_ids):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env.setdefault("PYTHONUNBUFFERED", "1")

        cmd = [
            sys.executable,
            str(render_script),
            "--base_dir",
            str(pred_root.resolve()),
            "--num_shards",
            str(shard_count),
            "--shard_id",
            str(shard_id),
            "--quiet_blender",
            "--skip_existing",
        ]
        if skip_video:
            cmd.append("--skip_video")

        log_path = render_logs_dir / f"render_shard_{shard_id}.log"
        log_handle = open(log_path, "w", encoding="utf-8")
        process = subprocess.Popen(
            cmd,
            cwd=render_script.parent,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((gpu_id, process, log_handle, log_path))

    success = True
    for gpu_id, process, log_handle, log_path in processes:
        returncode = process.wait(timeout=timeout_seconds)
        log_handle.close()
        if returncode != 0:
            print(f"[Render] Shard on cuda:{gpu_id} failed -> {log_path}")
            success = False
            continue

        tail = ""
        try:
            lines = log_path.read_text(encoding="utf-8").splitlines()
            tail = "\n".join(lines[-4:])
        except Exception:
            tail = ""
        print(f"[Render] Shard on cuda:{gpu_id} finished")
        if tail:
            print(tail)
    return success


def _normalize_case_tuples(raw_cases: Sequence[Any] | None) -> list[tuple[str, str, int]]:
    if raw_cases is None:
        return []

    resolved: list[tuple[str, str, int]] = []
    seen: set[tuple[str, str, int]] = set()
    for item in raw_cases:
        if isinstance(item, (tuple, list)) and len(item) == 3:
            dataset = str(item[0]).strip()
            object_name = str(item[1]).strip()
            prompt_id = int(item[2])
        elif isinstance(item, dict):
            dataset = str(item.get("dataset") or "").strip()
            object_name = str(item.get("object_name") or item.get("source_model") or "").strip()
            prompt_id = int(item.get("prompt_id"))
        else:
            raise RuntimeError(
                "Single-view task cases must be tuples like "
                "(dataset, object_name, prompt_id) or mappings with the same fields."
            )
        case_tuple = (dataset, object_name, prompt_id)
        if case_tuple in seen:
            continue
        seen.add(case_tuple)
        resolved.append(case_tuple)
    return resolved


def _discover_case_tuples(
    *,
    gt_root: Path,
    pred_root: Path,
    metadata: list[dict[str, Any]],
    explicit_cases: Sequence[Any] | None,
) -> list[tuple[str, str, int]]:
    selected_cases = _normalize_case_tuples(explicit_cases)
    if selected_cases:
        return selected_cases

    cases: list[tuple[str, str, int]] = []
    for row in metadata:
        dataset = str(row.get("dataset") or "").strip()
        object_name = str(row.get("source_model") or "").strip()
        if not dataset or not object_name:
            continue
        for prompt_id in (1, 2, 3):
            prompt_key = f"prompt_{prompt_id}"
            prompt_dir = gt_root / dataset / object_name / prompt_key
            pred_dir = pred_root / dataset / object_name / prompt_key
            if not prompt_dir.is_dir():
                continue
            if not pred_dir.exists() and not (pred_dir / "edit.glb").is_file():
                continue
            cases.append((dataset, object_name, prompt_id))
    return cases


def _build_cases(
    *,
    gt_root: Path,
    pred_root: Path,
    metadata: list[dict[str, Any]],
    case_tuples: list[tuple[str, str, int]],
) -> list[SingleViewCase]:
    prompt_text_map = build_single_view_prompt_text_map(metadata)
    cases: list[SingleViewCase] = []
    for dataset, object_name, prompt_id in case_tuples:
        identity = CaseIdentity(
            case_id=build_single_view_case_id(dataset, object_name, prompt_id),
            dataset=dataset,
            object_name=object_name,
            prompt_id=prompt_id,
        )
        prompt_key = f"prompt_{prompt_id}"
        cases.append(
            SingleViewCase(
                identity=identity,
                prompt_dir=gt_root / dataset / object_name / prompt_key,
                pred_dir=pred_root / dataset / object_name / prompt_key,
                prompt_text=prompt_text_map.get((dataset, object_name, prompt_id)),
            )
        )
    return cases


def _case_artifacts(case: SingleViewCase) -> dict[str, Path]:
    return {
        "source_image": case.prompt_dir / "2d_render.png",
        "edit_image": case.prompt_dir / "2d_edit.png",
        "mask_image": case.prompt_dir / "2d_mask.png",
        "edit_front_image": case.pred_dir / "images" / "render_0000.png",
        "source_model_glb": case.prompt_dir.parent / "source_model" / "model.glb",
        "source_voxelmesh_glb": case.pred_dir / "source_voxelmesh" / "voxel_mesh.glb",
        "source_voxelmesh_transform": case.pred_dir / "source_voxelmesh" / "voxel_mesh_transform.json",
        "mask_glb": case.prompt_dir / "3d_edit_region.glb",
        "edit_glb": case.pred_dir / "edit.glb",
        "ss_voxelmesh_glb": case.pred_dir / "ss" / "voxel_mesh.glb",
        "ss_voxelmesh_transform": case.pred_dir / "ss" / "voxel_mesh_transform.json",
        "ss_coords_ply": case.pred_dir / "ss" / "coords.ply",
        "ss_metadata": case.pred_dir / "ss" / "ss_metadata.json",
        "images": case.pred_dir / "images",
        "videos": case.pred_dir / "videos",
    }


class SingleViewEditTask(BenchmarkTask):
    name = "single_view_edit"

    def __init__(self, gt_root: Path):
        self.gt_root = gt_root.expanduser().resolve()
        self._metadata = load_single_view_metadata(self.gt_root)
        self._summary_payload: dict[str, Any] = {}
        self._summary_metrics: dict[str, Any] = {}
        self._output_dir: Path | None = None

    def discover_cases(self, context: BenchmarkContext) -> Sequence[SingleViewCase]:
        pred_root = context.pred_root.expanduser().resolve()
        case_tuples = _discover_case_tuples(
            gt_root=self.gt_root,
            pred_root=pred_root,
            metadata=self._metadata,
            explicit_cases=context.task_config.get("cases"),
        )
        return _build_cases(
            gt_root=self.gt_root,
            pred_root=pred_root,
            metadata=self._metadata,
            case_tuples=case_tuples,
        )

    def evaluate_cases(
        self,
        prepared_cases: Sequence[SingleViewCase],
        context: BenchmarkContext,
    ) -> Sequence[CaseResult]:
        pred_root = context.pred_root.expanduser().resolve()
        output_dir = (
            context.output_root.expanduser().resolve()
            if context.output_root is not None
            else (pred_root / "evaluation_output").resolve()
        )
        ensure_dir(output_dir)
        self._output_dir = output_dir

        task_config = dict(context.task_config)
        requested_metrics = list(context.requested_metrics)
        external_metrics, local_mesh_metrics = split_local_mesh_metrics(requested_metrics)

        skip_render = bool(task_config.get("skip_render", False))
        render_gpu_ids = [
            str(gpu_id).strip()
            for gpu_id in task_config.get("render_gpu_ids", [])
            if str(gpu_id).strip()
        ]
        if not skip_render:
            if not render_gpu_ids:
                render_gpu_ids = _default_render_gpu_ids(device=context.device)
            _render_single_view_predictions(
                pred_root=pred_root,
                gpu_ids=render_gpu_ids,
                metrics=requested_metrics,
            )

        if external_metrics:
            summary_payload = evaluate_single_view_predictions(
                gt_root=self.gt_root,
                pred_root=pred_root,
                metrics=external_metrics,
                output_dir=output_dir,
                device=context.device,
                batch_size=int(task_config.get("batch_size", 32)),
                num_workers=int(task_config.get("num_workers", 4)),
                save_detailed=True,
            )
        else:
            summary_payload = {
                "config": {
                    "gt_root": str(self.gt_root),
                    "pred_root": str(pred_root),
                    "metrics": [],
                    "device": context.device,
                },
                "results": {},
            }

        if local_mesh_metrics:
            local_payloads = compute_local_mesh_metric_payloads(
                gt_root=self.gt_root,
                pred_root=pred_root,
                metrics=local_mesh_metrics,
            )
            summary_payload.setdefault("results", {}).update(
                {
                    metric_name: _compact_metric_payload(metric_payload)
                    for metric_name, metric_payload in local_payloads.items()
                }
            )
            detailed_payload = load_optional_json(output_dir / "detailed_results.json")
            detailed_payload.update(local_payloads)
            write_json(output_dir / "detailed_results.json", detailed_payload)

        summary_payload.setdefault("config", {})["metrics"] = requested_metrics
        write_json(output_dir / "summary.json", summary_payload)

        detailed_results = load_optional_json(output_dir / "detailed_results.json")
        case_tuples = [
            (
                str(case.identity.dataset or ""),
                str(case.identity.object_name or ""),
                int(case.identity.prompt_id or 0),
            )
            for case in prepared_cases
        ]
        per_case_metrics = build_single_view_per_case_metrics(
            gt_root=self.gt_root,
            pred_root=pred_root,
            metadata=self._metadata,
            cases=case_tuples,
            requested_metrics=requested_metrics,
            detailed_results=detailed_results,
        )
        requested_case_metrics = requested_single_view_case_metrics(requested_metrics)

        case_results: list[CaseResult] = []
        for case in prepared_cases:
            artifact_map = _case_artifacts(case)
            metrics = {
                metric_name: per_case_metrics.get(case.identity.case_id, {}).get(metric_name)
                for metric_name in requested_case_metrics
            }
            has_edit = artifact_map["edit_glb"].is_file()
            has_render = artifact_map["images"].is_dir() or artifact_map["videos"].is_dir()
            if not has_edit:
                status = "failed"
            elif skip_render or has_render:
                status = "ok"
            else:
                status = "partial"
            case_results.append(
                CaseResult(
                    identity=case.identity,
                    status=status,
                    metrics=metrics,
                    artifacts=artifact_map,
                    prompt_text=case.prompt_text,
                )
            )

        self._summary_payload = summary_payload
        self._summary_metrics = dict(summary_payload.get("results") or {})
        return case_results

    def summarize_run(
        self,
        case_results: Sequence[CaseResult],
        context: BenchmarkContext,
    ) -> dict[str, Any]:
        return dict(self._summary_metrics)

    def build_manifest(
        self,
        case_results: Sequence[CaseResult],
        context: BenchmarkContext,
    ) -> dict[str, Any]:
        manifest = super().build_manifest(case_results, context)
        manifest.update(
            {
                "gt_root": str(self.gt_root),
                "case_count": len(case_results),
                "output_dir": str(self._output_dir) if self._output_dir is not None else None,
            }
        )
        return manifest

    def collect_run_artifacts(
        self,
        case_results: Sequence[CaseResult],
        context: BenchmarkContext,
    ) -> dict[str, Any]:
        return {
            "summary_payload": dict(self._summary_payload),
            "evaluation_output_dir": str(self._output_dir) if self._output_dir is not None else None,
        }


def run_single_view_benchmark(
    *,
    gt_root: Path,
    pred_root: Path,
    metrics: list[str],
    output_dir: Path,
    device: str = "cuda:0",
    render_gpu_ids: list[str] | None = None,
    skip_render: bool = False,
    batch_size: int = 32,
    num_workers: int = 4,
    cases: Sequence[Any] | None = None,
):
    task = SingleViewEditTask(gt_root)
    runner = BenchmarkRunner()
    return runner.run(
        task,
        BenchmarkContext(
            pred_root=pred_root.expanduser().resolve(),
            requested_metrics=tuple(metrics),
            device=device,
            output_root=output_dir.expanduser().resolve(),
            task_config={
                "cases": list(cases) if cases is not None else None,
                "skip_render": bool(skip_render),
                "render_gpu_ids": list(render_gpu_ids or []),
                "batch_size": int(batch_size),
                "num_workers": int(num_workers),
            },
        ),
    )


def run_single_view_evaluation(
    *,
    gt_root: Path,
    pred_root: Path,
    metrics: list[str],
    output_dir: Path,
    device: str = "cuda:0",
    batch_size: int = 32,
    num_workers: int = 4,
) -> dict[str, Any]:
    run_result = run_single_view_benchmark(
        gt_root=gt_root,
        pred_root=pred_root,
        metrics=metrics,
        output_dir=output_dir,
        device=device,
        skip_render=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return dict(run_result.run_artifacts.get("summary_payload") or {})
