#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import run_edit_experiment
from trellis_edit.common import build_backend_config, ensure_dir, release_cuda_memory
from trellis_edit.composable import ComposableExperimentRunner, get_entrypoint, has_entrypoint


@dataclass(frozen=True)
class WorkerJob:
    case_name: str
    config_path: Path
    expected_glb: Path
    final_glb: Path
    case_output_dir: Path
    log_path: Path


@dataclass(frozen=True)
class WorkerManifest:
    worker_id: str
    gpu_id: str
    keep_case_outputs: bool
    result_path: Path
    worker_log_path: Path
    jobs: tuple[WorkerJob, ...]


class SharedPipelineComposableExperimentRunner(ComposableExperimentRunner):
    """Composable runner variant that reuses an already-loaded pipeline."""

    def __init__(self, config, pipeline):
        super().__init__(config)
        self._shared_pipeline = pipeline

    def _load_pipeline(self):
        return self._shared_pipeline


def _serialize_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _load_manifest(path: Path) -> WorkerManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    jobs = tuple(
        WorkerJob(
            case_name=str(item["case_name"]),
            config_path=_serialize_path(item["config_path"]),
            expected_glb=_serialize_path(item["expected_glb"]),
            final_glb=_serialize_path(item["final_glb"]),
            case_output_dir=_serialize_path(item["case_output_dir"]),
            log_path=_serialize_path(item["log_path"]),
        )
        for item in payload.get("jobs", [])
    )
    return WorkerManifest(
        worker_id=str(payload["worker_id"]),
        gpu_id=str(payload["gpu_id"]),
        keep_case_outputs=bool(payload.get("keep_case_outputs", False)),
        result_path=_serialize_path(payload["result_path"]),
        worker_log_path=_serialize_path(payload["worker_log_path"]),
        jobs=jobs,
    )


def _load_case_config(config_path: Path):
    parser = run_edit_experiment.build_parser()
    args = parser.parse_args(["--config", str(config_path)])
    config_data = run_edit_experiment._load_config(args, parser)

    entrypoint_name = args.entrypoint or config_data.get("entrypoint")
    if not entrypoint_name:
        raise RuntimeError(f"Config is missing entrypoint: {config_path}")
    if not has_entrypoint(entrypoint_name):
        raise RuntimeError(f"Unknown composable entrypoint in {config_path}: {entrypoint_name}")

    runtime = run_edit_experiment._build_runtime(args)
    inputs = run_edit_experiment._build_inputs(args)
    entrypoint = get_entrypoint(entrypoint_name)
    config = entrypoint.build(runtime=runtime, inputs=inputs)
    config = run_edit_experiment._apply_config_overrides(config, config_data)
    config.validate_inputs()
    return config


def _shared_pipeline_signature(config) -> tuple[str, str, str, str]:
    return (
        str(config.runtime.model),
        str(config.runtime.attn_backend),
        str(config.runtime.sparse_attn_backend),
        str(config.runtime.spconv_algo),
    )


def _load_shared_pipeline(config):
    backend = build_backend_config(
        attn_backend=config.runtime.attn_backend,
        sparse_attn_backend=config.runtime.sparse_attn_backend,
        spconv_algo=config.runtime.spconv_algo,
    )
    for key, value in backend.env.items():
        os.environ[key] = value

    import torch
    from trellis.pipelines import TrellisImageTo3DPipeline

    requested_device = str(config.runtime.device)
    if "CUDA_VISIBLE_DEVICES" in os.environ and requested_device.startswith("cuda:"):
        load_device = "cuda:0"
    else:
        load_device = requested_device

    print(
        "[WORKER] Loading shared pipeline "
        f"(model={config.runtime.model}, device={load_device}, "
        f"attn={backend.attn_backend}, sparse_attn={backend.sparse_attn_backend}, "
        f"spconv={backend.spconv_algo})"
    )
    pipeline = TrellisImageTo3DPipeline.from_pretrained(config.runtime.model)
    pipeline.to(torch.device(load_device))
    return pipeline


def _write_case_log(
    path: Path,
    *,
    worker_log_path: Path,
    case_name: str,
    gpu_id: str,
    success: bool,
    duration_seconds: float,
    final_glb: Path,
    error: str | None = None,
    traceback_text: str | None = None,
) -> None:
    ensure_dir(path.parent)
    lines = [
        f"case_name: {case_name}",
        f"gpu: cuda:{gpu_id}",
        f"status: {'success' if success else 'fail'}",
        f"duration_seconds: {duration_seconds:.3f}",
        f"final_glb: {final_glb}",
        f"worker_log: {worker_log_path}",
    ]
    if error:
        lines.append(f"error: {error}")
    if traceback_text:
        lines.append("")
        lines.append("traceback:")
        lines.append(traceback_text.rstrip())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_result(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def _run_job(
    *,
    job: WorkerJob,
    manifest: WorkerManifest,
    pipeline,
    expected_signature: tuple[str, str, str, str],
) -> bool:
    case_start = time.time()
    print(f"[CASE] start {job.case_name}")

    try:
        config = _load_case_config(job.config_path)
        actual_signature = _shared_pipeline_signature(config)
        if actual_signature != expected_signature:
            raise RuntimeError(
                "Compact worker requires a shared model/backend setup per worker. "
                f"Expected {expected_signature}, got {actual_signature} for {job.case_name}."
            )

        runner = SharedPipelineComposableExperimentRunner(config, pipeline)
        runner.run()

        if not job.expected_glb.is_file():
            raise RuntimeError(f"Expected GLB was not produced: {job.expected_glb}")

        ensure_dir(job.final_glb.parent)
        shutil.copy2(job.expected_glb, job.final_glb)
        if not manifest.keep_case_outputs and job.case_output_dir.exists():
            shutil.rmtree(job.case_output_dir)

        duration = time.time() - case_start
        payload = {
            "case_name": job.case_name,
            "success": True,
            "gpu_id": manifest.gpu_id,
            "duration_seconds": duration,
            "final_glb": str(job.final_glb),
            "worker_log_path": str(manifest.worker_log_path),
            "case_log_path": str(job.log_path),
        }
        _write_case_log(
            job.log_path,
            worker_log_path=manifest.worker_log_path,
            case_name=job.case_name,
            gpu_id=manifest.gpu_id,
            success=True,
            duration_seconds=duration,
            final_glb=job.final_glb,
        )
        _append_result(manifest.result_path, payload)
        print(f"[CASE] success {job.case_name} -> {job.final_glb}")
        return True
    except Exception as exc:
        duration = time.time() - case_start
        traceback_text = traceback.format_exc()
        print(f"[CASE] fail {job.case_name}: {exc}")
        print(traceback_text, file=sys.stderr, end="")
        payload = {
            "case_name": job.case_name,
            "success": False,
            "gpu_id": manifest.gpu_id,
            "duration_seconds": duration,
            "final_glb": str(job.final_glb),
            "worker_log_path": str(manifest.worker_log_path),
            "case_log_path": str(job.log_path),
            "error": str(exc),
            "traceback": traceback_text,
        }
        _write_case_log(
            job.log_path,
            worker_log_path=manifest.worker_log_path,
            case_name=job.case_name,
            gpu_id=manifest.gpu_id,
            success=False,
            duration_seconds=duration,
            final_glb=job.final_glb,
            error=str(exc),
            traceback_text=traceback_text,
        )
        _append_result(manifest.result_path, payload)
        return False
    finally:
        gc.collect()
        release_cuda_memory()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Persistent composable batch worker that reuses one loaded pipeline across multiple cases."
    )
    parser.add_argument("--manifest", required=True, help="Path to compact worker manifest JSON.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    manifest = _load_manifest(_serialize_path(args.manifest))
    if not manifest.jobs:
        print(f"[WORKER] No jobs assigned to worker {manifest.worker_id} (cuda:{manifest.gpu_id})")
        return 0

    first_config = _load_case_config(manifest.jobs[0].config_path)
    expected_signature = _shared_pipeline_signature(first_config)
    pipeline = _load_shared_pipeline(first_config)

    success_count = 0
    for index, job in enumerate(manifest.jobs, start=1):
        print(
            f"[WORKER] ({index}/{len(manifest.jobs)}) "
            f"worker={manifest.worker_id} gpu=cuda:{manifest.gpu_id} case={job.case_name}"
        )
        if _run_job(job=job, manifest=manifest, pipeline=pipeline, expected_signature=expected_signature):
            success_count += 1

    print(
        f"[WORKER] done worker={manifest.worker_id} gpu=cuda:{manifest.gpu_id} "
        f"success={success_count}/{len(manifest.jobs)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
