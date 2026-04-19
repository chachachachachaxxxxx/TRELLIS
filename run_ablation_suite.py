#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SuiteJob:
    name: str
    base_config: str
    script: str
    overrides: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and optionally run an ablation suite.")
    parser.add_argument("--suite", required=True, help="Suite YAML path.")
    parser.add_argument(
        "--generated-config-dir",
        default="edit_configs/ablations/generated",
        help="Directory to place generated concrete configs.",
    )
    parser.add_argument("--only", default="", help="Optional comma-separated subset of job names.")
    parser.add_argument("--generate-only", action="store_true", help="Only generate concrete configs.")
    parser.add_argument("--dry-run", action="store_true", help="Append --dry-run to each launched job.")
    parser.add_argument("--gpus", default="1,2", help="Comma-separated GPU ids for parallel launch.")
    parser.add_argument("--logs-dir", default="outputs/ablation_runs/logs", help="Per-job log directory.")
    parser.add_argument("--poll-seconds", type=float, default=5.0, help="Polling interval for worker status.")
    return parser


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"YAML must deserialize to a mapping: {path}")
    return payload


def _set_dotted_key(payload: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor: dict[str, Any] = payload
    for key in parts[:-1]:
        child = cursor.get(key)
        if child is None:
            child = {}
            cursor[key] = child
        if not isinstance(child, dict):
            raise RuntimeError(f"Cannot descend into non-mapping key '{key}' while setting '{dotted_key}'")
        cursor = child
    cursor[parts[-1]] = value


def _parse_jobs(suite_payload: dict[str, Any], only: str) -> tuple[str, list[SuiteJob]]:
    suite_name = str(suite_payload.get("suite_name") or "ablation_suite")
    default_script = str(suite_payload.get("default_script") or "run_edit_experiment.py")
    raw_jobs = suite_payload.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise RuntimeError("Suite YAML requires a non-empty 'jobs' list.")

    requested = {item.strip() for item in only.split(",") if item.strip()}
    jobs: list[SuiteJob] = []
    for raw_job in raw_jobs:
        if not isinstance(raw_job, dict):
            raise RuntimeError("Each suite job must be a mapping.")
        name = str(raw_job.get("name") or "").strip()
        if not name:
            raise RuntimeError("Each suite job requires a non-empty name.")
        if requested and name not in requested:
            continue
        base_config = str(raw_job.get("base_config") or "").strip()
        if not base_config:
            raise RuntimeError(f"Suite job '{name}' is missing base_config.")
        script = str(raw_job.get("script") or default_script).strip()
        overrides = raw_job.get("overrides") or {}
        if not isinstance(overrides, dict):
            raise RuntimeError(f"Suite job '{name}' overrides must be a mapping.")
        jobs.append(
            SuiteJob(
                name=name,
                base_config=base_config,
                script=script,
                overrides=overrides,
            )
        )

    if requested:
        missing = requested - {job.name for job in jobs}
        if missing:
            raise RuntimeError(f"Unknown --only job names: {', '.join(sorted(missing))}")
    if not jobs:
        raise RuntimeError("No jobs selected from suite.")
    return suite_name, jobs


def _generate_job_config(repo_root: Path, suite_name: str, job: SuiteJob, out_dir: Path) -> Path:
    base_path = (repo_root / job.base_config).resolve()
    base_payload = _load_yaml(base_path)
    payload = copy.deepcopy(base_payload)
    for dotted_key, value in job.overrides.items():
        _set_dotted_key(payload, str(dotted_key), value)

    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        runtime = {}
        payload["runtime"] = runtime
    runtime.setdefault("case_name", job.name)
    runtime.setdefault("output_group", suite_name)

    out_path = out_dir / f"{job.name}.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    return out_path


def _launch_jobs(
    repo_root: Path,
    jobs: list[tuple[SuiteJob, Path]],
    *,
    gpus: list[str],
    logs_dir: Path,
    dry_run: bool,
    poll_seconds: float,
) -> int:
    pending = list(jobs)
    gpu_pool = list(gpus)
    running: list[dict[str, Any]] = []
    failures: list[tuple[str, int, Path]] = []

    while pending or running:
        while pending and gpu_pool:
            job, config_path = pending.pop(0)
            gpu_id = gpu_pool.pop(0)
            log_path = logs_dir / f"{job.name}.log"
            cmd = [sys.executable, job.script, "--config", str(config_path)]
            if dry_run:
                cmd.append("--dry-run")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            env.setdefault("PYTHONUNBUFFERED", "1")
            handle = open(log_path, "w", encoding="utf-8")
            process = subprocess.Popen(
                cmd,
                cwd=repo_root,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            running.append(
                {
                    "job": job,
                    "config_path": config_path,
                    "gpu_id": gpu_id,
                    "log_path": log_path,
                    "handle": handle,
                    "process": process,
                }
            )
            print(f"[LAUNCH] {job.name} on cuda:{gpu_id} -> {log_path}")

        if not running:
            continue

        time.sleep(max(poll_seconds, 0.1))
        still_running: list[dict[str, Any]] = []
        for item in running:
            returncode = item["process"].poll()
            if returncode is None:
                still_running.append(item)
                continue

            item["handle"].close()
            gpu_pool.append(item["gpu_id"])
            gpu_pool.sort(key=int)
            if returncode == 0:
                print(f"[DONE] {item['job'].name} on cuda:{item['gpu_id']}")
            else:
                print(f"[FAIL] {item['job'].name} on cuda:{item['gpu_id']} -> {item['log_path']}")
                failures.append((item["job"].name, returncode, item["log_path"]))
        running = still_running

    if failures:
        print("Failures:")
        for job_name, returncode, log_path in failures:
            print(f"  - {job_name}: exit={returncode}, log={log_path}")
        return 1
    print("All ablation jobs completed successfully.")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parent
    suite_path = (repo_root / args.suite).resolve()
    suite_payload = _load_yaml(suite_path)
    suite_name, jobs = _parse_jobs(suite_payload, args.only)

    generated_root = (repo_root / args.generated_config_dir / suite_name).resolve()
    logs_dir = (repo_root / args.logs_dir / suite_name).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    generated_jobs: list[tuple[SuiteJob, Path]] = []
    for job in jobs:
        config_path = _generate_job_config(repo_root, suite_name, job, generated_root)
        generated_jobs.append((job, config_path))

    print(f"Generated {len(generated_jobs)} configs under {generated_root}")
    for job, config_path in generated_jobs:
        print(f"  - {job.name}: {config_path}")

    if args.generate_only:
        return 0

    gpu_pool = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpu_pool:
        raise RuntimeError("No GPUs provided.")
    return _launch_jobs(
        repo_root,
        generated_jobs,
        gpus=gpu_pool,
        logs_dir=logs_dir,
        dry_run=args.dry_run,
        poll_seconds=args.poll_seconds,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)
