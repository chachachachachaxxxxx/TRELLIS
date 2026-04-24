from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .base import CaseIdentity, CaseResult, RunResult


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def select_case_shard(
    cases: Sequence[Any],
    *,
    shard_count: int,
    shard_index: int,
) -> list[Any]:
    if shard_count <= 1:
        return list(cases)
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"Invalid shard index {shard_index} for shard count {shard_count}.")
    return [case for idx, case in enumerate(cases) if idx % shard_count == shard_index]


def shard_state_dir(pred_root: Path, *, state_dir_name: str) -> Path:
    return pred_root / state_dir_name


def shard_result_path(
    pred_root: Path,
    *,
    shard_count: int,
    shard_index: int,
    state_dir_name: str,
    result_prefix: str,
    merged_filename: str,
) -> Path:
    if shard_count <= 1:
        return pred_root / merged_filename
    return (
        shard_state_dir(pred_root, state_dir_name=state_dir_name)
        / f"{result_prefix}.shard_{shard_index:02d}_of_{shard_count:02d}.json"
    )


def serialize_case_result(case_result: CaseResult) -> dict[str, Any]:
    artifacts = {
        key: str(value)
        for key, value in (case_result.artifacts or {}).items()
        if value not in (None, "")
    }
    return {
        "identity": {
            "case_id": case_result.identity.case_id,
            "dataset": case_result.identity.dataset,
            "object_name": case_result.identity.object_name,
            "prompt_id": case_result.identity.prompt_id,
            "sample_name": case_result.identity.sample_name,
            "display_name": case_result.identity.display_name,
            "path_tokens": list(case_result.identity.path_tokens),
        },
        "status": case_result.status,
        "metrics": dict(case_result.metrics),
        "artifacts": artifacts,
        "prompt_text": case_result.prompt_text,
        "task_meta": case_result.task_meta,
    }


def deserialize_case_result(payload: Mapping[str, Any]) -> CaseResult:
    identity_payload = dict(payload.get("identity") or {})
    identity = CaseIdentity(
        case_id=str(identity_payload.get("case_id") or ""),
        dataset=identity_payload.get("dataset"),
        object_name=identity_payload.get("object_name"),
        prompt_id=identity_payload.get("prompt_id"),
        sample_name=identity_payload.get("sample_name"),
        display_name=identity_payload.get("display_name"),
        path_tokens=tuple(identity_payload.get("path_tokens") or ()),
    )
    return CaseResult(
        identity=identity,
        status=str(payload.get("status") or "failed"),
        metrics=dict(payload.get("metrics") or {}),
        artifacts=dict(payload.get("artifacts") or {}),
        prompt_text=payload.get("prompt_text"),
        task_meta=payload.get("task_meta"),
    )


def serialize_run_result(run_result: RunResult) -> dict[str, Any]:
    return {
        "task_name": run_result.task_name,
        "requested_metrics": list(run_result.requested_metrics),
        "summary_metrics": dict(run_result.summary_metrics),
        "case_results": [serialize_case_result(case_result) for case_result in run_result.case_results],
        "manifest": dict(run_result.manifest),
        "run_artifacts": dict(run_result.run_artifacts),
    }


def deserialize_run_result(payload: Mapping[str, Any]) -> RunResult:
    return RunResult(
        task_name=str(payload.get("task_name") or "benchmark"),
        requested_metrics=list(payload.get("requested_metrics") or []),
        summary_metrics=dict(payload.get("summary_metrics") or {}),
        case_results=[
            deserialize_case_result(case_payload)
            for case_payload in (payload.get("case_results") or [])
        ],
        manifest=dict(payload.get("manifest") or {}),
        run_artifacts=dict(payload.get("run_artifacts") or {}),
    )


def load_run_result(path: Path) -> RunResult:
    return deserialize_run_result(read_json(path))


def merge_sharded_run_results(
    shard_runs: Sequence[RunResult],
    *,
    summary_metrics: Mapping[str, Any],
    manifest: Mapping[str, Any] | None = None,
    run_artifacts: Mapping[str, Any] | None = None,
    case_sort_key: Callable[[CaseResult], Any] | None = None,
    task_name: str | None = None,
    requested_metrics: Sequence[str] | None = None,
) -> RunResult:
    merged_case_results: list[CaseResult] = []
    for shard_run in shard_runs:
        merged_case_results.extend(shard_run.case_results)
    if case_sort_key is not None:
        merged_case_results.sort(key=case_sort_key)

    resolved_task_name = task_name or (
        shard_runs[0].task_name if shard_runs else "benchmark"
    )
    resolved_requested_metrics = list(requested_metrics) if requested_metrics is not None else (
        list(shard_runs[0].requested_metrics) if shard_runs else []
    )
    resolved_manifest = dict(manifest) if manifest is not None else (
        dict(shard_runs[0].manifest) if shard_runs else {}
    )
    resolved_run_artifacts = dict(run_artifacts) if run_artifacts is not None else {}
    return RunResult(
        task_name=resolved_task_name,
        requested_metrics=resolved_requested_metrics,
        summary_metrics=dict(summary_metrics),
        case_results=merged_case_results,
        manifest=resolved_manifest,
        run_artifacts=resolved_run_artifacts,
    )


__all__ = [
    "deserialize_case_result",
    "deserialize_run_result",
    "load_run_result",
    "merge_sharded_run_results",
    "read_json",
    "select_case_shard",
    "serialize_case_result",
    "serialize_run_result",
    "shard_result_path",
    "shard_state_dir",
]
