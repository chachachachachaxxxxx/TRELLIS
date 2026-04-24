from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from trellis_edit.benchmark_pages import (
    _bundle_relpath,
    _link_or_copy,
    _render_daily_run_pages,
    _summarize_statuses,
    _write_jsonl,
    build_daily_index,
    build_focus_index,
    build_run_id,
)
from trellis_edit.common import ensure_dir, write_json

from .base import CaseResult, RunResult


DEFAULT_ARTIFACT_TARGET_NAMES = {
    "source_image": "source_image.png",
    "edit_image": "edit_image.png",
    "mask_image": "mask_image.png",
    "edit_front_image": "edit_front_image.png",
    "source_model_glb": "source_model.glb",
    "target_model_glb": "target_model.glb",
    "source_voxelmesh_glb": "source_voxelmesh.glb",
    "source_voxelmesh_transform": "source_voxelmesh_transform.json",
    "mask_glb": "mask.glb",
    "edit_glb": "edit.glb",
    "ss_voxelmesh_glb": "ss_voxelmesh.glb",
    "ss_voxelmesh_transform": "ss_voxelmesh_transform.json",
    "ss_coords_ply": "ss_coords.ply",
    "ss_metadata": "ss_metadata.json",
    "input_views": "input_views",
    "eval_target_views": "eval_target_views",
    "eval_masks": "eval_masks",
    "images": "images",
    "videos": "videos",
    "seen_views": "seen_views",
    "novel_views": "novel_views",
    "geo_gray_views": "gray_views",
    "geo_normal_views": "normal_views",
    "geo_mask_views": "mask_views",
    "geo_case_metrics": "case_metrics.json",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_run_labels_from_pred_root(
    pred_root: Path,
    *,
    fallback_entrypoint: str,
    fallback_config_name: str | None = None,
) -> dict[str, Any]:
    manifest_path = pred_root / "manifest.json"
    raw_manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    return {
        "entrypoint_name": str(
            raw_manifest.get("entrypoint")
            or fallback_entrypoint
        ),
        "config_name": str(
            raw_manifest.get("config_name")
            or fallback_config_name
            or pred_root.name
        ),
        "group": raw_manifest.get("group"),
        "total_time_seconds": float(
            raw_manifest.get("total_time_seconds")
            or 0.0
        ),
        "raw_manifest": raw_manifest,
    }


def infer_run_labels_from_daily_history(
    daily_root: Path,
    pred_root: Path,
    *,
    preferred_groups: tuple[str, ...] = ("basic_baselines",),
    excluded_groups: tuple[str, ...] = ("geometry",),
) -> dict[str, Any] | None:
    resolved_pred_root = pred_root.expanduser().resolve()
    if not daily_root.is_dir():
        return None

    candidates: list[tuple[int, str, dict[str, Any]]] = []
    for run_root in sorted(daily_root.iterdir()):
        manifest_path = run_root / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest_payload = _read_json(manifest_path)
        manifest_pred_root = str(manifest_payload.get("pred_root") or "").strip()
        if not manifest_pred_root:
            continue
        try:
            resolved_manifest_pred_root = Path(manifest_pred_root).expanduser().resolve()
        except OSError:
            continue
        if resolved_manifest_pred_root != resolved_pred_root:
            continue

        group_name = str(manifest_payload.get("group") or "").strip()
        if group_name in excluded_groups:
            continue

        if group_name in preferred_groups:
            priority = preferred_groups.index(group_name)
        else:
            priority = len(preferred_groups)
        candidates.append((priority, run_root.name, manifest_payload))

    if not candidates:
        return None

    _, _, manifest_payload = sorted(candidates, key=lambda item: (item[0], item[1]))[0]
    entrypoint_name = str(manifest_payload.get("entrypoint") or "").strip()
    config_name = str(manifest_payload.get("config_name") or "").strip()
    if not entrypoint_name and not config_name:
        return None
    return {
        "entrypoint_name": entrypoint_name or None,
        "config_name": config_name or None,
        "group": manifest_payload.get("group"),
        "total_time_seconds": float(manifest_payload.get("total_time_seconds") or 0.0),
        "raw_manifest": manifest_payload,
    }


def _artifact_target_name(
    key: str,
    source_path: Path,
    artifact_name_overrides: Mapping[str, str] | None = None,
) -> str:
    if artifact_name_overrides and key in artifact_name_overrides:
        return str(artifact_name_overrides[key])
    return DEFAULT_ARTIFACT_TARGET_NAMES.get(key, source_path.name)


def _case_json_path(root: Path, case_result: CaseResult) -> Path:
    tokens = case_result.identity.resolved_path_tokens()
    stem = tokens[-1] if tokens else case_result.identity.case_id.replace("/", "__")
    return root.joinpath(*tokens[:-1], f"{stem}.json")


def _case_page_path(root: Path, case_result: CaseResult) -> Path:
    tokens = case_result.identity.resolved_path_tokens()
    stem = tokens[-1] if tokens else case_result.identity.case_id.replace("/", "__")
    return root.joinpath(*tokens[:-1], f"{stem}.html")


def materialize_case_artifacts(
    *,
    bundle_root: Path,
    case_result: CaseResult,
    artifact_name_overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    artifact_dir = ensure_dir(
        bundle_root / "artifacts" / Path(*case_result.identity.resolved_path_tokens())
    )
    relpaths: dict[str, str] = {}
    for key, value in (case_result.artifacts or {}).items():
        if value in (None, ""):
            continue
        source_path = Path(value)
        target_path = artifact_dir / _artifact_target_name(
            key,
            source_path,
            artifact_name_overrides=artifact_name_overrides,
        )
        if _link_or_copy(source_path, target_path):
            relpaths[key] = _bundle_relpath(bundle_root, target_path)
    return relpaths


def _case_payload(
    *,
    case_result: CaseResult,
    artifact_relpaths: dict[str, str],
) -> dict[str, Any]:
    payload = {
        "case_id": case_result.identity.case_id,
        "dataset": case_result.identity.dataset,
        "object_name": case_result.identity.object_name,
        "prompt_id": case_result.identity.prompt_id,
        "sample_name": case_result.identity.sample_name,
        "display_name": case_result.identity.display_name,
        "status": case_result.status,
        "prompt_text": case_result.prompt_text,
        "metrics": dict(case_result.metrics),
        "artifacts": artifact_relpaths,
    }
    if case_result.task_meta:
        payload.update(case_result.task_meta)
    return payload


def _case_record(
    *,
    bundle_root: Path,
    case_result: CaseResult,
    case_json_path: Path,
    page_path: Path,
) -> dict[str, Any]:
    return {
        "case_id": case_result.identity.case_id,
        "dataset": case_result.identity.dataset,
        "object_name": case_result.identity.object_name,
        "prompt_id": case_result.identity.prompt_id,
        "sample_name": case_result.identity.sample_name,
        "display_name": case_result.identity.display_name,
        "status": case_result.status,
        "metrics": dict(case_result.metrics),
        "detail_path": _bundle_relpath(bundle_root, case_json_path),
        "page_path": _bundle_relpath(bundle_root, page_path),
        "artifact_dir": _bundle_relpath(
            bundle_root,
            bundle_root / "artifacts" / Path(*case_result.identity.resolved_path_tokens()),
        ),
    }


def _write_bundle_run_artifacts(
    *,
    bundle_root: Path,
    run_result: RunResult,
    bundle_artifact_files: Mapping[str, str] | None = None,
) -> None:
    if not bundle_artifact_files:
        return
    for artifact_key, filename in bundle_artifact_files.items():
        if artifact_key not in run_result.run_artifacts:
            continue
        write_json(bundle_root / filename, run_result.run_artifacts[artifact_key])


def create_daily_bundle_from_run_result(
    *,
    daily_root: Path,
    pred_root: Path,
    entrypoint_name: str,
    config_name: str,
    run_group: str | None,
    total_time_seconds: float,
    run_result: RunResult,
    manifest_extra: Mapping[str, Any] | None = None,
    artifact_name_overrides: Mapping[str, str] | None = None,
    bundle_artifact_files: Mapping[str, str] | None = None,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = build_run_id(entrypoint_name, config_name, timestamp)
    bundle_root = (daily_root / run_id).resolve()
    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    ensure_dir(bundle_root)

    case_records: list[dict[str, Any]] = []
    cases_dir = ensure_dir(bundle_root / "cases")
    pages_dir = ensure_dir(bundle_root / "pages")
    for case_result in run_result.case_results:
        artifact_relpaths = materialize_case_artifacts(
            bundle_root=bundle_root,
            case_result=case_result,
            artifact_name_overrides=artifact_name_overrides,
        )
        case_payload = _case_payload(
            case_result=case_result,
            artifact_relpaths=artifact_relpaths,
        )
        case_json_path = _case_json_path(cases_dir, case_result)
        ensure_dir(case_json_path.parent)
        write_json(case_json_path, case_payload)

        page_path = _case_page_path(pages_dir, case_result)
        ensure_dir(page_path.parent)
        case_records.append(
            _case_record(
                bundle_root=bundle_root,
                case_result=case_result,
                case_json_path=case_json_path,
                page_path=page_path,
            )
        )

    summary_payload = {
        "totals": _summarize_statuses(case_records),
        "metrics": dict(run_result.summary_metrics),
    }
    manifest_payload = {
        "run_id": run_id,
        "entrypoint": entrypoint_name,
        "config_name": config_name,
        "group": run_group,
        "created_at": datetime.now().astimezone().isoformat(),
        "task_name": run_result.task_name,
        "metrics": list(run_result.requested_metrics),
        "pred_root": str(pred_root),
        "total_time_seconds": float(total_time_seconds),
    }
    if run_result.manifest:
        manifest_payload.update(run_result.manifest)
    if manifest_extra:
        manifest_payload.update(dict(manifest_extra))

    write_json(bundle_root / "manifest.json", manifest_payload)
    write_json(bundle_root / "summary.json", summary_payload)
    _write_jsonl(bundle_root / "cases.jsonl", case_records)
    _write_bundle_run_artifacts(
        bundle_root=bundle_root,
        run_result=run_result,
        bundle_artifact_files=bundle_artifact_files,
    )
    _render_daily_run_pages(
        bundle_root=bundle_root,
        manifest_payload=manifest_payload,
        summary_payload=summary_payload,
        case_records=case_records,
    )
    return bundle_root


def update_daily_bundle_from_run_result(
    *,
    run_root: Path,
    run_result: RunResult,
    manifest_extra: Mapping[str, Any] | None = None,
    bundle_artifact_files: Mapping[str, str] | None = None,
) -> None:
    summary_path = run_root / "summary.json"
    manifest_path = run_root / "manifest.json"
    cases_jsonl_path = run_root / "cases.jsonl"

    summary_payload = _read_json(summary_path)
    summary_payload.setdefault("metrics", {}).update(run_result.summary_metrics)
    write_json(summary_path, summary_payload)

    manifest_payload = _read_json(manifest_path)
    existing_metrics = list(manifest_payload.get("metrics") or [])
    for metric_name in run_result.requested_metrics:
        if metric_name not in existing_metrics:
            existing_metrics.append(metric_name)
    manifest_payload["metrics"] = existing_metrics
    manifest_payload["task_name"] = run_result.task_name
    if run_result.manifest:
        manifest_payload.update(run_result.manifest)
    if manifest_extra:
        manifest_payload.update(dict(manifest_extra))
    write_json(manifest_path, manifest_payload)

    case_results_by_id = {
        case_result.identity.case_id: case_result
        for case_result in run_result.case_results
    }

    case_records = []
    for line in cases_jsonl_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        case_result = case_results_by_id.get(str(record.get("case_id") or ""))
        if case_result is not None:
            metrics = dict(record.get("metrics") or {})
            metrics.update(case_result.metrics)
            record["metrics"] = metrics
            record["status"] = case_result.status
            if case_result.identity.sample_name:
                record["sample_name"] = case_result.identity.sample_name
            if case_result.identity.display_name:
                record["display_name"] = case_result.identity.display_name
        case_records.append(record)
    cases_jsonl_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in case_records) + "\n",
        encoding="utf-8",
    )

    for record in case_records:
        detail_path = run_root / str(record.get("detail_path") or "")
        if not detail_path.is_file():
            continue
        detail_payload = _read_json(detail_path)
        case_result = case_results_by_id.get(str(record.get("case_id") or ""))
        if case_result is None:
            continue
        detail_payload["status"] = case_result.status
        metrics = dict(detail_payload.get("metrics") or {})
        metrics.update(case_result.metrics)
        detail_payload["metrics"] = metrics
        if case_result.identity.sample_name:
            detail_payload["sample_name"] = case_result.identity.sample_name
        if case_result.identity.display_name:
            detail_payload["display_name"] = case_result.identity.display_name
        if case_result.task_meta:
            detail_payload.update(case_result.task_meta)
        write_json(detail_path, detail_payload)

    _write_bundle_run_artifacts(
        bundle_root=run_root,
        run_result=run_result,
        bundle_artifact_files=bundle_artifact_files,
    )
    _render_daily_run_pages(
        bundle_root=run_root,
        manifest_payload=_read_json(manifest_path),
        summary_payload=_read_json(summary_path),
        case_records=case_records,
    )


def rebuild_benchmark_indexes(
    *,
    benchmark_root: Path,
    suppress_focus_errors: bool = False,
) -> None:
    build_daily_index(benchmark_root)
    if suppress_focus_errors:
        try:
            build_focus_index(benchmark_root)
        except Exception as exc:
            print(f"[INFO] Skip focus index rebuild for current task layout: {exc}")
        return
    build_focus_index(benchmark_root)
