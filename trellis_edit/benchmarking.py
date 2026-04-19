from __future__ import annotations

import html
import json
import math
import os
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from trellis_edit.common import ensure_dir, write_json


DEFAULT_BENCHMARK_ROOT = Path("/cache/wangxinxing/data/trellis_edit_benchmark")
RUN_LEVEL_METRICS = {"fid", "fid_buffered", "fvd"}
TEXT_ALIGNMENT_VIEW_IDS = {"0000", "0001", "0007", "0008", "0009", "0015"}
DEFAULT_RUN_GROUP_LABEL = "ungrouped"
METRIC_FAMILY_ORDER = (
    "Legacy Metrics",
    "Multi-View Metrics",
)
METRIC_FAMILY_META = {
    "Legacy Metrics": {"description": "Original Edit3D-Bench metrics kept for backward comparison."},
    "Multi-View Metrics": {"description": "New metrics tailored for multi-view editing fidelity and stability."},
}
METRIC_GROUP_ORDER_BY_FAMILY = {
    "Legacy Metrics": (
        "Instruction Following",
        "Image Quality",
        "Geometry",
        "Distribution",
    ),
    "Multi-View Metrics": (
        "Seen-View Fidelity",
        "Novel-View Fidelity",
        "Multi-View Geometry",
        "Stability",
    ),
}
PRIMARY_METRIC_NAMES = (
    "lpips_in_novel",
    "ssim_in_novel",
    "dino_if_novel",
    "dino_if_max",
    "dino_if_mean",
    "clip_t",
    "psnr",
)
LEADING_METRIC_NAMES = (
    "lpips_in_novel",
    "ssim_in_novel",
    "dino_if_novel",
    "dino_if_max",
    "dino_if_mean",
    "clip_t",
)
METRIC_GROUP_META = {
    "Image Quality": {"description": "Rendered-view fidelity against the reference renders."},
    "Instruction Following": {"description": "Whether the edited output follows the edit instruction."},
    "Geometry": {"description": "3D shape consistency and edit-region geometry quality."},
    "Distribution": {"description": "Run-level distribution metrics across all outputs."},
    "Seen-View Fidelity": {"description": "How well the result matches the provided edited views."},
    "Novel-View Fidelity": {"description": "How well the edited result generalizes to held-out target views."},
    "Multi-View Geometry": {"description": "3D agreement with the aligned multi-view target model."},
    "Stability": {"description": "How sensitive the result is across different random seeds."},
}
METRIC_META = {
    "dino_if_max": {"label": "DINO-IF Max", "family": "Legacy Metrics", "group": "Instruction Following", "order": 0},
    "dino_if_mean": {"label": "DINO-IF Mean", "family": "Legacy Metrics", "group": "Instruction Following", "order": 1},
    "clip_t": {"label": "CLIP-T", "family": "Legacy Metrics", "group": "Instruction Following", "order": 2},
    "psnr": {"label": "PSNR", "family": "Legacy Metrics", "group": "Image Quality", "order": 3},
    "ssim": {"label": "SSIM", "family": "Legacy Metrics", "group": "Image Quality", "order": 4},
    "lpips": {"label": "LPIPS", "family": "Legacy Metrics", "group": "Image Quality", "order": 5},
    "ssim_buffered": {"label": "SSIM Buffered", "family": "Legacy Metrics", "group": "Image Quality", "order": 6},
    "lpips_buffered": {"label": "LPIPS Buffered", "family": "Legacy Metrics", "group": "Image Quality", "order": 7},
    "chamfer": {"label": "Chamfer", "family": "Legacy Metrics", "group": "Geometry", "order": 8},
    "fid": {"label": "FID", "family": "Legacy Metrics", "group": "Distribution", "order": 9},
    "fid_buffered": {"label": "FID Buffered", "family": "Legacy Metrics", "group": "Distribution", "order": 10},
    "fvd": {"label": "FVD", "family": "Legacy Metrics", "group": "Distribution", "order": 11},
    "lpips_in_seen": {"label": "LPIPS In Seen", "family": "Multi-View Metrics", "group": "Seen-View Fidelity", "order": 0},
    "ssim_in_seen": {"label": "SSIM In Seen", "family": "Multi-View Metrics", "group": "Seen-View Fidelity", "order": 1},
    "dino_if_seen": {"label": "DINO-IF Seen", "family": "Multi-View Metrics", "group": "Seen-View Fidelity", "order": 2},
    "lpips_in_novel": {"label": "LPIPS In Novel", "family": "Multi-View Metrics", "group": "Novel-View Fidelity", "order": 3},
    "ssim_in_novel": {"label": "SSIM In Novel", "family": "Multi-View Metrics", "group": "Novel-View Fidelity", "order": 4},
    "dino_if_novel": {"label": "DINO-IF Novel", "family": "Multi-View Metrics", "group": "Novel-View Fidelity", "order": 5},
    "chamfer_target": {"label": "Chamfer Target", "family": "Multi-View Metrics", "group": "Multi-View Geometry", "order": 6},
    "cross_seed_novel_lpips": {"label": "Cross-Seed Novel LPIPS", "family": "Multi-View Metrics", "group": "Stability", "order": 7},
}


def build_run_id(entrypoint_name: str, config_name: str, timestamp: str) -> str:
    return "_".join((_slug(entrypoint_name), _slug(config_name), timestamp))


def create_daily_benchmark_bundle(
    *,
    benchmark_root: Path,
    entrypoint_name: str,
    config_name: str,
    run_group: str | None,
    gt_root: Path,
    pred_root: Path,
    cases: list[tuple[str, str, int]],
    requested_metrics: list[str],
    summary_results: dict[str, Any],
    total_time_seconds: float,
    skip_benchmark_render: bool,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = build_run_id(entrypoint_name, config_name, timestamp)
    bundle_root = (benchmark_root / "daily" / run_id).resolve()
    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    ensure_dir(bundle_root)

    metadata = _load_metadata(gt_root)
    prompt_text_map = _build_prompt_text_map(metadata)
    detailed_results = _load_json(pred_root / "evaluation_output" / "detailed_results.json")
    per_case_metrics = _build_per_case_metrics(
        gt_root=gt_root,
        pred_root=pred_root,
        metadata=metadata,
        cases=cases,
        requested_metrics=requested_metrics,
        detailed_results=detailed_results,
    )

    case_records: list[dict[str, Any]] = []
    cases_dir = ensure_dir(bundle_root / "cases")
    pages_dir = ensure_dir(bundle_root / "pages")
    for dataset, object_name, prompt_id in cases:
        prompt_key = f"prompt_{prompt_id}"
        case_id = _case_id(dataset, object_name, prompt_id)
        artifact_relpaths = _materialize_case_artifacts(
            bundle_root=bundle_root,
            gt_root=gt_root,
            pred_root=pred_root,
            dataset=dataset,
            object_name=object_name,
            prompt_key=prompt_key,
        )

        metrics = {
            metric: per_case_metrics.get(case_id, {}).get(metric)
            for metric in _requested_case_metrics(requested_metrics)
        }
        has_edit = "edit_glb" in artifact_relpaths
        has_render = "images" in artifact_relpaths or "videos" in artifact_relpaths
        if not has_edit:
            status = "failed"
        elif skip_benchmark_render or has_render:
            status = "ok"
        else:
            status = "partial"

        case_payload = {
            "case_id": case_id,
            "dataset": dataset,
            "object_name": object_name,
            "prompt_id": prompt_id,
            "status": status,
            "prompt_text": prompt_text_map.get((dataset, object_name, prompt_id)),
            "metrics": metrics,
            "artifacts": artifact_relpaths,
        }
        case_json_path = cases_dir / dataset / object_name / f"prompt_{prompt_id}.json"
        write_json(case_json_path, case_payload)

        page_path = pages_dir / dataset / object_name / f"prompt_{prompt_id}.html"
        ensure_dir(page_path.parent)

        case_records.append(
            {
                "case_id": case_id,
                "dataset": dataset,
                "object_name": object_name,
                "prompt_id": prompt_id,
                "status": status,
                "metrics": metrics,
                "detail_path": _bundle_relpath(bundle_root, case_json_path),
                "page_path": _bundle_relpath(bundle_root, page_path),
                "artifact_dir": f"artifacts/{dataset}/{object_name}/{prompt_key}",
            }
        )

    summary_payload = {
        "totals": _summarize_statuses(case_records),
        "metrics": dict(summary_results.get("results", {})),
    }
    manifest_payload = {
        "run_id": run_id,
        "entrypoint": entrypoint_name,
        "config_name": config_name,
        "group": run_group,
        "created_at": datetime.now().astimezone().isoformat(),
        "metrics": list(requested_metrics),
        "pred_root": str(pred_root),
        "total_time_seconds": total_time_seconds,
    }

    write_json(bundle_root / "manifest.json", manifest_payload)
    write_json(bundle_root / "summary.json", summary_payload)
    _write_jsonl(bundle_root / "cases.jsonl", case_records)
    _render_daily_run_pages(
        bundle_root=bundle_root,
        manifest_payload=manifest_payload,
        summary_payload=summary_payload,
        case_records=case_records,
    )
    return bundle_root


def build_daily_index(benchmark_root: Path) -> Path | None:
    daily_root = (benchmark_root / "daily").resolve()
    if not daily_root.is_dir():
        return None

    runs = _load_daily_runs(daily_root)
    if not runs:
        return None

    for run in runs:
        _render_daily_run_pages(
            bundle_root=run["root"],
            manifest_payload=run["manifest"],
            summary_payload=run["summary"],
            case_records=_load_jsonl(run["root"] / "cases.jsonl"),
        )

    group_names = _ordered_run_groups(runs)
    groups_root = daily_root / "groups"
    if groups_root.exists():
        shutil.rmtree(groups_root)
    ensure_dir(groups_root)

    grouped_runs = [
        (group_name, [run for run in runs if run["group_name"] == group_name])
        for group_name in group_names
    ]
    group_pages: list[dict[str, Any]] = []
    for group_name, group_runs in grouped_runs:
        page_path = groups_root / f"{_slug(group_name)}.html"
        group_pages.append(
            {
                "name": group_name,
                "count": len(group_runs),
                "path": page_path.relative_to(daily_root).as_posix(),
            }
        )

    for group_name, group_runs in grouped_runs:
        page_path = groups_root / f"{_slug(group_name)}.html"
        page_path.write_text(
            _render_daily_collection_page(
                daily_root=daily_root,
                page_path=page_path,
                title=f"Benchmark Daily / {group_name}",
                heading=f"Daily Group: {group_name}",
                subtitle=f"{len(group_runs)} run(s)",
                grouped_runs=[(group_name, group_runs)],
                group_pages=group_pages,
                current_group=group_name,
            ),
            encoding="utf-8",
        )

    index_path = daily_root / "index.html"
    index_path.write_text(
        _render_daily_collection_page(
            daily_root=daily_root,
            page_path=index_path,
            title="Benchmark Daily",
            heading="Benchmark Daily",
            subtitle=f"{len(runs)} run(s) across {len(group_names)} group(s)",
            grouped_runs=grouped_runs,
            group_pages=group_pages,
            current_group=None,
        ),
        encoding="utf-8",
    )
    write_json(
        daily_root / "collection_manifest.json",
        {
            "groups": group_pages,
            "runs": [
                {
                    "id": run["id"],
                    "label": run["label"],
                    "group": run["group_name"],
                    "path": run["path"],
                }
                for run in runs
            ],
        },
    )
    return index_path


def _render_daily_run_pages(
    *,
    bundle_root: Path,
    manifest_payload: dict[str, Any],
    summary_payload: dict[str, Any],
    case_records: list[dict[str, Any]],
) -> None:
    run_label = str(manifest_payload.get("config_name") or manifest_payload.get("run_id") or bundle_root.name)
    run_id = str(manifest_payload.get("run_id") or bundle_root.name)

    for case_record in case_records:
        page_path = _case_record_page_path(bundle_root, case_record)
        detail_relpath = str(case_record.get("detail_path") or "").strip()
        case_payload = _load_json(bundle_root / detail_relpath) if detail_relpath else {}
        if not case_payload:
            case_payload = {
                "case_id": case_record.get("case_id"),
                "dataset": case_record.get("dataset"),
                "object_name": case_record.get("object_name"),
                "prompt_id": case_record.get("prompt_id"),
                "status": case_record.get("status"),
                "prompt_text": "",
                "metrics": case_record.get("metrics") or {},
                "artifacts": {},
            }
        page_path.write_text(
            _render_case_page(
                case_payload=case_payload,
                bundle_root=bundle_root,
                page_path=page_path,
                navigation_payload=_build_case_navigation_payload(
                    bundle_root=bundle_root,
                    page_path=page_path,
                    case_records=case_records,
                    current_case_record=case_record,
                ),
                run_label=run_label,
                run_id=run_id,
            ),
            encoding="utf-8",
        )

    (bundle_root / "index.html").write_text(
        _render_run_index(
            bundle_root=bundle_root,
            manifest_payload=manifest_payload,
            summary_payload=summary_payload,
            case_records=case_records,
        ),
        encoding="utf-8",
    )


def _case_record_page_path(bundle_root: Path, case_record: dict[str, Any]) -> Path:
    page_relpath = str(case_record.get("page_path") or "").strip()
    if page_relpath:
        return bundle_root / page_relpath
    dataset = str(case_record.get("dataset") or "").strip()
    object_name = str(case_record.get("object_name") or "").strip()
    prompt_id = int(case_record.get("prompt_id") or 0)
    return bundle_root / "pages" / dataset / object_name / f"prompt_{prompt_id}.html"


def _build_case_navigation_payload(
    *,
    bundle_root: Path,
    page_path: Path,
    case_records: list[dict[str, Any]],
    current_case_record: dict[str, Any],
) -> dict[str, Any]:
    dataset_map: dict[str, list[dict[str, Any]]] = {}
    for record in case_records:
        dataset = str(record.get("dataset") or "").strip()
        if not dataset:
            continue
        dataset_map.setdefault(dataset, []).append(record)

    current_case_id = str(current_case_record.get("case_id") or "")
    current_dataset = str(current_case_record.get("dataset") or "").strip()
    current_dataset_records = dataset_map.get(current_dataset) or [current_case_record]
    current_index = next(
        (
            index
            for index, record in enumerate(current_dataset_records)
            if str(record.get("case_id") or "") == current_case_id
        ),
        0,
    )

    dataset_entries = []
    for dataset, records in dataset_map.items():
        first_page_path = _case_record_page_path(bundle_root, records[0])
        dataset_entries.append(
            {
                "dataset": dataset,
                "count": len(records),
                "href": _relative_href(page_path, first_page_path),
                "selected": dataset == current_dataset,
            }
        )

    case_entries = []
    for index, record in enumerate(current_dataset_records, start=1):
        prompt_id = int(record.get("prompt_id") or 0)
        object_name = str(record.get("object_name") or "").strip()
        case_entries.append(
            {
                "index": index,
                "case_id": str(record.get("case_id") or ""),
                "object_name": object_name,
                "prompt_id": prompt_id,
                "prompt_label": f"prompt_{prompt_id}",
                "status": str(record.get("status") or ""),
                "href": _relative_href(page_path, _case_record_page_path(bundle_root, record)),
                "search_text": " ".join(
                    (
                        str(record.get("case_id") or ""),
                        object_name,
                        f"prompt_{prompt_id}",
                    )
                ).lower(),
            }
        )

    prev_href = None
    next_href = None
    if len(current_dataset_records) > 1:
        prev_href = _relative_href(
            page_path,
            _case_record_page_path(bundle_root, current_dataset_records[(current_index - 1) % len(current_dataset_records)]),
        )
        next_href = _relative_href(
            page_path,
            _case_record_page_path(bundle_root, current_dataset_records[(current_index + 1) % len(current_dataset_records)]),
        )

    return {
        "run_href": _relative_href(page_path, bundle_root / "index.html"),
        "current_dataset": current_dataset,
        "current_case_id": current_case_id,
        "current_index": current_index + 1,
        "total_cases": len(current_dataset_records),
        "prev_href": prev_href,
        "next_href": next_href,
        "datasets": dataset_entries,
        "cases": case_entries,
    }


def build_focus_index(benchmark_root: Path) -> Path | None:
    focus_root = (benchmark_root / "focus").resolve()
    manifest_path = focus_root / "collection_manifest.json"
    if not manifest_path.is_file():
        return None

    payload = _load_json(manifest_path)
    baseline_run_id = payload.get("baseline_run_id")
    runs = []
    for item in payload.get("runs", []):
        rel_path = str(item.get("path") or item.get("id") or "").strip()
        if not rel_path:
            continue
        run_root = (focus_root / rel_path).resolve()
        run_manifest = _load_json(run_root / "manifest.json")
        run_summary = _load_json(run_root / "summary.json")
        case_records = _load_jsonl(run_root / "cases.jsonl")
        if not run_manifest or not run_summary:
            continue
        runs.append(
            {
                "id": item.get("id") or run_manifest.get("run_id") or rel_path,
                "label": item.get("label") or run_manifest.get("config_name") or rel_path,
                "path": rel_path,
                "group": item.get("group") or run_manifest.get("group"),
                "manifest": run_manifest,
                "summary": run_summary,
                "root": run_root,
                "case_records": case_records,
                "case_map": {
                    str(case_record.get("case_id")): case_record
                    for case_record in case_records
                    if case_record.get("case_id")
                },
            }
        )

    if not runs:
        return None

    case_pages = _build_focus_case_pages(
        benchmark_root=benchmark_root,
        focus_root=focus_root,
        runs=runs,
    )
    output_path = focus_root / "index.html"
    rendered = _render_focus_index(
        runs,
        case_pages=case_pages,
        baseline_run_id=baseline_run_id,
    )
    output_path.write_text(rendered, encoding="utf-8")
    return output_path


def _slug(value: str) -> str:
    text = value.replace("/", "_").replace(" ", "_").strip("_")
    return "".join(ch for ch in text if ch.isalnum() or ch in {"_", "-", "."}) or "run"


def _normalize_run_group(value: Any) -> str:
    if value is None:
        return DEFAULT_RUN_GROUP_LABEL
    text = str(value).strip()
    return text or DEFAULT_RUN_GROUP_LABEL


def _case_id(dataset: str, object_name: str, prompt_id: int) -> str:
    return f"{dataset}/{object_name}/prompt_{prompt_id}"


def _requested_case_metrics(requested_metrics: list[str]) -> list[str]:
    return [metric for metric in requested_metrics if metric not in RUN_LEVEL_METRICS]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _resolve_daily_run_group(
    run_manifest: dict[str, Any],
    run_summary: dict[str, Any],
) -> str:
    _ = run_summary
    raw_group = run_manifest.get("group")
    if isinstance(raw_group, str) and raw_group.strip():
        return _normalize_run_group(raw_group)
    return DEFAULT_RUN_GROUP_LABEL


def _load_daily_runs(daily_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for run_root in daily_root.iterdir():
        if not run_root.is_dir() or run_root.name == "groups":
            continue
        run_manifest = _load_json(run_root / "manifest.json")
        run_summary = _load_json(run_root / "summary.json")
        if not run_manifest or not run_summary:
            continue
        runs.append(
            {
                "id": str(run_manifest.get("run_id") or run_root.name),
                "label": str(run_manifest.get("config_name") or run_root.name),
                "path": run_root.name,
                "root": run_root,
                "group_name": _resolve_daily_run_group(run_manifest, run_summary),
                "manifest": run_manifest,
                "summary": run_summary,
                "created_at": str(run_manifest.get("created_at") or ""),
            }
        )
    runs.sort(
        key=lambda item: (
            str(item.get("created_at") or ""),
            str(item.get("id") or ""),
        ),
        reverse=True,
    )
    return runs


def _ordered_run_groups(runs: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {str(run.get("group_name") or DEFAULT_RUN_GROUP_LABEL) for run in runs},
        key=lambda value: (
            value == DEFAULT_RUN_GROUP_LABEL,
            value.lower(),
        ),
    )


def _load_metadata(gt_root: Path) -> list[dict[str, Any]]:
    metadata_path = gt_root / "metadata.json"
    if not metadata_path.is_file():
        return []
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _build_prompt_text_map(metadata: list[dict[str, Any]]) -> dict[tuple[str, str, int], str]:
    prompt_text_map: dict[tuple[str, str, int], str] = {}
    for item in metadata:
        dataset = item.get("dataset")
        object_name = item.get("source_model")
        if not dataset or not object_name:
            continue
        for prompt_id in (1, 2, 3):
            prompt_key = f"prompt_{prompt_id}"
            prompt_text = item.get(prompt_key)
            if isinstance(prompt_text, str) and prompt_text.strip():
                prompt_text_map[(dataset, object_name, prompt_id)] = prompt_text.strip()
    return prompt_text_map


def _materialize_case_artifacts(
    *,
    bundle_root: Path,
    gt_root: Path,
    pred_root: Path,
    dataset: str,
    object_name: str,
    prompt_key: str,
) -> dict[str, str]:
    artifact_dir = ensure_dir(bundle_root / "artifacts" / dataset / object_name / prompt_key)
    prompt_dir = gt_root / dataset / object_name / prompt_key
    pred_dir = pred_root / dataset / object_name / prompt_key

    candidates = {
        "source_image": (prompt_dir / "2d_render.png", artifact_dir / "source_image.png"),
        "edit_image": (prompt_dir / "2d_edit.png", artifact_dir / "edit_image.png"),
        "mask_image": (prompt_dir / "2d_mask.png", artifact_dir / "mask_image.png"),
        "source_model_glb": (
            gt_root / dataset / object_name / "source_model" / "model.glb",
            artifact_dir / "source_model.glb",
        ),
        "source_voxelmesh_glb": (
            pred_dir / "source_voxelmesh" / "voxel_mesh.glb",
            artifact_dir / "source_voxelmesh.glb",
        ),
        "source_voxelmesh_transform": (
            pred_dir / "source_voxelmesh" / "voxel_mesh_transform.json",
            artifact_dir / "source_voxelmesh_transform.json",
        ),
        "mask_glb": (prompt_dir / "3d_edit_region.glb", artifact_dir / "mask.glb"),
        "edit_glb": (pred_dir / "edit.glb", artifact_dir / "edit.glb"),
        "ss_voxelmesh_glb": (pred_dir / "ss" / "voxel_mesh.glb", artifact_dir / "ss_voxelmesh.glb"),
        "ss_voxelmesh_transform": (
            pred_dir / "ss" / "voxel_mesh_transform.json",
            artifact_dir / "ss_voxelmesh_transform.json",
        ),
        "ss_coords_ply": (pred_dir / "ss" / "coords.ply", artifact_dir / "ss_coords.ply"),
        "ss_metadata": (pred_dir / "ss" / "ss_metadata.json", artifact_dir / "ss_metadata.json"),
        "images": (pred_dir / "images", artifact_dir / "images"),
        "videos": (pred_dir / "videos", artifact_dir / "videos"),
    }

    relpaths: dict[str, str] = {}
    for name, (src, dst) in candidates.items():
        if _link_or_copy(src, dst):
            relpaths[name] = _bundle_relpath(bundle_root, dst)
    return relpaths


def _link_or_copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    try:
        relative_target = os.path.relpath(src, start=dst.parent)
        os.symlink(relative_target, dst, target_is_directory=src.is_dir())
        return True
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        return True


def _extract_id_by_prefix(filename: str, prefix: str) -> str | None:
    prefix_token = f"{prefix}_"
    if not filename.startswith(prefix_token):
        return None
    suffix = filename[len(prefix_token) :].split(".", 1)[0]
    return suffix if suffix.isdigit() else None


def _build_eval_index(
    *,
    gt_root: Path,
    pred_root: Path,
    metadata: list[dict[str, Any]],
    selected_cases: set[tuple[str, str, int]],
) -> dict[str, Any]:
    image_id_map: dict[str, str] = {}
    clip_id_map: dict[str, str] = {}
    model_id_map: dict[str, str] = {}
    dino_id_map: dict[str, str] = {}

    for item in metadata:
        dataset = item.get("dataset")
        object_name = item.get("source_model")
        if not dataset or not object_name:
            continue

        for prompt_id in (1, 2, 3):
            if (dataset, object_name, prompt_id) not in selected_cases:
                continue
            prompt_key = f"prompt_{prompt_id}"
            if prompt_key not in item:
                continue

            case_id = _case_id(dataset, object_name, prompt_id)
            gt_render_dir = gt_root / dataset / object_name / "source_model" / "render"
            gt_mask_dir = gt_root / dataset / object_name / prompt_key / "render"
            pred_images_dir = pred_root / dataset / object_name / prompt_key / "images"
            if gt_render_dir.exists() and gt_mask_dir.exists() and pred_images_dir.exists():
                gt_images = {
                    image_id: file
                    for file in gt_render_dir.glob("render_*.png")
                    if (image_id := _extract_id_by_prefix(file.name, "render"))
                }
                gt_masks = {
                    mask_id: file
                    for file in gt_mask_dir.glob("mask_*.png")
                    if (mask_id := _extract_id_by_prefix(file.name, "mask"))
                }
                pred_images = {
                    image_id: file
                    for file in pred_images_dir.glob("render_*.png")
                    if (image_id := _extract_id_by_prefix(file.name, "render"))
                }
                common_ids = set(gt_images) & set(gt_masks) & set(pred_images)
                for image_id in sorted(common_ids, key=int):
                    image_id_map[
                        f"{dataset}_{object_name}_{prompt_key}_{image_id}"
                    ] = case_id

                for file in pred_images_dir.glob("render_*.png"):
                    image_id = _extract_id_by_prefix(file.name, "render")
                    if image_id and image_id in TEXT_ALIGNMENT_VIEW_IDS:
                        clip_id_map[
                            f"{dataset}_{object_name}_{prompt_key}_{image_id}"
                        ] = case_id

                pred_samples = []
                for file in pred_images_dir.glob("render_*.png"):
                    image_id = _extract_id_by_prefix(file.name, "render")
                    if image_id:
                        pred_samples.append((int(image_id), file))
                if pred_samples and (gt_root / dataset / object_name / prompt_key / "2d_edit.png").exists():
                    dino_id_map[f"{dataset}_{object_name}_{prompt_key}"] = case_id

            gt_model_path = gt_root / dataset / object_name / "source_model" / "model.glb"
            pred_model_path = pred_root / dataset / object_name / prompt_key / "edit.glb"
            mask_path = gt_root / dataset / object_name / prompt_key / "3d_edit_region.glb"
            if gt_model_path.exists() and pred_model_path.exists() and mask_path.exists():
                model_id_map[f"{dataset}_{object_name}_{prompt_key}"] = case_id

    return {
        "image_id_map": image_id_map,
        "clip_id_map": clip_id_map,
        "model_id_map": model_id_map,
        "dino_id_map": dino_id_map,
    }


def _build_per_case_metrics(
    *,
    gt_root: Path,
    pred_root: Path,
    metadata: list[dict[str, Any]],
    cases: list[tuple[str, str, int]],
    requested_metrics: list[str],
    detailed_results: dict[str, Any],
) -> dict[str, dict[str, float]]:
    per_case: dict[str, dict[str, float]] = defaultdict(dict)
    if not detailed_results:
        return per_case

    eval_index = _build_eval_index(
        gt_root=gt_root,
        pred_root=pred_root,
        metadata=metadata,
        selected_cases=set(cases),
    )
    image_id_map = eval_index["image_id_map"]
    clip_id_map = eval_index["clip_id_map"]
    model_id_map = eval_index["model_id_map"]
    dino_id_map = eval_index["dino_id_map"]

    for metric in ("psnr", "ssim", "lpips", "ssim_buffered", "lpips_buffered"):
        if metric not in requested_metrics:
            continue
        payload = detailed_results.get(metric)
        if not isinstance(payload, dict):
            continue
        explicit_scores = _aggregate_scores_by_eval_ids_from_payload(
            payload, image_id_map
        )
        if explicit_scores is None:
            continue
        for case_id, value in explicit_scores.items():
            per_case[case_id][metric] = value

    if "clip_t" in requested_metrics:
        payload = detailed_results.get("clip_t")
        explicit_scores = _aggregate_scores_by_eval_ids_from_payload(
            payload, clip_id_map
        )
        if explicit_scores is None:
            pass
        else:
            for case_id, value in explicit_scores.items():
                per_case[case_id]["clip_t"] = value

    if "chamfer" in requested_metrics:
        payload = detailed_results.get("chamfer")
        explicit_scores = _aggregate_scores_by_eval_ids_from_payload(
            payload, model_id_map
        )
        if explicit_scores is None:
            pass
        else:
            for case_id, value in explicit_scores.items():
                per_case[case_id]["chamfer"] = value

    for metric in ("dino_if_max", "dino_if_mean"):
        if metric not in requested_metrics:
            continue
        payload = detailed_results.get(metric)
        explicit_scores = _aggregate_scores_by_eval_ids_from_payload(
            payload, dino_id_map
        )
        if explicit_scores is None:
            continue
        for case_id, value in explicit_scores.items():
            per_case[case_id][metric] = value

    return per_case


def _aggregate_scores_by_eval_ids(
    eval_ids: list[Any],
    scores: list[Any],
    eval_id_map: dict[str, str],
) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for eval_id, score in zip(eval_ids, scores):
        if score is None:
            continue
        case_id = eval_id_map.get(str(eval_id))
        if case_id is None:
            continue
        buckets[case_id].append(float(score))
    return {
        case_id: float(sum(values) / len(values))
        for case_id, values in buckets.items()
        if values
    }


def _aggregate_scores_by_eval_ids_from_payload(
    payload: Any,
    eval_id_map: dict[str, str],
) -> dict[str, float] | None:
    if not isinstance(payload, dict):
        return None

    eval_ids = payload.get("all_ids")
    scores = payload.get("all_scores")
    if not isinstance(eval_ids, list) or not isinstance(scores, list):
        return None
    if len(eval_ids) != len(scores):
        return None
    return _aggregate_scores_by_eval_ids(eval_ids, scores, eval_id_map)


def _summarize_statuses(case_records: list[dict[str, Any]]) -> dict[str, int]:
    summary = {"cases": len(case_records), "ok": 0, "partial": 0, "failed": 0}
    for item in case_records:
        status = item.get("status")
        if status in summary:
            summary[status] += 1
    return summary


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _bundle_relpath(bundle_root: Path, path: Path) -> str:
    return path.relative_to(bundle_root).as_posix()


def _relative_href(page_path: Path, target_path: Path) -> str:
    return Path(os.path.relpath(target_path, start=page_path.parent)).as_posix()


def _href_from_page(page_path: Path, bundle_root: Path, bundle_relative_path: str | None) -> str | None:
    if not bundle_relative_path:
        return None
    target = bundle_root / bundle_relative_path
    return _relative_href(page_path, target)


def _href_between_paths(page_path: Path, target_path: Path | None) -> str | None:
    if target_path is None or not target_path.exists():
        return None
    return _relative_href(page_path, target_path)


def _ordered_metric_names(metric_names: list[str] | tuple[str, ...] | set[str]) -> list[str]:
    supported = {metric for metric in metric_names if metric in METRIC_META}
    return sorted(
        supported,
        key=lambda metric: (
            METRIC_FAMILY_ORDER.index(METRIC_META[metric]["family"]),
            METRIC_GROUP_ORDER_BY_FAMILY[METRIC_META[metric]["family"]].index(METRIC_META[metric]["group"]),
            int(METRIC_META[metric]["order"]),
            METRIC_META[metric]["label"],
        ),
    )


def _metric_label(metric_name: str) -> str:
    return str(METRIC_META[metric_name]["label"])


def _metric_prefers_higher(metric_name: str) -> bool:
    return metric_name not in {
        "lpips",
        "lpips_buffered",
        "lpips_in_seen",
        "lpips_in_novel",
        "fid",
        "fid_buffered",
        "fvd",
        "chamfer",
        "chamfer_target",
        "cross_seed_novel_lpips",
    }


def _metric_column_classes(metric_name: str) -> list[str]:
    classes: list[str] = []
    if metric_name in LEADING_METRIC_NAMES:
        classes.append("metric-leading")
    if metric_name in PRIMARY_METRIC_NAMES:
        classes.append("metric-emphasis")
    return classes


def _metric_table_class(metric_name: str, rank_class: str | None = None) -> str:
    classes = _metric_column_classes(metric_name)
    if rank_class:
        classes.append(rank_class)
    return " ".join(classes)


def _build_metric_rank_classes(
    row_metrics: list[dict[str, Any]],
    metric_names: list[str],
) -> dict[tuple[int, str], str]:
    rank_classes: dict[tuple[int, str], str] = {}
    for metric_name in metric_names:
        scored_rows: list[tuple[int, float]] = []
        for row_index, metrics in enumerate(row_metrics):
            scalar = _extract_scalar_metric((metrics or {}).get(metric_name))
            if scalar is None or not math.isfinite(scalar):
                continue
            scored_rows.append((row_index, scalar))
        scored_rows.sort(
            key=lambda item: item[1],
            reverse=_metric_prefers_higher(metric_name),
        )
        rank = 0
        previous_value: float | None = None
        for position, (row_index, scalar) in enumerate(scored_rows):
            if previous_value is None or abs(scalar - previous_value) > 1e-12:
                rank = position + 1
                previous_value = scalar
            if rank <= 3:
                rank_classes[(row_index, metric_name)] = f"rank-{rank}"
    return rank_classes


def _comparison_table_css() -> str:
    return """
    .table-wrap { overflow-x: auto; }
    .heatmap-note { margin: 12px 0 0; color: #666; font-size: 13px; }
    th.metric-leading { background: #e8f0fe; }
    th.metric-emphasis { background: #eef6ff; }
    td.metric-leading, td.metric-emphasis { font-weight: 600; color: #0f172a; }
    td.metric-leading { box-shadow: inset 0 0 0 1px #d7e3ff; }
    td.metric-emphasis { box-shadow: inset 0 0 0 1px #dbeafe; }
    td.rank-1, .metric-pill.rank-1 { background: #dcfce7; }
    td.rank-2, .metric-pill.rank-2 { background: #fef3c7; }
    td.rank-3, .metric-pill.rank-3 { background: #fee2e2; }
    .metric-stack { display: flex; flex-direction: column; gap: 6px; margin-top: 6px; }
    .metric-pill { display: flex; flex-direction: column; gap: 2px; padding: 6px 8px; border-radius: 10px; border: 1px solid #e5e7eb; background: #fafafa; }
    .metric-pill strong { font-size: 11px; font-weight: 600; color: #475569; }
    .metric-pill em { font-style: normal; font-size: 13px; color: #111827; }
    .metric-pill.metric-leading, .metric-pill.metric-emphasis { border-color: #cbd5e1; }
    """


def _primary_metric_names(metric_names: list[str] | tuple[str, ...] | set[str]) -> list[str]:
    supported = {metric_name for metric_name in metric_names if metric_name in METRIC_META}
    return [metric_name for metric_name in PRIMARY_METRIC_NAMES if metric_name in supported]


def _secondary_metric_names(metric_names: list[str] | tuple[str, ...] | set[str]) -> list[str]:
    primary = set(_primary_metric_names(metric_names))
    return [
        metric_name
        for metric_name in _ordered_metric_names(metric_names)
        if metric_name not in primary
    ]


def _table_metric_names(metric_names: list[str] | tuple[str, ...] | set[str]) -> list[str]:
    return _ordered_metric_names(metric_names)


def _collect_case_metric_names(case_records: list[dict[str, Any]]) -> list[str]:
    has_value_by_metric: dict[str, bool] = {}
    for case_record in case_records:
        for metric_name, metric_value in (case_record.get("metrics") or {}).items():
            if metric_name not in METRIC_META:
                continue
            has_value_by_metric[metric_name] = has_value_by_metric.get(metric_name, False) or metric_value is not None
    return _ordered_metric_names(
        [metric_name for metric_name, has_value in has_value_by_metric.items() if has_value]
    )


def _format_summary_metric_value(value: Any) -> str:
    if isinstance(value, dict):
        if value.get("value") is not None:
            return _format_metric_value(value.get("value"))
        if value.get("mean") is not None:
            return _format_metric_value(value.get("mean"))
        return "N/A"
    return _format_metric_value(value)


def _format_metric_meta_line(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    count = value.get("count")
    if count is None:
        return ""
    return f"count {count}"


def _render_metric_cards(items: list[tuple[str, Any]], *, extra_class: str = "") -> str:
    cards = []
    class_attr = "metric"
    if extra_class:
        class_attr = f"{class_attr} {extra_class}"
    for metric_name, metric_value in items:
        meta_line = _format_metric_meta_line(metric_value)
        cards.append(
            f"<div class='{html.escape(class_attr)}'>"
            f"<span>{html.escape(_metric_label(metric_name))}</span>"
            f"<strong>{html.escape(_format_summary_metric_value(metric_value))}</strong>"
            f"{f'<small>{html.escape(meta_line)}</small>' if meta_line else ''}"
            "</div>"
        )
    return "".join(cards)


def _group_metric_items(
    metrics: dict[str, Any],
    *,
    hide_none: bool,
    metric_names: list[str] | None = None,
    family_name: str | None = None,
) -> list[tuple[str, list[tuple[str, Any]]]]:
    if family_name is None:
        group_names = [
            group_name
            for current_family in METRIC_FAMILY_ORDER
            for group_name in METRIC_GROUP_ORDER_BY_FAMILY[current_family]
        ]
    else:
        group_names = list(METRIC_GROUP_ORDER_BY_FAMILY[family_name])
    grouped: dict[str, list[tuple[str, Any]]] = {group_name: [] for group_name in group_names}
    ordered_metric_names = metric_names or _ordered_metric_names(list(metrics.keys()))
    for metric_name in ordered_metric_names:
        if family_name is not None and METRIC_META[metric_name]["family"] != family_name:
            continue
        metric_value = metrics.get(metric_name)
        if hide_none and metric_value is None:
            continue
        grouped[METRIC_META[metric_name]["group"]].append((metric_name, metric_value))
    return [
        (group_name, grouped[group_name])
        for group_name in group_names
        if grouped[group_name]
    ]


def _render_metric_sections(
    metrics: dict[str, Any],
    *,
    hide_none: bool,
    empty_message: str,
    collapse_optional: bool = False,
) -> str:
    sections = []
    visible_metric_names = [
        metric_name
        for metric_name in _ordered_metric_names(list(metrics.keys()))
        if not hide_none or metrics.get(metric_name) is not None
    ]
    primary_items = [
        (metric_name, metrics.get(metric_name))
        for metric_name in _primary_metric_names(visible_metric_names)
        if not hide_none or metrics.get(metric_name) is not None
    ]
    if primary_items:
        sections.append(
            "<section class='metric-group metric-spotlight'>"
            "<div class='metric-group-head'><h3>Priority Metrics</h3><p class='muted'>Headline legacy and multi-view metrics are shown first.</p></div>"
            f"<div class='metric-grid'>{_render_metric_cards(primary_items, extra_class='metric-primary')}</div>"
            "</section>"
        )
    secondary_metric_names = _secondary_metric_names(visible_metric_names)
    for family_name in METRIC_FAMILY_ORDER:
        family_sections = []
        for group_name, items in _group_metric_items(
            metrics,
            hide_none=hide_none,
            metric_names=secondary_metric_names,
            family_name=family_name,
        ):
            family_sections.append(
                "<section class='metric-group'>"
                f"<div class='metric-group-head'><h3>{html.escape(group_name)}</h3><p class='muted'>{html.escape(METRIC_GROUP_META[group_name]['description'])}</p></div>"
                f"<div class='metric-grid'>{_render_metric_cards(items)}</div>"
                "</section>"
            )
        if family_sections:
            sections.append(
                "<section class='metric-family'>"
                f"<div class='metric-family-head'><h3>{html.escape(family_name)}</h3><p class='muted'>{html.escape(METRIC_FAMILY_META[family_name]['description'])}</p></div>"
                + "".join(family_sections)
                + "</section>"
            )
    if sections:
        return "".join(sections)
    return f"<p class='muted'>{html.escape(empty_message)}</p>"


def _render_metric_details(metrics: dict[str, Any], metric_names: list[str]) -> str:
    items = []
    for metric_name in metric_names:
        metric_value = metrics.get(metric_name)
        if metric_value is None:
            continue
        items.append(
            f"<div>{html.escape(_metric_label(metric_name))} {html.escape(_format_summary_metric_value(metric_value))}</div>"
        )
    if not items:
        return "<span class='muted'>-</span>"
    return (
        "<details class='metric-inline-details'>"
        "<summary>other</summary>"
        f"<div class='metric-inline-list'>{''.join(items)}</div>"
        "</details>"
    )


def _render_run_index(
    *,
    bundle_root: Path,
    manifest_payload: dict[str, Any],
    summary_payload: dict[str, Any],
    case_records: list[dict[str, Any]],
) -> str:
    daily_root = bundle_root.parent
    daily_href = _relative_href(bundle_root / "index.html", daily_root / "index.html")
    group_name = str(manifest_payload.get("group") or "").strip()
    group_href = None
    if group_name:
        group_href = _relative_href(
            bundle_root / "index.html",
            daily_root / "groups" / f"{_slug(group_name)}.html",
        )

    summary_metrics = _render_metric_sections(
        summary_payload.get("metrics", {}),
        hide_none=True,
        empty_message="No supported summary metrics were found in this bundle.",
    )
    case_metric_names = _table_metric_names(_collect_case_metric_names(case_records))
    rank_classes = _build_metric_rank_classes(
        [item.get("metrics") or {} for item in case_records],
        case_metric_names,
    )

    case_rows = []
    for row_index, item in enumerate(case_records):
        metric_cells = []
        row_metrics = item.get("metrics") or {}
        for metric_name in case_metric_names:
            class_name = _metric_table_class(
                metric_name,
                rank_classes.get((row_index, metric_name)),
            )
            class_attr = f" class=\"{html.escape(class_name)}\"" if class_name else ""
            metric_cells.append(
                f"<td{class_attr}>{html.escape(_format_metric_value(row_metrics.get(metric_name)))}</td>"
            )
        case_rows.append(
            "<tr>"
            f"<td><a href=\"{html.escape(item['page_path'])}\">{html.escape(item['case_id'])}</a></td>"
            f"<td>{html.escape(item['status'])}</td>"
            + "".join(metric_cells)
            + "</tr>"
        )

    metric_headers = "".join(
        (
            f"<th class=\"{html.escape(_metric_table_class(metric_name))}\">"
            f"{html.escape(_metric_label(metric_name))}</th>"
        )
        for metric_name in case_metric_names
    )
    return f'''<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(manifest_payload['run_id'])}</title>
  <style>
    body {{ font-family: "Segoe UI", sans-serif; margin: 0; background: #f7f7f7; color: #222; }}
    .page {{ width: min(1680px, calc(100vw - 32px)); margin: 24px auto 48px; }}
    .hero, .panel {{ background: #fff; border: 1px solid #ddd; border-radius: 14px; padding: 18px 20px; }}
    .hero {{ margin-bottom: 18px; }}
    .nav-links {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 14px; }}
    .nav-link {{ display: inline-flex; align-items: center; padding: 8px 12px; border-radius: 999px; border: 1px solid #d9d9d9; background: #fafafa; color: #222; text-decoration: none; font-size: 13px; }}
    .stats {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 14px; }}
    .stat, .metric {{ background: #fafafa; border: 1px solid #e5e5e5; border-radius: 12px; padding: 12px 14px; }}
    .stat strong, .metric strong {{ display: block; font-size: 20px; margin-top: 4px; }}
    .metric-family + .metric-family {{ margin-top: 22px; }}
    .metric-family-head {{ margin-bottom: 12px; }}
    .metric-family-head h3 {{ margin: 0 0 4px; }}
    .metric-family-head p {{ margin: 0; }}
    .metric-group + .metric-group {{ margin-top: 18px; }}
    .metric-spotlight {{ margin-bottom: 14px; }}
    .metric-primary {{ border-color: #cbd5e1; background: #f8fbff; }}
    .metric-group-head {{ margin-bottom: 10px; }}
    .metric-group-head h3 {{ margin: 0 0 4px; }}
    .metric-group-head p {{ margin: 0; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }}
    .metric small {{ display: block; margin-top: 6px; color: #666; font-size: 12px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 10px 8px; border-bottom: 1px solid #eee; text-align: left; font-size: 14px; vertical-align: top; }}
    th {{ background: #fafafa; position: sticky; top: 0; }}
    a {{ color: #0b57d0; text-decoration: none; }}
    h1, h2 {{ margin: 0 0 10px; }}
    .muted {{ color: #666; }}
    {_comparison_table_css()}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="nav-links">
        <a class="nav-link" href="{html.escape(daily_href)}">Back To Daily</a>
        {f'<a class="nav-link" href="{html.escape(group_href)}">Back To Group: {html.escape(group_name)}</a>' if group_href else ''}
      </div>
      <h1>{html.escape(manifest_payload['config_name'])}</h1>
      <p class="muted">run_id: {html.escape(manifest_payload['run_id'])}</p>
      <div class="stats">
        <div class="stat"><span>cases</span><strong>{summary_payload['totals']['cases']}</strong></div>
        <div class="stat"><span>ok</span><strong>{summary_payload['totals']['ok']}</strong></div>
        <div class="stat"><span>partial</span><strong>{summary_payload['totals']['partial']}</strong></div>
        <div class="stat"><span>failed</span><strong>{summary_payload['totals']['failed']}</strong></div>
      </div>
    </section>
    <section class="panel" style="margin-bottom: 18px;">
      <h2>Summary Metrics</h2>
      {summary_metrics}
    </section>
    <section class="panel">
      <h2>Cases</h2>
      <p class="heatmap-note">Column heatmap shows the best, second, and third values for each metric across cases.</p>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>case</th>
              <th>status</th>
              {metric_headers}
            </tr>
          </thead>
          <tbody>
            {''.join(case_rows)}
          </tbody>
        </table>
      </div>
      {"<p class='muted' style='margin-top: 12px;'>This bundle has no per-case metrics under the current protocol.</p>" if not case_metric_names else ""}
    </section>
  </div>
</body>
</html>
'''


def _render_daily_runs_table(
    *,
    daily_root: Path,
    page_path: Path,
    runs: list[dict[str, Any]],
) -> str:
    all_metric_names = _ordered_metric_names(
        {
            metric_name
            for run in runs
            for metric_name in (run.get("summary") or {}).get("metrics", {}).keys()
        }
    )
    metric_names = _table_metric_names(all_metric_names)
    rank_classes = _build_metric_rank_classes(
        [((run.get("summary") or {}).get("metrics") or {}) for run in runs],
        metric_names,
    )
    headers = "".join(
        f"<th class='{html.escape(_metric_table_class(metric_name))}'>{html.escape(_metric_label(metric_name))}</th>"
        for metric_name in metric_names
    )

    rows = []
    for row_index, run in enumerate(runs):
        run_href = _href_between_paths(page_path, daily_root / run["path"] / "index.html") or "#"
        totals = (run.get("summary") or {}).get("totals") or {}
        summary_metrics = (run.get("summary") or {}).get("metrics", {})
        metric_cells = []
        for metric_name in metric_names:
            class_name = _metric_table_class(
                metric_name,
                rank_classes.get((row_index, metric_name)),
            )
            class_attr = f" class='{html.escape(class_name)}'" if class_name else ""
            metric_cells.append(
                f"<td{class_attr}>{html.escape(_format_summary_metric_value(summary_metrics.get(metric_name)))}</td>"
            )
        rows.append(
            "<tr>"
            f"<td><a href='{html.escape(run_href)}'>{html.escape(run['label'])}</a><div class='muted small'>{html.escape(run['id'])}</div></td>"
            + "".join(metric_cells)
            + f"<td>{html.escape(str(run.get('created_at') or ''))}</td>"
            + f"<td>{html.escape(str(totals.get('cases', 0)))}</td>"
            + f"<td>{html.escape(str(totals.get('ok', 0)))}</td>"
            + f"<td>{html.escape(str(totals.get('partial', 0)))}</td>"
            + f"<td>{html.escape(str(totals.get('failed', 0)))}</td>"
            + "</tr>"
        )

    return (
        "<p class='heatmap-note'>Column heatmap shows the best, second, and third runs for each metric.</p>"
        "<div class='table-wrap'><table>"
        "<thead>"
        "<tr>"
        "<th>run</th>"
        f"{headers}"
        "<th>created_at</th>"
        "<th>cases</th>"
        "<th>ok</th>"
        "<th>partial</th>"
        "<th>failed</th>"
        "</tr>"
        "</thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table></div>"
    )


def _render_daily_collection_page(
    *,
    daily_root: Path,
    page_path: Path,
    title: str,
    heading: str,
    subtitle: str,
    grouped_runs: list[tuple[str, list[dict[str, Any]]]],
    group_pages: list[dict[str, Any]],
    current_group: str | None,
) -> str:
    back_href = None
    if current_group is not None:
        back_href = _relative_href(page_path, daily_root / "index.html")

    nav_links = []
    for item in group_pages:
        if current_group is None:
            href = f"#group-{_slug(item['name'])}"
        else:
            href = _relative_href(page_path, daily_root / item["path"])
        class_name = "chip current" if item["name"] == current_group else "chip"
        nav_links.append(
            f"<a class=\"{html.escape(class_name)}\" href=\"{html.escape(href)}\">{html.escape(item['name'])} ({item['count']})</a>"
        )

    sections = []
    for group_name, runs in grouped_runs:
        if not runs:
            continue
        group_page_relpath = next(
            (item["path"] for item in group_pages if item["name"] == group_name),
            None,
        )
        group_page_href = (
            _relative_href(page_path, daily_root / group_page_relpath)
            if group_page_relpath
            else None
        )
        actions = []
        if current_group is None and group_page_href is not None:
            actions.append(
                f"<a href=\"{html.escape(group_page_href)}\">group page</a>"
            )
        sections.append(
            "<section class='panel'>"
            "<div class='section-head'>"
            f"<div><h2 id=\"group-{html.escape(_slug(group_name))}\">{html.escape(group_name)}</h2><p class='muted'>{len(runs)} run(s)</p></div>"
            f"<div class='section-links'>{' · '.join(actions)}</div>"
            "</div>"
            f"{_render_daily_runs_table(daily_root=daily_root, page_path=page_path, runs=runs)}"
            "</section>"
        )

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: "Segoe UI", sans-serif; margin: 0; background: #f7f7f7; color: #222; }}
    .page {{ width: min(1500px, calc(100vw - 32px)); margin: 24px auto 48px; }}
    .hero, .panel {{ background: #fff; border: 1px solid #ddd; border-radius: 14px; padding: 18px 20px; }}
    .panel + .panel {{ margin-top: 18px; }}
    .chips {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 14px; }}
    .chip {{ display: inline-flex; align-items: center; padding: 8px 12px; border-radius: 999px; border: 1px solid #d9d9d9; background: #fafafa; color: #222; text-decoration: none; font-size: 13px; }}
    .chip.current {{ background: #e8f0fe; border-color: #b6ccff; }}
    .section-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; margin-bottom: 12px; }}
    .section-links {{ font-size: 13px; }}
    h2[id] {{ scroll-margin-top: 20px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 10px 8px; border-bottom: 1px solid #eee; text-align: left; font-size: 14px; vertical-align: top; }}
    th {{ background: #fafafa; position: sticky; top: 0; }}
    a {{ color: #0b57d0; text-decoration: none; }}
    h1, h2 {{ margin: 0; }}
    .muted {{ color: #666; }}
    .small {{ font-size: 12px; margin-top: 4px; }}
    {_comparison_table_css()}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      {f'<p><a href="{html.escape(back_href)}">Back To Daily</a></p>' if back_href else ''}
      <h1>{html.escape(heading)}</h1>
      <p class="muted" style="margin-top: 8px;">{html.escape(subtitle)}</p>
      {f'<div class="chips">{"".join(nav_links)}</div>' if nav_links else ''}
    </section>
    {''.join(sections)}
  </div>
</body>
</html>
"""


def _render_case_page(
    *,
    case_payload: dict[str, Any],
    bundle_root: Path,
    page_path: Path,
    navigation_payload: dict[str, Any],
    run_label: str,
    run_id: str,
) -> str:
    artifact_links = []
    for label, bundle_relative_path in case_payload.get("artifacts", {}).items():
        href = _href_from_page(page_path, bundle_root, bundle_relative_path)
        if href is None:
            continue
        artifact_links.append(
            f"<div class='artifact'><span>{html.escape(label)}</span><a href=\"{html.escape(href)}\" target=\"_blank\" rel=\"noopener\">open</a></div>"
        )

    image_blocks = []
    for key in ("source_image", "edit_image", "mask_image"):
        bundle_relative_path = case_payload.get("artifacts", {}).get(key)
        href = _href_from_page(page_path, bundle_root, bundle_relative_path)
        if href is None:
            continue
        image_blocks.append(
            f"<section class='image-card'><h2>{html.escape(key)}</h2><img src=\"{html.escape(href)}\" alt=\"{html.escape(key)}\"></section>"
        )

    source_model_href = _href_from_page(
        page_path,
        bundle_root,
        case_payload.get("artifacts", {}).get("source_model_glb"),
    )
    source_voxelmesh_href = _href_from_page(
        page_path,
        bundle_root,
        case_payload.get("artifacts", {}).get("source_voxelmesh_glb"),
    )
    mask_glb_href = _href_from_page(
        page_path,
        bundle_root,
        case_payload.get("artifacts", {}).get("mask_glb"),
    )
    edit_glb_href = _href_from_page(
        page_path,
        bundle_root,
        case_payload.get("artifacts", {}).get("edit_glb"),
    )
    ss_voxelmesh_href = _href_from_page(
        page_path,
        bundle_root,
        case_payload.get("artifacts", {}).get("ss_voxelmesh_glb"),
    )
    metrics = _render_metric_sections(
        case_payload.get("metrics", {}),
        hide_none=True,
        empty_message="No per-case metrics are available for this case under the current protocol.",
    )
    dataset_options = "".join(
        (
            f"<option value=\"{html.escape(str(item.get('dataset') or ''))}\""
            f"{' selected' if item.get('selected') else ''}>"
            f"{html.escape(str(item.get('dataset') or ''))} ({html.escape(str(item.get('count') or 0))})"
            "</option>"
        )
        for item in navigation_payload.get("datasets", [])
    )
    back_href = str(navigation_payload.get("run_href") or _relative_href(page_path, bundle_root / "index.html"))
    progress_text = (
        f"{navigation_payload.get('current_index', 1)} / {navigation_payload.get('total_cases', 1)}"
    )
    if navigation_payload.get("current_dataset"):
        progress_text = f"{progress_text} · {navigation_payload['current_dataset']}"

    dataset = str(case_payload.get("dataset") or "")
    object_name = str(case_payload.get("object_name") or "")
    prompt_id = int(case_payload.get("prompt_id") or 0)
    prompt_label = f"prompt_{prompt_id}"
    prompt_text = str(case_payload.get("prompt_text") or "").strip()
    sample_title = object_name or case_payload["case_id"]
    image_section = (
        f"<div class=\"image-grid\">{''.join(image_blocks)}</div>"
        if image_blocks
        else "<p class='muted'>No reference images were packaged for this case.</p>"
    )
    artifact_section = (
        f"<div class=\"artifact-grid\">{''.join(artifact_links)}</div>"
        if artifact_links
        else "<p class='muted'>No downloadable artifacts were packaged for this case.</p>"
    )
    status = str(case_payload.get("status") or "unknown")
    status_class = status if status in {"ok", "partial", "failed"} else "unknown"
    navigation_json = json.dumps(navigation_payload, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(case_payload['case_id'])}</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three/examples/js/loaders/GLTFLoader.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three/examples/js/controls/OrbitControls.js"></script>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: "Segoe UI", sans-serif; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #222; }}
    .page-shell {{ width: min(1800px, calc(100vw - 32px)); margin: 20px auto; }}
    .container {{ background: rgba(255, 255, 255, 0.96); backdrop-filter: blur(10px); border-radius: 20px; box-shadow: 0 20px 40px rgba(0, 0, 0, 0.12); overflow: hidden; }}
    .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #fff; padding: 32px 30px; position: relative; overflow: hidden; }}
    .header::before {{ content: ""; position: absolute; inset: 0; background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="white" opacity="0.1"/><circle cx="75" cy="75" r="1" fill="white" opacity="0.1"/><circle cx="50" cy="10" r="0.5" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>'); opacity: 0.3; }}
    .header-content {{ position: relative; z-index: 1; }}
    .header h1 {{ margin: 0; font-size: 2.1em; font-weight: 700; letter-spacing: -0.02em; }}
    .header p {{ margin: 8px 0 0; opacity: 0.9; }}
    .back-link {{ display: inline-flex; align-items: center; margin-bottom: 14px; color: #fff; text-decoration: none; font-weight: 600; }}
    .navigation {{ background: #f8fafc; padding: 24px 30px; border-bottom: 1px solid #e2e8f0; }}
    .top-controls {{ display: flex; align-items: center; justify-content: center; gap: 24px; margin-bottom: 18px; flex-wrap: wrap; }}
    .control-group {{ display: flex; align-items: center; gap: 12px; background: #fff; padding: 12px 18px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05); }}
    .control-group label {{ font-weight: 600; color: #475569; font-size: 14px; }}
    .control-group select, .control-group input {{ padding: 10px 14px; border-radius: 8px; border: 2px solid #e2e8f0; font-size: 14px; background: #fff; transition: all 0.2s ease; }}
    .control-group select:focus, .control-group input:focus, .page-input-group input:focus {{ outline: none; border-color: #667eea; box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.12); }}
    .control-group input {{ width: 220px; }}
    .search-btn, .nav-button {{ border: none; color: #fff; cursor: pointer; font-size: 14px; font-weight: 600; transition: all 0.2s ease; }}
    .search-btn {{ background: linear-gradient(135deg, #10b981 0%, #059669 100%); padding: 10px 18px; border-radius: 10px; }}
    .nav-button {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 12px 24px; border-radius: 12px; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3); }}
    .search-btn:hover, .nav-button:hover {{ transform: translateY(-1px); }}
    .search-btn:disabled, .nav-button:disabled {{ background: #cbd5e1; cursor: not-allowed; transform: none; box-shadow: none; }}
    .navigation-controls {{ display: flex; align-items: center; justify-content: center; gap: 18px; flex-wrap: wrap; }}
    .page-input-group {{ display: flex; align-items: center; gap: 10px; background: #fff; padding: 8px 16px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05); }}
    .page-input-group input {{ width: 86px; padding: 8px 12px; border: 2px solid #e2e8f0; border-radius: 8px; text-align: center; font-size: 14px; }}
    .progress-info {{ font-size: 15px; color: #475569; font-weight: 600; background: #fff; padding: 10px 18px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05); }}
    .content {{ padding: 30px; }}
    .panel {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 16px; padding: 18px 20px; margin-bottom: 18px; box-shadow: 0 4px 18px rgba(15, 23, 42, 0.04); }}
    .sample-info {{ background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%); padding: 24px; border-radius: 16px; margin-bottom: 22px; border-left: 4px solid #667eea; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05); }}
    .sample-head {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; flex-wrap: wrap; }}
    .sample-tags {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 14px; }}
    .tag, .status-badge {{ display: inline-flex; align-items: center; padding: 7px 12px; border-radius: 999px; font-size: 13px; font-weight: 600; }}
    .tag {{ background: rgba(255, 255, 255, 0.86); color: #334155; border: 1px solid rgba(148, 163, 184, 0.35); }}
    .status-badge.ok {{ background: rgba(16, 185, 129, 0.16); color: #047857; }}
    .status-badge.partial {{ background: rgba(245, 158, 11, 0.16); color: #b45309; }}
    .status-badge.failed {{ background: rgba(239, 68, 68, 0.16); color: #b91c1c; }}
    .status-badge.unknown {{ background: rgba(100, 116, 139, 0.16); color: #475569; }}
    .sample-info h2 {{ margin: 0; color: #0f172a; font-size: 1.6em; }}
    .prompt-info {{ background: rgba(255, 255, 255, 0.72); padding: 16px; border-radius: 12px; margin-top: 16px; border-left: 3px solid #667eea; line-height: 1.7; }}
    .metric-grid, .artifact-grid, .image-grid {{ display: grid; gap: 12px; }}
    .metric-sections {{ display: grid; gap: 12px; }}
    .metric-grid {{ grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); }}
    .artifact-grid {{ grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }}
    .image-grid {{ grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }}
    .viewer-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
    .metric, .artifact {{ background: #fafafa; border: 1px solid #e5e5e5; border-radius: 12px; padding: 12px 14px; }}
    .metric strong {{ display: block; font-size: 20px; margin-top: 4px; }}
    .metric small {{ display: block; margin-top: 6px; color: #666; font-size: 12px; }}
    .metric-family + .metric-family {{ margin-top: 22px; }}
    .metric-family-head {{ margin-bottom: 12px; }}
    .metric-family-head h3 {{ margin: 0 0 4px; }}
    .metric-family-head p {{ margin: 0; }}
    .metric-group + .metric-group {{ margin-top: 18px; }}
    .metric-spotlight {{ margin-bottom: 14px; }}
    .metric-primary {{ border-color: #cbd5e1; background: #f8fbff; }}
    .metric-group-head {{ margin-bottom: 10px; }}
    .metric-group-head h3 {{ margin: 0 0 4px; }}
    .metric-group-head p {{ margin: 0; }}
    .metric-more {{ margin-top: 12px; }}
    .metric-more > summary {{ cursor: pointer; color: #0b57d0; }}
    img {{ width: 100%; border-radius: 12px; border: 1px solid #ddd; background: #fff; }}
    .viewer-card {{ background: #fafafa; border: 1px solid #e5e5e5; border-radius: 12px; padding: 12px; }}
    .viewer-card h2 {{ margin: 0 0 10px; font-size: 16px; }}
    .viewer-3d {{ width: 100%; height: 360px; border: 2px solid #e2e8f0; border-radius: 12px; background: #f8fafc; overflow: hidden; }}
    a {{ color: #0b57d0; text-decoration: none; }}
    .muted {{ color: #666; }}
    h2, h3 {{ margin: 0 0 10px; }}
    @media (max-width: 960px) {{
      .viewer-grid {{ grid-template-columns: 1fr; }}
      .content {{ padding: 22px; }}
      .navigation {{ padding: 20px 22px; }}
      .header {{ padding: 28px 22px; }}
      .control-group input {{ width: 180px; }}
    }}
  </style>
</head>
<body>
  <div class="page-shell">
    <div class="container">
      <header class="header">
        <div class="header-content">
          <a class="back-link" href="{html.escape(back_href)}">Back To Run</a>
          <h1>{html.escape(run_label)}</h1>
          <p>{html.escape(run_id)}</p>
        </div>
      </header>
      <section class="navigation">
        <div class="top-controls">
          <div class="control-group">
            <label for="datasetSelect">Dataset:</label>
            <select id="datasetSelect">
              {dataset_options}
            </select>
          </div>
          <div class="control-group">
            <label for="modelSearch">Sample Search:</label>
            <input type="text" id="modelSearch" placeholder="Enter object name..." />
            <button id="searchBtn" class="search-btn">Search</button>
          </div>
        </div>
        <div class="navigation-controls">
          <button class="nav-button" id="prevBtn">Previous</button>
          <div class="page-input-group">
            <label for="pageInput">Go to:</label>
            <input type="number" id="pageInput" placeholder="Page" min="1" max="{html.escape(str(navigation_payload.get('total_cases') or 1))}" />
            <button class="nav-button" id="goToBtn">Go</button>
          </div>
          <span class="progress-info" id="progressInfo">{html.escape(progress_text)}</span>
          <button class="nav-button" id="nextBtn">Next</button>
        </div>
      </section>
      <main class="content">
        <section class="sample-info">
          <div class="sample-head">
            <div>
              <div class="sample-tags">
                <span class="tag">{html.escape(dataset)}</span>
                <span class="tag">{html.escape(prompt_label)}</span>
                <span class="status-badge {html.escape(status_class)}">{html.escape(status)}</span>
              </div>
              <h2>{html.escape(sample_title)}</h2>
            </div>
          </div>
          <div class="prompt-info">
            <strong>Case:</strong> {html.escape(case_payload['case_id'])}<br>
            <strong>Prompt:</strong> {html.escape(prompt_text or "No prompt text was recorded for this case.")}
          </div>
        </section>
        <section class="panel">
          <h2>Metrics</h2>
          <div class="metric-sections">{metrics}</div>
        </section>
        <section class="panel">
          <h2>Images</h2>
          {image_section}
        </section>
        <section class="panel">
          <h2>3D Compare</h2>
          <div class="viewer-grid">
            <section class="viewer-card">
              <h2>Source Model</h2>
              <div id="viewer-source" class="viewer-3d"></div>
            </section>
            <section class="viewer-card">
              <h2>Source VoxelMesh</h2>
              <div id="viewer-source-voxelmesh" class="viewer-3d"></div>
            </section>
            <section class="viewer-card">
              <h2>Source + Mask</h2>
              <div id="viewer-combined" class="viewer-3d"></div>
            </section>
            <section class="viewer-card">
              <h2>SS Coords</h2>
              <div id="viewer-ss-voxelmesh" class="viewer-3d"></div>
            </section>
            <section class="viewer-card">
              <h2>Final Edit</h2>
              <div id="viewer-edit" class="viewer-3d"></div>
            </section>
          </div>
        </section>
        <section class="panel">
          <h2>Artifacts</h2>
          {artifact_section}
        </section>
      </main>
    </div>
  </div>
  <script>
    const navigationData = {navigation_json};
    const sourceModelHref = {json.dumps(source_model_href)};
    const sourceVoxelmeshHref = {json.dumps(source_voxelmesh_href)};
    const maskGlbHref = {json.dumps(mask_glb_href)};
    const editGlbHref = {json.dumps(edit_glb_href)};
    const ssVoxelmeshHref = {json.dumps(ss_voxelmesh_href)};

    function navigateSample(direction) {{
      const cases = navigationData.cases || [];
      if (!cases.length) return;
      const currentIndex = Math.max((navigationData.current_index || 1) - 1, 0);
      let nextIndex = currentIndex + direction;
      if (nextIndex < 0) nextIndex = cases.length - 1;
      if (nextIndex >= cases.length) nextIndex = 0;
      const target = cases[nextIndex];
      if (target && target.href) {{
        window.location.href = target.href;
      }}
    }}

    function goToPage() {{
      const pageInput = document.getElementById("pageInput");
      const pageNumber = parseInt(pageInput.value, 10);
      const totalCases = navigationData.total_cases || 0;
      if (Number.isNaN(pageNumber) || pageNumber < 1 || pageNumber > totalCases) {{
        alert(`Please enter a valid page number between 1 and ${{totalCases}}`);
        return;
      }}
      const target = (navigationData.cases || []).find((item) => item.index === pageNumber);
      if (target && target.href) {{
        window.location.href = target.href;
      }}
    }}

    function searchSample() {{
      const searchInput = document.getElementById("modelSearch");
      const query = searchInput.value.trim().toLowerCase();
      if (!query) {{
        alert("Please enter an object name");
        return;
      }}
      const target = (navigationData.cases || []).find((item) => (item.search_text || "").includes(query));
      if (!target || !target.href) {{
        alert(`No sample found containing "${{searchInput.value.trim()}}" in ${{navigationData.current_dataset || "this dataset"}}`);
        return;
      }}
      window.location.href = target.href;
    }}

    function setupNavigation() {{
      const prevBtn = document.getElementById("prevBtn");
      const nextBtn = document.getElementById("nextBtn");
      const goToBtn = document.getElementById("goToBtn");
      const searchBtn = document.getElementById("searchBtn");
      const datasetSelect = document.getElementById("datasetSelect");
      const pageInput = document.getElementById("pageInput");
      const modelSearch = document.getElementById("modelSearch");
      const hasMultipleCases = (navigationData.total_cases || 0) > 1;

      prevBtn.disabled = !hasMultipleCases;
      nextBtn.disabled = !hasMultipleCases;

      prevBtn.addEventListener("click", () => navigateSample(-1));
      nextBtn.addEventListener("click", () => navigateSample(1));
      goToBtn.addEventListener("click", goToPage);
      searchBtn.addEventListener("click", searchSample);

      datasetSelect.addEventListener("change", (event) => {{
        const targetDataset = (navigationData.datasets || []).find((item) => item.dataset === event.target.value);
        if (targetDataset && targetDataset.href) {{
          window.location.href = targetDataset.href;
        }}
      }});

      pageInput.addEventListener("keypress", (event) => {{
        if (event.key === "Enter") {{
          goToPage();
        }}
      }});

      modelSearch.addEventListener("keypress", (event) => {{
        if (event.key === "Enter") {{
          searchSample();
        }}
      }});

      document.addEventListener("keydown", (event) => {{
        if (event.target && ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) {{
          return;
        }}
        if (event.key === "ArrowLeft" && hasMultipleCases) {{
          navigateSample(-1);
        }}
        if (event.key === "ArrowRight" && hasMultipleCases) {{
          navigateSample(1);
        }}
      }});
    }}

    function createViewer(containerId) {{
      const container = document.getElementById(containerId);
      if (!container) return null;
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0xf8fafc);
      const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
      const renderer = new THREE.WebGLRenderer({{ antialias: true }});
      renderer.setSize(container.clientWidth, container.clientHeight);
      renderer.setPixelRatio(window.devicePixelRatio || 1);
      renderer.setClearColor(0xffffff);
      container.appendChild(renderer.domElement);

      renderer.outputEncoding = THREE.sRGBEncoding;
      renderer.physicallyCorrectLights = true;

      const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
      directionalLight.position.set(5, 10, 7);
      scene.add(directionalLight);

      const lightIntensity = 30;
      const lightDistance = 100;
      const directions = [
        [10, 0, 0], [-10, 0, 0], [0, 10, 0], [0, -10, 0], [0, 0, 10], [0, 0, -10],
      ];
      directions.forEach((dir, index) => {{
        const pointLight = new THREE.PointLight(0xffffff, lightIntensity, lightDistance);
        pointLight.position.set(...dir);
        pointLight.castShadow = true;
        pointLight.name = `PointLight_${{index}}`;
        scene.add(pointLight);
      }});

      const controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;

      function frameObject(object) {{
        const box = new THREE.Box3().setFromObject(object);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z) || 1;
        camera.position.copy(center);
        camera.position.z += maxDim * 1.2;
        controls.target.copy(center);
        controls.update();
      }}

      function animate() {{
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
      }}
      animate();

      const resizeObserver = new ResizeObserver(() => {{
        const width = container.clientWidth;
        const height = container.clientHeight;
        camera.aspect = width / Math.max(height, 1);
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);
      }});
      resizeObserver.observe(container);

      return {{ scene, camera, renderer, controls, frameObject }};
    }}

    function loadGlb(sceneBundle, href, materialPatch) {{
      return new Promise((resolve, reject) => {{
        if (!sceneBundle || !href) {{
          resolve(null);
          return;
        }}
        const loader = new THREE.GLTFLoader();
        loader.load(
          href,
          (gltf) => {{
            const model = gltf.scene;
            if (materialPatch) {{
              model.traverse((node) => {{
                if (node.isMesh) {{
                  materialPatch(node);
                }}
              }});
            }}
            sceneBundle.scene.add(model);
            resolve(model);
          }},
          undefined,
          reject
        );
      }});
    }}

    async function initViewers() {{
      try {{
        const sourceViewer = createViewer("viewer-source");
        const sourceVoxelViewer = createViewer("viewer-source-voxelmesh");
        const combinedViewer = createViewer("viewer-combined");
        const ssVoxelViewer = createViewer("viewer-ss-voxelmesh");
        const editViewer = createViewer("viewer-edit");

        const sourceModel = await loadGlb(sourceViewer, sourceModelHref);
        if (sourceModel) sourceViewer.frameObject(sourceModel);

        const sourceVoxelMesh = await loadGlb(sourceVoxelViewer, sourceVoxelmeshHref);
        if (sourceVoxelMesh) sourceVoxelViewer.frameObject(sourceVoxelMesh);

        let combinedFrameTarget = null;
        const combinedSource = await loadGlb(combinedViewer, sourceModelHref);
        if (combinedSource) combinedFrameTarget = combinedSource;
        await loadGlb(combinedViewer, maskGlbHref, (node) => {{
          node.material = new THREE.MeshPhongMaterial({{
            color: 0xcccccc,
            transparent: true,
            opacity: 0.7,
          }});
        }});
        if (combinedFrameTarget) combinedViewer.frameObject(combinedFrameTarget);

        const ssVoxelMesh = await loadGlb(ssVoxelViewer, ssVoxelmeshHref);
        if (ssVoxelMesh) ssVoxelViewer.frameObject(ssVoxelMesh);

        const editModel = await loadGlb(editViewer, editGlbHref);
        if (editModel) editViewer.frameObject(editModel);
      }} catch (error) {{
        console.error("Failed to initialize 3D viewers", error);
      }}
    }}

    setupNavigation();
    initViewers();
  </script>
</body>
</html>
"""


def _build_focus_case_pages(
    *,
    benchmark_root: Path,
    focus_root: Path,
    runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    case_ids = sorted(
        {
            str(case_record.get("case_id"))
            for run in runs
            for case_record in run.get("case_records", [])
            if case_record.get("case_id")
        }
    )
    case_pages: list[dict[str, Any]] = []
    focus_cases_root = ensure_dir(focus_root / "cases")
    for case_id in case_ids:
        dataset, object_name, prompt_name = case_id.split("/", 2)
        prompt_id = int(prompt_name.rsplit("_", 1)[-1])
        page_path = focus_cases_root / dataset / object_name / f"prompt_{prompt_id}.html"
        ensure_dir(page_path.parent)

        prompt_text = ""
        reference_run_root: Path | None = None
        reference_artifacts: dict[str, Any] = {}
        run_entries: list[dict[str, Any]] = []
        for run in runs:
            case_record = run["case_map"].get(case_id)
            if case_record is None:
                continue

            detail_relpath = case_record.get("detail_path")
            detail_payload = _load_json(run["root"] / detail_relpath) if detail_relpath else {}
            if not prompt_text:
                prompt_text = str(detail_payload.get("prompt_text") or "")

            detail_page_relpath = case_record.get("page_path")
            detail_page_href = None
            if detail_page_relpath:
                detail_page_href = _href_between_paths(page_path, run["root"] / detail_page_relpath)

            artifacts = detail_payload.get("artifacts", {})
            if not reference_artifacts and artifacts:
                reference_run_root = run["root"]
                reference_artifacts = dict(artifacts)
            edit_glb_relpath = artifacts.get("edit_glb")
            render_relpath = artifacts.get("images") or artifacts.get("videos")
            edit_glb_href = None
            render_href = None
            if edit_glb_relpath:
                edit_glb_href = _href_between_paths(page_path, run["root"] / edit_glb_relpath)
            if render_relpath:
                render_href = _href_between_paths(page_path, run["root"] / render_relpath)

            run_entries.append(
                {
                    "id": run["id"],
                    "label": run["label"],
                    "group": run.get("group"),
                    "status": case_record.get("status") or detail_payload.get("status") or "unknown",
                    "metrics": detail_payload.get("metrics") or case_record.get("metrics") or {},
                    "detail_page_href": detail_page_href,
                    "edit_glb_href": edit_glb_href,
                    "render_href": render_href,
                }
            )

        page_path.write_text(
            _render_focus_case_page(
                benchmark_root=benchmark_root,
                case_id=case_id,
                dataset=dataset,
                object_name=object_name,
                prompt_id=prompt_id,
                prompt_text=prompt_text,
                run_entries=run_entries,
                reference_run_root=reference_run_root,
                reference_artifacts=reference_artifacts,
                page_path=page_path,
                focus_root=focus_root,
            ),
            encoding="utf-8",
        )
        case_pages.append(
            {
                "case_id": case_id,
                "prompt_text": prompt_text,
                "page_path": _bundle_relpath(focus_root, page_path),
                "run_entries": run_entries,
            }
        )
    return case_pages


def _render_focus_index(
    runs: list[dict[str, Any]],
    *,
    case_pages: list[dict[str, Any]],
    baseline_run_id: str | None,
) -> str:
    all_metric_names = _ordered_metric_names(
        {
            metric_name
            for item in runs
            for metric_name in item["summary"].get("metrics", {}).keys()
        }
    )
    metric_names = _table_metric_names(all_metric_names)

    baseline = None
    if baseline_run_id:
        baseline = next((item for item in runs if item["id"] == baseline_run_id), None)
    if baseline is None and runs:
        baseline = runs[0]

    summary_rank_classes = _build_metric_rank_classes(
        [item["summary"].get("metrics", {}) for item in runs],
        metric_names,
    )

    rows = []
    for row_index, item in enumerate(runs):
        metric_cells = []
        for metric in metric_names:
            value = item["summary"].get("metrics", {}).get(metric)
            baseline_value = baseline["summary"].get("metrics", {}).get(metric) if baseline else None
            rendered = _format_metric_value(_extract_scalar_metric(value))
            delta = _format_delta(value, baseline_value)
            class_name = _metric_table_class(metric, summary_rank_classes.get((row_index, metric)))
            class_attr = f" class=\"{html.escape(class_name)}\"" if class_name else ""
            metric_cells.append(
                f"<td{class_attr}><div>{html.escape(rendered)}</div><div class='delta'>{html.escape(delta)}</div></td>"
            )
        run_href = Path(item["path"]) / "index.html"
        rows.append(
            "<tr>"
            f"<td><a href=\"{html.escape(run_href.as_posix())}\">{html.escape(item['label'])}</a></td>"
            f"<td>{html.escape(item.get('group') or '')}</td>"
            + "".join(metric_cells)
            + "</tr>"
        )

    headers = "".join(
        f"<th class=\"{html.escape(_metric_table_class(metric_name))}\">{html.escape(_metric_label(metric_name))}</th>"
        for metric_name in metric_names
    )
    case_headers = "".join(f"<th>{html.escape(item['label'])}</th>" for item in runs)
    case_rows = []
    for case_page in case_pages:
        run_entry_map = {
            str(run_entry.get("id")): run_entry for run_entry in case_page.get("run_entries", [])
        }
        case_metric_names = _table_metric_names(
            {
                metric_name
                for run_entry in case_page.get("run_entries", [])
                for metric_name in (run_entry.get("metrics") or {}).keys()
            }
        )
        case_rank_classes = _build_metric_rank_classes(
            [
                ((run_entry_map.get(run["id"]) or {}).get("metrics") or {})
                for run in runs
            ],
            case_metric_names,
        )
        run_cells = []
        for run_index, run in enumerate(runs):
            run_entry = run_entry_map.get(run["id"])
            if run_entry is None:
                run_cells.append("<td class='muted'>-</td>")
                continue
            metric_parts = []
            for metric_name in case_metric_names:
                metric_value = run_entry.get("metrics", {}).get(metric_name)
                if metric_value is None:
                    continue
                class_name = _metric_table_class(
                    metric_name,
                    case_rank_classes.get((run_index, metric_name)),
                )
                metric_parts.append(
                    f"<span class='metric-pill {html.escape(class_name)}'><strong>{html.escape(_metric_label(metric_name))}</strong><em>{html.escape(_format_metric_value(metric_value))}</em></span>"
                )
            link_parts = []
            if run_entry.get("detail_page_href"):
                link_parts.append(f"<a href=\"{html.escape(run_entry['detail_page_href'])}\">run page</a>")
            if run_entry.get("edit_glb_href"):
                link_parts.append(
                    f"<a href=\"{html.escape(run_entry['edit_glb_href'])}\" target=\"_blank\" rel=\"noopener\">edit.glb</a>"
                )
            run_cells.append(
                "<td>"
                f"<div><strong>{html.escape(str(run_entry.get('status') or 'unknown'))}</strong></div>"
                f"<div class='metric-stack'>{''.join(metric_parts)}</div>"
                f"<div class='case-links'>{' · '.join(link_parts)}</div>"
                "</td>"
            )
        case_rows.append(
            "<tr>"
            f"<td><a href=\"{html.escape(case_page['page_path'])}\">{html.escape(case_page['case_id'])}</a></td>"
            f"<td>{html.escape(case_page.get('prompt_text') or '')}</td>"
            + "".join(run_cells)
            + "</tr>"
        )
    return f'''<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Benchmark Focus</title>
  <style>
    body {{ font-family: "Segoe UI", sans-serif; margin: 0; background: #f7f7f7; color: #222; }}
    .page {{ width: min(1680px, calc(100vw - 32px)); margin: 24px auto 48px; }}
    .panel {{ background: #fff; border: 1px solid #ddd; border-radius: 14px; padding: 18px 20px; }}
    .panel + .panel {{ margin-top: 18px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 10px 8px; border-bottom: 1px solid #eee; text-align: left; font-size: 14px; vertical-align: top; }}
    th {{ background: #fafafa; }}
    a {{ color: #0b57d0; text-decoration: none; }}
    .delta {{ color: #666; font-size: 12px; margin-top: 2px; }}
    .muted {{ color: #666; }}
    .case-links {{ margin-top: 8px; font-size: 12px; }}
    {_comparison_table_css()}
  </style>
</head>
<body>
  <div class="page">
    <section class="panel">
      <h1>Benchmark Focus</h1>
      <p>baseline: {html.escape(baseline['label'] if baseline else '')}</p>
      <p class="heatmap-note">Summary table ranks each metric column across runs. Case cells keep the same ranking logic per metric.</p>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>run</th>
              <th>group</th>
              {headers}
            </tr>
          </thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
      </div>
    </section>
    <section class="panel">
      <h2>Cases</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>case</th>
              <th>prompt</th>
              {case_headers}
            </tr>
          </thead>
          <tbody>
            {''.join(case_rows)}
          </tbody>
        </table>
      </div>
    </section>
  </div>
</body>
</html>
'''


def _render_focus_case_page(
    *,
    benchmark_root: Path,
    case_id: str,
    dataset: str,
    object_name: str,
    prompt_id: int,
    prompt_text: str,
    run_entries: list[dict[str, Any]],
    reference_run_root: Path | None,
    reference_artifacts: dict[str, Any],
    page_path: Path,
    focus_root: Path,
) -> str:
    prompt_key = f"prompt_{prompt_id}"
    data_root = benchmark_root / "edit3d_data" / "data"
    source_model_href = _href_between_paths(
        page_path,
        data_root / dataset / object_name / "source_model" / "model.glb",
    )
    mask_glb_href = _href_between_paths(
        page_path,
        data_root / dataset / object_name / prompt_key / "3d_edit_region.glb",
    )
    source_image_href = _href_between_paths(
        page_path,
        data_root / dataset / object_name / prompt_key / "2d_render.png",
    )
    edit_image_href = _href_between_paths(
        page_path,
        data_root / dataset / object_name / prompt_key / "2d_edit.png",
    )
    mask_image_href = _href_between_paths(
        page_path,
        data_root / dataset / object_name / prompt_key / "2d_mask.png",
    )

    if reference_run_root is not None and reference_artifacts:
        if source_model_href is None:
            relpath = reference_artifacts.get("source_model_glb")
            if relpath:
                source_model_href = _href_between_paths(page_path, reference_run_root / str(relpath))
        if mask_glb_href is None:
            relpath = reference_artifacts.get("mask_glb")
            if relpath:
                mask_glb_href = _href_between_paths(page_path, reference_run_root / str(relpath))
        if source_image_href is None:
            relpath = reference_artifacts.get("source_image")
            if relpath:
                source_image_href = _href_between_paths(page_path, reference_run_root / str(relpath))
        if edit_image_href is None:
            relpath = reference_artifacts.get("edit_image")
            if relpath:
                edit_image_href = _href_between_paths(page_path, reference_run_root / str(relpath))
        if mask_image_href is None:
            relpath = reference_artifacts.get("mask_image")
            if relpath:
                mask_image_href = _href_between_paths(page_path, reference_run_root / str(relpath))

    back_href = _href_between_paths(page_path, focus_root / "index.html") or "../../index.html"

    image_cards = []
    for label, href in (
        ("source_image", source_image_href),
        ("edit_image", edit_image_href),
        ("mask_image", mask_image_href),
    ):
        if href is None:
            continue
        image_cards.append(
            f"<section class='image-card'><h3>{html.escape(label)}</h3><img src=\"{html.escape(href)}\" alt=\"{html.escape(label)}\"></section>"
        )

    run_sections = []
    viewer_specs = []
    for index, run_entry in enumerate(run_entries):
        source_id = f"viewer-source-{index}"
        combined_id = f"viewer-combined-{index}"
        edit_id = f"viewer-edit-{index}"
        viewer_specs.append(
            {
                "sourceId": source_id,
                "combinedId": combined_id,
                "editId": edit_id,
                "sourceHref": source_model_href,
                "maskHref": mask_glb_href,
                "editHref": run_entry.get("edit_glb_href"),
            }
        )
        metric_cards = _render_metric_sections(
            run_entry.get("metrics", {}),
            hide_none=True,
            empty_message="No per-case metrics are available for this run under the current protocol.",
        )
        links = []
        if run_entry.get("detail_page_href"):
            links.append(f"<a href=\"{html.escape(run_entry['detail_page_href'])}\">run page</a>")
        if run_entry.get("edit_glb_href"):
            links.append(
                f"<a href=\"{html.escape(run_entry['edit_glb_href'])}\" target=\"_blank\" rel=\"noopener\">edit.glb</a>"
            )
        if run_entry.get("render_href"):
            links.append(
                f"<a href=\"{html.escape(run_entry['render_href'])}\" target=\"_blank\" rel=\"noopener\">renders</a>"
            )
        run_sections.append(
            "<section class='run-card'>"
            f"<div class='run-header'><div><h2>{html.escape(run_entry['label'])}</h2><p class='muted'>{html.escape(str(run_entry.get('group') or ''))}</p></div><div class='status'>{html.escape(str(run_entry.get('status') or 'unknown'))}</div></div>"
            "<div class='viewer-grid'>"
            f"<section class='viewer-card'><h3>Source Model</h3><div id=\"{html.escape(source_id)}\" class='viewer-3d'></div></section>"
            f"<section class='viewer-card'><h3>Source + Mask</h3><div id=\"{html.escape(combined_id)}\" class='viewer-3d'></div></section>"
            f"<section class='viewer-card'><h3>Final Edit</h3><div id=\"{html.escape(edit_id)}\" class='viewer-3d'></div></section>"
            "</div>"
            f"<div class='metric-sections'>{metric_cards}</div>"
            f"<div class='run-links'>{' · '.join(links)}</div>"
            "</section>"
        )

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(case_id)}</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three/examples/js/loaders/GLTFLoader.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three/examples/js/controls/OrbitControls.js"></script>
  <style>
    body {{ font-family: "Segoe UI", sans-serif; margin: 0; background: #f7f7f7; color: #222; }}
    .page {{ width: min(1500px, calc(100vw - 32px)); margin: 24px auto 48px; }}
    .panel {{ background: #fff; border: 1px solid #ddd; border-radius: 14px; padding: 18px 20px; }}
    .panel + .panel {{ margin-top: 18px; }}
    .image-grid, .metric-grid, .viewer-grid, .metric-sections {{ display: grid; gap: 12px; }}
    .image-grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .viewer-grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); margin-top: 12px; }}
    .metric-grid {{ grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); margin-top: 12px; }}
    .metric-sections {{ margin-top: 12px; }}
    .run-card, .viewer-card, .metric {{ background: #fafafa; border: 1px solid #e5e5e5; border-radius: 12px; }}
    .run-card {{ padding: 14px; }}
    .viewer-card {{ padding: 12px; }}
    .metric {{ padding: 12px 14px; }}
    .metric strong {{ display: block; font-size: 18px; margin-top: 4px; }}
    .metric small {{ display: block; margin-top: 6px; color: #666; font-size: 12px; }}
    .metric-family + .metric-family {{ margin-top: 22px; }}
    .metric-family-head {{ margin-bottom: 12px; }}
    .metric-family-head h3 {{ margin: 0 0 4px; }}
    .metric-family-head p {{ margin: 0; }}
    .metric-group + .metric-group {{ margin-top: 18px; }}
    .metric-spotlight {{ margin-bottom: 14px; }}
    .metric-primary {{ border-color: #cbd5e1; background: #f8fbff; }}
    .metric-group-head {{ margin-bottom: 10px; }}
    .metric-group-head h3 {{ margin: 0 0 4px; }}
    .metric-group-head p {{ margin: 0; }}
    .metric-more {{ margin-top: 12px; }}
    .metric-more > summary {{ cursor: pointer; color: #0b57d0; }}
    .viewer-3d {{ width: 100%; height: 300px; border: 2px solid #e2e8f0; border-radius: 12px; background: #f8fafc; overflow: hidden; }}
    .run-header {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; }}
    .status {{ font-weight: 600; }}
    .run-links {{ margin-top: 12px; font-size: 13px; }}
    .muted {{ color: #666; }}
    img {{ width: 100%; border-radius: 12px; border: 1px solid #ddd; background: #fff; }}
    a {{ color: #0b57d0; text-decoration: none; }}
    h1, h2, h3, p {{ margin: 0; }}
    h1 {{ margin-top: 8px; }}
    h2, h3 {{ margin-bottom: 10px; }}
    @media (max-width: 1080px) {{ .viewer-grid, .image-grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="page">
    <section class="panel">
      <p><a href="{html.escape(back_href)}">Back</a></p>
      <h1>{html.escape(case_id)}</h1>
      <p class="muted" style="margin-top: 8px;">{html.escape(prompt_text)}</p>
    </section>
    <section class="panel">
      <h2>Input</h2>
      <div class="image-grid">{''.join(image_cards)}</div>
    </section>
    <section class="panel">
      <h2>Run Compare</h2>
      {''.join(run_sections)}
    </section>
  </div>
  <script>
    const viewerSpecs = {json.dumps(viewer_specs)};

    function createViewer(containerId) {{
      const container = document.getElementById(containerId);
      if (!container) return null;
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0xf8fafc);
      const camera = new THREE.PerspectiveCamera(75, container.clientWidth / Math.max(container.clientHeight, 1), 0.1, 1000);
      const renderer = new THREE.WebGLRenderer({{ antialias: true }});
      renderer.setSize(container.clientWidth, container.clientHeight);
      renderer.setPixelRatio(window.devicePixelRatio || 1);
      renderer.setClearColor(0xffffff);
      container.appendChild(renderer.domElement);

      renderer.outputEncoding = THREE.sRGBEncoding;
      renderer.physicallyCorrectLights = true;

      const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
      directionalLight.position.set(5, 10, 7);
      scene.add(directionalLight);

      const lightIntensity = 30;
      const lightDistance = 100;
      const directions = [
        [10, 0, 0], [-10, 0, 0], [0, 10, 0], [0, -10, 0], [0, 0, 10], [0, 0, -10],
      ];
      directions.forEach((dir, index) => {{
        const pointLight = new THREE.PointLight(0xffffff, lightIntensity, lightDistance);
        pointLight.position.set(...dir);
        pointLight.castShadow = true;
        pointLight.name = `PointLight_${{index}}`;
        scene.add(pointLight);
      }});

      const controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;

      function frameObject(object) {{
        const box = new THREE.Box3().setFromObject(object);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z) || 1;
        camera.position.copy(center);
        camera.position.z += maxDim * 1.2;
        controls.target.copy(center);
        controls.update();
      }}

      function animate() {{
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
      }}
      animate();

      const resizeObserver = new ResizeObserver(() => {{
        const width = container.clientWidth;
        const height = Math.max(container.clientHeight, 1);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);
      }});
      resizeObserver.observe(container);

      return {{ scene, frameObject }};
    }}

    function loadGlb(sceneBundle, href, materialPatch) {{
      return new Promise((resolve, reject) => {{
        if (!sceneBundle || !href) {{
          resolve(null);
          return;
        }}
        const loader = new THREE.GLTFLoader();
        loader.load(
          href,
          (gltf) => {{
            const model = gltf.scene;
            if (materialPatch) {{
              model.traverse((node) => {{
                if (node.isMesh) {{
                  materialPatch(node);
                }}
              }});
            }}
            sceneBundle.scene.add(model);
            resolve(model);
          }},
          undefined,
          reject
        );
      }});
    }}

    async function initViewerSet(spec) {{
      const sourceViewer = createViewer(spec.sourceId);
      const combinedViewer = createViewer(spec.combinedId);
      const editViewer = createViewer(spec.editId);

      const sourceModel = await loadGlb(sourceViewer, spec.sourceHref);
      if (sourceModel) {{
        sourceViewer.frameObject(sourceModel);
      }}

      let combinedFrameTarget = null;
      const combinedSource = await loadGlb(combinedViewer, spec.sourceHref);
      if (combinedSource) {{
        combinedFrameTarget = combinedSource;
      }}
      await loadGlb(combinedViewer, spec.maskHref, (node) => {{
        node.material = new THREE.MeshPhongMaterial({{
          color: 0xcccccc,
          transparent: true,
          opacity: 0.7,
        }});
      }});
      if (combinedFrameTarget) {{
        combinedViewer.frameObject(combinedFrameTarget);
      }}

      const editModel = await loadGlb(editViewer, spec.editHref);
      if (editModel) {{
        editViewer.frameObject(editModel);
      }}
    }}

    async function initAllViewers() {{
      for (const spec of viewerSpecs) {{
        try {{
          await initViewerSet(spec);
        }} catch (error) {{
          console.error("Failed to initialize viewer set", spec, error);
        }}
      }}
    }}

    initAllViewers();
  </script>
</body>
</html>
"""


def _format_metric_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _extract_scalar_metric(value: Any) -> float | None:
    if isinstance(value, dict):
        if "value" in value and value.get("value") is not None:
            return float(value["value"])
        if "mean" in value and value.get("mean") is not None:
            return float(value["mean"])
        return None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_delta(current: Any, baseline: Any) -> str:
    current_scalar = _extract_scalar_metric(current)
    baseline_scalar = _extract_scalar_metric(baseline)
    if current_scalar is None or baseline_scalar is None:
        return ""
    delta = current_scalar - baseline_scalar
    return f"{delta:+.4f}"
