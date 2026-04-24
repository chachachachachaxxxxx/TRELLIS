from __future__ import annotations

import html
import json
import math
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from trellis_edit.common import ensure_dir, write_json


DEFAULT_BENCHMARK_ROOT = Path("/cache/wangxinxing/data/trellis_edit_benchmark")
TEXT_ALIGNMENT_VIEW_IDS = {"0000", "0001", "0007", "0008", "0009", "0015"}
SHOWCASE_VIEW_SPECS = (
    ("front", "render_0000.png"),
    ("right", "render_0004.png"),
    ("back", "render_0008.png"),
    ("left", "render_0012.png"),
)
DEFAULT_RUN_GROUP_LABEL = "ungrouped"
EXPERIMENT_CONFIG_FILENAMES = (
    "_batch_source_config.yaml",
    "_batch_source_config.yml",
    "_batch_source_config.json",
    "experiment_config.yaml",
    "experiment_config.yml",
    "experiment_config.json",
    "config.yaml",
    "config.yml",
    "config.json",
)
EXPERIMENT_CONFIG_SOURCE_KEYS = (
    "experiment_config_source_path",
    "config_source_path",
    "config_path",
    "source_config_path",
    "source_config",
)
METRIC_FAMILY_ORDER = (
    "Legacy Metrics",
    "Multi-View Metrics",
    "Geometry Edit Metrics",
)
METRIC_FAMILY_META = {
    "Legacy Metrics": {"description": "Original Edit3D-Bench metrics kept for backward comparison."},
    "Multi-View Metrics": {"description": "New metrics tailored for multi-view editing fidelity and stability."},
    "Geometry Edit Metrics": {"description": "Geometry-only edit evaluation with gray/normal renders and mask-outside preservation."},
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
    "Geometry Edit Metrics": (
        "Edit Alignment",
        "Local Preservation",
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
    "clipi_mv",
    "uni3d",
    "cd",
)
LEADING_METRIC_NAMES = (
    "lpips_in_novel",
    "ssim_in_novel",
    "dino_if_novel",
    "dino_if_max",
    "dino_if_mean",
    "clip_t",
    "clipi_mv",
    "uni3d",
    "cd",
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
    "Edit Alignment": {"description": "Whether the geometry-only edit matches the target edited image under gray or normal rendering."},
    "Local Preservation": {"description": "Whether the geometry outside the 3D edit mask remains close to the source model."},
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
    "floaters": {"label": "Floaters", "family": "Geometry Edit Metrics", "group": "Local Preservation", "order": 6},
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
    "clipi": {"label": "CLIP-I", "family": "Geometry Edit Metrics", "group": "Edit Alignment", "order": 0},
    "clipi_mv": {"label": "CLIP-I-MV", "family": "Geometry Edit Metrics", "group": "Edit Alignment", "order": 1},
    "clipi_n": {"label": "CLIP-I-N", "family": "Geometry Edit Metrics", "group": "Edit Alignment", "order": 2},
    "clipi_mv_n": {"label": "CLIP-I-MV-N", "family": "Geometry Edit Metrics", "group": "Edit Alignment", "order": 3},
    "uni3d": {"label": "Uni3D", "family": "Geometry Edit Metrics", "group": "Local Preservation", "order": 4},
    "cd": {"label": "CD", "family": "Geometry Edit Metrics", "group": "Local Preservation", "order": 5},
}


def build_run_id(entrypoint_name: str, config_name: str, timestamp: str) -> str:
    return "_".join((_slug(entrypoint_name), _slug(config_name), timestamp))


def _slug(value: str) -> str:
    text = value.replace("/", "_").replace(" ", "_").strip("_")
    return "".join(ch for ch in text if ch.isalnum() or ch in {"_", "-", "."}) or "run"


def _normalize_run_group(value: Any) -> str:
    if value is None:
        return DEFAULT_RUN_GROUP_LABEL
    text = str(value).strip()
    return text or DEFAULT_RUN_GROUP_LABEL


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


def _natural_name_parts(value: str) -> tuple[tuple[int, Any], ...]:
    parts: list[tuple[int, Any]] = []
    for token in re.findall(r"[a-zA-Z]+|\d+", value.lower()):
        if token.isdigit():
            parts.append((0, int(token)))
        else:
            parts.append((1, token))
    return tuple(parts)


def _run_name_sort_key(run: dict[str, Any]) -> tuple[Any, ...]:
    label = str(run.get("label") or run.get("id") or run.get("path") or "")
    run_id = str(run.get("id") or "")
    match = re.match(r"^(.+?)_(autoencode|edit)_(.+)$", label)
    if match is None:
        return (_natural_name_parts(label), 99, (), label.lower(), run_id.lower())

    method, action, suffix = match.groups()
    action_order = {"autoencode": 0, "edit": 1}.get(action, 99)
    return (
        _natural_name_parts(method),
        action_order,
        _natural_name_parts(suffix),
        label.lower(),
        run_id.lower(),
    )


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
        "floaters",
        "cross_seed_novel_lpips",
        "cd",
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


def _resolve_pred_root_from_manifest(manifest_payload: dict[str, Any]) -> Path | None:
    pred_root_text = str(manifest_payload.get("pred_root") or "").strip()
    if not pred_root_text:
        return None
    return Path(pred_root_text).expanduser().resolve()


def _resolve_experiment_config_candidate(
    value: Any,
    *,
    pred_root: Path | None,
) -> Path | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        if pred_root is None:
            return None
        path = (pred_root / path).resolve()
    else:
        path = path.resolve()
    return path if path.is_file() else None


def _iter_experiment_config_sources(
    *,
    pred_root: Path | None,
    manifest_payload: dict[str, Any],
) -> list[Path]:
    sources: list[Path] = []
    seen: set[str] = set()

    def add(path: Path | None) -> None:
        if path is None:
            return
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        sources.append(path)

    for key in EXPERIMENT_CONFIG_SOURCE_KEYS:
        add(_resolve_experiment_config_candidate(manifest_payload.get(key), pred_root=pred_root))

    if pred_root is not None:
        pred_root_manifest = _load_json(pred_root / "manifest.json")
        for key in EXPERIMENT_CONFIG_SOURCE_KEYS:
            add(_resolve_experiment_config_candidate(pred_root_manifest.get(key), pred_root=pred_root))
        for filename in EXPERIMENT_CONFIG_FILENAMES:
            candidate = pred_root / filename
            add(candidate if candidate.is_file() else None)

    return sources


def _ensure_experiment_config_page(
    *,
    bundle_root: Path,
    manifest_payload: dict[str, Any],
) -> None:
    manifest_path = bundle_root / "manifest.json"
    updated = False

    config_relpath = str(manifest_payload.get("experiment_config_path") or "").strip()
    config_path = bundle_root / config_relpath if config_relpath else None
    if config_path is not None and not config_path.is_file():
        config_path = None
        config_relpath = ""

    if config_path is None:
        pred_root = _resolve_pred_root_from_manifest(manifest_payload)
        for source_path in _iter_experiment_config_sources(
            pred_root=pred_root,
            manifest_payload=manifest_payload,
        ):
            bundled_config_dir = ensure_dir(bundle_root / "config")
            bundled_config_path = bundled_config_dir / source_path.name
            if not bundled_config_path.is_file():
                _link_or_copy(source_path, bundled_config_path)
            config_path = bundled_config_path
            config_relpath = _bundle_relpath(bundle_root, bundled_config_path)
            manifest_payload["experiment_config_path"] = config_relpath
            manifest_payload["experiment_config_source_path"] = str(source_path)
            manifest_payload["experiment_config_label"] = source_path.name
            updated = True
            break

    if config_path is None:
        return

    if not str(manifest_payload.get("experiment_config_label") or "").strip():
        manifest_payload["experiment_config_label"] = config_path.name
        updated = True

    page_relpath = str(manifest_payload.get("experiment_config_page_path") or "").strip()
    if not page_relpath:
        page_relpath = "config.html"
        manifest_payload["experiment_config_page_path"] = page_relpath
        updated = True
    page_path = bundle_root / page_relpath
    page_path.write_text(
        _render_experiment_config_page(
            bundle_root=bundle_root,
            page_path=page_path,
            manifest_payload=manifest_payload,
            config_path=config_path,
        ),
        encoding="utf-8",
    )

    if updated:
        write_json(manifest_path, manifest_payload)


def _render_experiment_config_page(
    *,
    bundle_root: Path,
    page_path: Path,
    manifest_payload: dict[str, Any],
    config_path: Path,
) -> str:
    run_href = _relative_href(page_path, bundle_root / "index.html")
    raw_href = _relative_href(page_path, config_path)
    label = str(manifest_payload.get("experiment_config_label") or config_path.name)
    run_label = str(manifest_payload.get("config_name") or manifest_payload.get("run_id") or bundle_root.name)
    source_path = str(manifest_payload.get("experiment_config_source_path") or "").strip()
    format_label = config_path.suffix.lstrip(".").upper() or "TEXT"
    content = config_path.read_text(encoding="utf-8", errors="replace")
    source_line = (
        f"<p class='muted source-path'>source: {html.escape(source_path)}</p>"
        if source_path
        else ""
    )
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(run_label)} / Experiment Config</title>
  <style>
    body {{ font-family: "Segoe UI", sans-serif; margin: 0; background: #f7f7f7; color: #222; }}
    .page {{ width: min(1480px, calc(100vw - 32px)); margin: 24px auto 48px; }}
    .hero, .panel {{ background: #fff; border: 1px solid #ddd; border-radius: 14px; padding: 18px 20px; }}
    .hero {{ margin-bottom: 18px; }}
    .nav-links {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 14px; }}
    .nav-link {{ display: inline-flex; align-items: center; padding: 8px 12px; border-radius: 999px; border: 1px solid #d9d9d9; background: #fafafa; color: #222; text-decoration: none; font-size: 13px; }}
    .meta-row {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 14px; }}
    .meta {{ background: #fafafa; border: 1px solid #e5e5e5; border-radius: 12px; padding: 12px 14px; }}
    .meta strong {{ display: block; font-size: 18px; margin-top: 4px; }}
    .code-wrap {{ overflow: auto; border-radius: 12px; border: 1px solid #e5e7eb; background: #0f172a; }}
    pre {{ margin: 0; padding: 18px; color: #e2e8f0; font-size: 13px; line-height: 1.6; font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace; }}
    a {{ color: #0b57d0; text-decoration: none; }}
    h1, h2, p {{ margin: 0; }}
    .muted {{ color: #666; }}
    .source-path {{ margin-top: 10px; word-break: break-all; }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="nav-links">
        <a class="nav-link" href="{html.escape(run_href)}">Back To Run</a>
        <a class="nav-link" href="{html.escape(raw_href)}" target="_blank" rel="noopener">Open Raw File</a>
      </div>
      <h1>{html.escape(run_label)}</h1>
      <p class="muted" style="margin-top: 8px;">Experiment config preview</p>
      <div class="meta-row">
        <div class="meta"><span>file</span><strong>{html.escape(label)}</strong></div>
        <div class="meta"><span>format</span><strong>{html.escape(format_label)}</strong></div>
      </div>
      {source_line}
    </section>
    <section class="panel">
      <h2 style="margin-bottom: 12px;">Config Content</h2>
      <div class="code-wrap"><pre>{html.escape(content)}</pre></div>
    </section>
  </div>
</body>
</html>
"""


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
            "<div class='metric-group-head'><h3>Priority Metrics</h3><p class='muted'>Headline metrics across single-view, multi-view, and geometry tasks are shown first.</p></div>"
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

__all__ = ['DEFAULT_BENCHMARK_ROOT', 'TEXT_ALIGNMENT_VIEW_IDS', 'SHOWCASE_VIEW_SPECS', 'DEFAULT_RUN_GROUP_LABEL', 'EXPERIMENT_CONFIG_FILENAMES', 'EXPERIMENT_CONFIG_SOURCE_KEYS', 'METRIC_FAMILY_ORDER', 'METRIC_FAMILY_META', 'METRIC_GROUP_ORDER_BY_FAMILY', 'PRIMARY_METRIC_NAMES', 'LEADING_METRIC_NAMES', 'METRIC_GROUP_META', 'METRIC_META', 'build_run_id', '_slug', '_normalize_run_group', '_load_json', '_load_jsonl', '_resolve_daily_run_group', '_load_daily_runs', '_ordered_run_groups', '_natural_name_parts', '_run_name_sort_key', '_summarize_statuses', '_write_jsonl', '_bundle_relpath', '_link_or_copy', '_relative_href', '_href_from_page', '_href_between_paths', '_ordered_metric_names', '_metric_label', '_metric_prefers_higher', '_metric_column_classes', '_metric_table_class', '_build_metric_rank_classes', '_comparison_table_css', '_primary_metric_names', '_secondary_metric_names', '_table_metric_names', '_collect_case_metric_names', '_resolve_pred_root_from_manifest', '_resolve_experiment_config_candidate', '_iter_experiment_config_sources', '_ensure_experiment_config_page', '_render_experiment_config_page', '_format_summary_metric_value', '_format_metric_meta_line', '_render_metric_cards', '_group_metric_items', '_render_metric_sections', '_render_metric_details', '_format_metric_value', '_extract_scalar_metric', '_format_delta']
