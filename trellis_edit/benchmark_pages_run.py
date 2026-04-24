from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from trellis_edit.common import ensure_dir

from .benchmark_pages_common import (
    _build_metric_rank_classes,
    _collect_case_metric_names,
    _comparison_table_css,
    _ensure_experiment_config_page,
    _format_metric_value,
    _format_summary_metric_value,
    _href_between_paths,
    _href_from_page,
    _metric_label,
    _metric_table_class,
    _ordered_metric_names,
    _relative_href,
    _render_metric_sections,
    _slug,
    _table_metric_names,
)
from .benchmark_pages_shared import (
    _case_prompt_label,
    _case_record_page_path,
    _load_case_payload,
    _run_dataset_gallery_page_path,
)

def _render_daily_run_pages(
    *,
    bundle_root: Path,
    manifest_payload: dict[str, Any],
    summary_payload: dict[str, Any],
    case_records: list[dict[str, Any]],
) -> None:
    _ensure_experiment_config_page(
        bundle_root=bundle_root,
        manifest_payload=manifest_payload,
    )
    run_label = str(manifest_payload.get("config_name") or manifest_payload.get("run_id") or bundle_root.name)
    run_id = str(manifest_payload.get("run_id") or bundle_root.name)

    for case_record in case_records:
        page_path = _case_record_page_path(bundle_root, case_record)
        case_payload = _load_case_payload(bundle_root, case_record)
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

    dataset_map: dict[str, list[dict[str, Any]]] = {}
    for case_record in case_records:
        dataset = str(case_record.get("dataset") or "").strip()
        if not dataset:
            continue
        dataset_map.setdefault(dataset, []).append(case_record)
    for dataset, dataset_case_records in dataset_map.items():
        page_path = _run_dataset_gallery_page_path(bundle_root, dataset)
        ensure_dir(page_path.parent)
        page_path.write_text(
            _render_run_dataset_gallery_page(
                bundle_root=bundle_root,
                page_path=page_path,
                run_label=run_label,
                run_id=run_id,
                dataset=dataset,
                case_records=dataset_case_records,
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
        object_name = str(record.get("object_name") or "").strip()
        prompt_label = _case_prompt_label(record)
        case_entries.append(
            {
                "index": index,
                "case_id": str(record.get("case_id") or ""),
                "object_name": object_name,
                "prompt_id": record.get("prompt_id"),
                "prompt_label": prompt_label,
                "status": str(record.get("status") or ""),
                "href": _relative_href(page_path, _case_record_page_path(bundle_root, record)),
                "search_text": " ".join(
                    (
                        str(record.get("case_id") or ""),
                        object_name,
                        prompt_label,
                        str(record.get("sample_name") or ""),
                        str(record.get("display_name") or ""),
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


def _build_run_dataset_entries(
    *,
    bundle_root: Path,
    page_path: Path,
    case_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    dataset_map: dict[str, list[dict[str, Any]]] = {}
    for record in case_records:
        dataset = str(record.get("dataset") or "").strip()
        if not dataset:
            continue
        dataset_map.setdefault(dataset, []).append(record)

    entries: list[dict[str, Any]] = []
    for dataset, records in sorted(dataset_map.items()):
        sorted_records = sorted(
            records,
            key=lambda item: (
                str(item.get("object_name") or ""),
                int(item.get("prompt_id") or 0),
                str(item.get("case_id") or ""),
            ),
        )
        entries.append(
            {
                "dataset": dataset,
                "count": len(sorted_records),
                "gallery_href": _relative_href(
                    page_path,
                    _run_dataset_gallery_page_path(bundle_root, dataset),
                ),
                "first_case_href": _relative_href(
                    page_path,
                    _case_record_page_path(bundle_root, sorted_records[0]),
                ),
            }
        )
    return entries


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
    dataset_entries = _build_run_dataset_entries(
        bundle_root=bundle_root,
        page_path=bundle_root / "index.html",
        case_records=case_records,
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
    dataset_card_items = []
    for item in dataset_entries:
        first_case_link = ""
        if item.get("first_case_href"):
            first_case_link = (
                f"<a href=\"{html.escape(str(item['first_case_href'] or '#'))}\">Open First Case</a>"
            )
        dataset_card_items.append(
            "<article class='dataset-card'>"
            f"<h3>{html.escape(item['dataset'])}</h3>"
            f"<p class='muted'>{item['count']} case(s)</p>"
            "<div class='dataset-card-links'>"
            f"<a href=\"{html.escape(str(item['gallery_href'] or '#'))}\">Browse Visuals</a>"
            f"{first_case_link}"
            "</div>"
            "</article>"
        )
    dataset_cards = "".join(dataset_card_items)
    dataset_section = (
        "<section class=\"panel\" style=\"margin-bottom: 18px;\">"
        "<h2>Datasets</h2>"
        "<p class='muted' style='margin-top: 8px;'>Open a dataset gallery to browse all packaged sample visualizations for this run.</p>"
        f"<div class='dataset-grid'>{dataset_cards}</div>"
        "</section>"
        if dataset_entries
        else ""
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
    .dataset-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }}
    .dataset-card {{ background: #fafafa; border: 1px solid #e5e5e5; border-radius: 12px; padding: 14px; }}
    .dataset-card h3 {{ margin: 0 0 6px; }}
    .dataset-card-links {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 12px; font-size: 13px; }}
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
    {dataset_section}
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

def _render_run_dataset_gallery_page(
    *,
    bundle_root: Path,
    page_path: Path,
    run_label: str,
    run_id: str,
    dataset: str,
    case_records: list[dict[str, Any]],
) -> str:
    sorted_records = sorted(
        case_records,
        key=lambda item: (
            str(item.get("object_name") or ""),
            int(item.get("prompt_id") or 0),
            str(item.get("case_id") or ""),
        ),
    )
    cards: list[str] = []
    for case_record in sorted_records:
        case_payload = _load_case_payload(bundle_root, case_record)
        artifacts = case_payload.get("artifacts") or {}
        image_cards = []
        for label, key in (
            ("source", "source_image"),
            ("edit", "edit_image"),
            ("result", "edit_front_image"),
        ):
            href = _href_from_page(page_path, bundle_root, artifacts.get(key))
            if href is None:
                continue
            image_cards.append(
                "<figure class='gallery-thumb'>"
                f"<div class='gallery-label'>{html.escape(label)}</div>"
                f"<img src=\"{html.escape(href)}\" alt=\"{html.escape(label)}\" loading=\"lazy\">"
                "</figure>"
            )
        links = []
        detail_href = _relative_href(page_path, _case_record_page_path(bundle_root, case_record))
        if detail_href:
            links.append(f"<a href=\"{html.escape(detail_href)}\">case page</a>")
        renders_href = _href_from_page(page_path, bundle_root, artifacts.get("images"))
        if renders_href:
            links.append(f"<a href=\"{html.escape(renders_href)}\" target=\"_blank\" rel=\"noopener\">renders</a>")
        edit_glb_href = _href_from_page(page_path, bundle_root, artifacts.get("edit_glb"))
        if edit_glb_href:
            links.append(f"<a href=\"{html.escape(edit_glb_href)}\" target=\"_blank\" rel=\"noopener\">edit.glb</a>")
        prompt_text = str(case_payload.get("prompt_text") or "").strip()
        status = str(case_payload.get("status") or case_record.get("status") or "unknown")
        prompt_line = ""
        if prompt_text:
            prompt_line = f"<p class='muted small'>{html.escape(prompt_text)}</p>"
        image_grid_html = "".join(image_cards) or "<p class='muted'>No packaged images.</p>"
        cards.append(
            "<article class='gallery-card'>"
            "<div class='gallery-card-head'>"
            "<div>"
            f"<h2>{html.escape(str(case_payload.get('object_name') or ''))}</h2>"
            f"<p class='muted'>{html.escape(str(case_payload.get('case_id') or ''))}</p>"
            f"{prompt_line}"
            "</div>"
            f"<div class='status'>{html.escape(status)}</div>"
            "</div>"
            f"<div class='gallery-grid'>{image_grid_html}</div>"
            f"<div class='gallery-links'>{' · '.join(links)}</div>"
            "</article>"
        )

    back_href = _relative_href(page_path, bundle_root / "index.html") or "../index.html"
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(run_id)} / {html.escape(dataset)}</title>
  <style>
    body {{ font-family: "Segoe UI", sans-serif; margin: 0; background: #f7f7f7; color: #222; }}
    .page {{ width: min(1680px, calc(100vw - 32px)); margin: 24px auto 48px; }}
    .hero, .panel, .gallery-card {{ background: #fff; border: 1px solid #ddd; border-radius: 14px; }}
    .hero {{ padding: 18px 20px; margin-bottom: 18px; }}
    .gallery-list {{ display: grid; gap: 16px; }}
    .gallery-card {{ padding: 16px; }}
    .gallery-card-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; margin-bottom: 14px; }}
    .gallery-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
    .gallery-thumb {{ margin: 0; }}
    .gallery-label {{ font-size: 12px; color: #666; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.04em; }}
    .gallery-thumb img {{ width: 100%; aspect-ratio: 1 / 1; object-fit: cover; border-radius: 10px; border: 1px solid #ddd; background: #fff; }}
    .gallery-links {{ margin-top: 12px; font-size: 13px; }}
    .status {{ font-weight: 600; }}
    a {{ color: #0b57d0; text-decoration: none; }}
    h1, h2, p {{ margin: 0; }}
    .muted {{ color: #666; }}
    .small {{ font-size: 12px; margin-top: 4px; }}
    @media (max-width: 960px) {{ .gallery-grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <p><a href="{html.escape(back_href)}">Back To Run</a></p>
      <h1>{html.escape(run_label)} / {html.escape(dataset)}</h1>
      <p class="muted" style="margin-top: 8px;">Browse all packaged sample visualizations for this dataset.</p>
    </section>
    <section class="panel" style="padding: 18px 20px;">
      <div class="gallery-list">{''.join(cards)}</div>
    </section>
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

    image_title_map = {
        "source_image": "source_image",
        "edit_image": "edit_image",
        "mask_image": "mask_image",
        "edit_front_image": "edit_front_view",
    }
    image_blocks = []
    for key in ("source_image", "edit_image", "mask_image", "edit_front_image"):
        bundle_relative_path = case_payload.get("artifacts", {}).get(key)
        href = _href_from_page(page_path, bundle_root, bundle_relative_path)
        if href is None:
            continue
        title = image_title_map.get(key, key)
        image_blocks.append(
            f"<section class='image-card'><h2>{html.escape(title)}</h2><img src=\"{html.escape(href)}\" alt=\"{html.escape(title)}\"></section>"
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
    prompt_label = _case_prompt_label(case_payload)
    prompt_text = str(case_payload.get("prompt_text") or "").strip()
    sample_title = str(case_payload.get("display_name") or "").strip() or object_name or case_payload["case_id"]
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
    combined_available = source_model_href is not None or mask_glb_href is not None
    source_model_button_label = "Load" if source_model_href is not None else "Unavailable"
    source_model_status = "Loads on demand." if source_model_href is not None else "Artifact unavailable."
    edit_model_button_label = "Load" if edit_glb_href is not None else "Unavailable"
    edit_model_status = "Loads on demand." if edit_glb_href is not None else "Artifact unavailable."
    source_voxelmesh_button_label = "Load" if source_voxelmesh_href is not None else "Unavailable"
    source_voxelmesh_status = "Loads on demand." if source_voxelmesh_href is not None else "Artifact unavailable."
    combined_button_label = "Load" if combined_available else "Unavailable"
    combined_status = "Loads on demand." if combined_available else "Artifact unavailable."
    ss_voxelmesh_button_label = "Load" if ss_voxelmesh_href is not None else "Unavailable"
    ss_voxelmesh_status = "Loads on demand." if ss_voxelmesh_href is not None else "Artifact unavailable."
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
    .back-link {{ display: inline-flex; align-items: center; color: #fff; text-decoration: none; font-weight: 600; }}
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
    .viewer-grid + .viewer-grid {{ margin-top: 16px; }}
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
    .viewer-card-head {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 10px; }}
    .viewer-card h2 {{ margin: 0; font-size: 16px; }}
    .viewer-status {{ margin: 0 0 10px; font-size: 13px; }}
    .viewer-3d {{ width: 100%; height: 360px; border: 2px solid #e2e8f0; border-radius: 12px; background: #f8fafc; overflow: hidden; }}
    .viewer-3d.is-hidden {{ display: none; }}
    .load-viewer-btn {{ border: none; color: #fff; cursor: pointer; font-size: 13px; font-weight: 600; background: linear-gradient(135deg, #475569 0%, #334155 100%); padding: 8px 12px; border-radius: 10px; transition: all 0.2s ease; }}
    .load-viewer-btn:hover {{ transform: translateY(-1px); }}
    .load-viewer-btn:disabled {{ background: #cbd5e1; cursor: not-allowed; transform: none; }}
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
              <div class="viewer-card-head">
                <h2>Source Model</h2>
                <button id="load-source-model" class="load-viewer-btn"{' disabled' if source_model_href is None else ''}>{html.escape(source_model_button_label)}</button>
              </div>
              <p id="status-source-model" class="viewer-status muted">{html.escape(source_model_status)}</p>
              <div id="viewer-source" class="viewer-3d is-hidden"></div>
            </section>
            <section class="viewer-card">
              <div class="viewer-card-head">
                <h2>Final Edit</h2>
                <button id="load-edit-model" class="load-viewer-btn"{' disabled' if edit_glb_href is None else ''}>{html.escape(edit_model_button_label)}</button>
              </div>
              <p id="status-edit-model" class="viewer-status muted">{html.escape(edit_model_status)}</p>
              <div id="viewer-edit" class="viewer-3d is-hidden"></div>
            </section>
          </div>
          <div class="viewer-grid">
            <section class="viewer-card">
              <div class="viewer-card-head">
                <h2>Source VoxelMesh</h2>
                <button id="load-source-voxelmesh" class="load-viewer-btn"{' disabled' if source_voxelmesh_href is None else ''}>{html.escape(source_voxelmesh_button_label)}</button>
              </div>
              <p id="status-source-voxelmesh" class="viewer-status muted">{html.escape(source_voxelmesh_status)}</p>
              <div id="viewer-source-voxelmesh" class="viewer-3d is-hidden"></div>
            </section>
            <section class="viewer-card">
              <div class="viewer-card-head">
                <h2>Source + Mask</h2>
                <button id="load-combined" class="load-viewer-btn"{' disabled' if not combined_available else ''}>{html.escape(combined_button_label)}</button>
              </div>
              <p id="status-combined" class="viewer-status muted">{html.escape(combined_status)}</p>
              <div id="viewer-combined" class="viewer-3d is-hidden"></div>
            </section>
            <section class="viewer-card">
              <div class="viewer-card-head">
                <h2>SS Coords</h2>
                <button id="load-ss-voxelmesh" class="load-viewer-btn"{' disabled' if ss_voxelmesh_href is None else ''}>{html.escape(ss_voxelmesh_button_label)}</button>
              </div>
              <p id="status-ss-voxelmesh" class="viewer-status muted">{html.escape(ss_voxelmesh_status)}</p>
              <div id="viewer-ss-voxelmesh" class="viewer-3d is-hidden"></div>
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
      container.innerHTML = "";
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

    function setupLazyViewer({{ buttonId, statusId, containerId, available, loadingText, load }}) {{
      const button = document.getElementById(buttonId);
      const status = document.getElementById(statusId);
      const container = document.getElementById(containerId);
      if (!button || !status || !container) {{
        return;
      }}
      if (!available) {{
        button.disabled = true;
        status.textContent = "Artifact unavailable.";
        return;
      }}

      let loaded = false;
      let loading = false;
      button.addEventListener("click", async () => {{
        if (loaded || loading) {{
          return;
        }}
        loading = true;
        button.disabled = true;
        button.textContent = "Loading...";
        status.textContent = loadingText;
        container.classList.remove("is-hidden");
        await new Promise((resolve) => requestAnimationFrame(resolve));
        try {{
          await load();
          loaded = true;
          status.textContent = "Loaded on demand.";
          button.textContent = "Loaded";
        }} catch (error) {{
          console.error("Failed to load lazy 3D viewer", containerId, error);
          container.classList.add("is-hidden");
          button.disabled = false;
          button.textContent = "Load";
          status.textContent = "Failed to load. Click to retry.";
        }} finally {{
          loading = false;
        }}
      }});
    }}

    async function initViewers() {{
      try {{
        setupLazyViewer({{
          buttonId: "load-source-model",
          statusId: "status-source-model",
          containerId: "viewer-source",
          available: Boolean(sourceModelHref),
          loadingText: "Loading source model...",
          load: async () => {{
            const sourceViewer = createViewer("viewer-source");
            const sourceModel = await loadGlb(sourceViewer, sourceModelHref);
            if (!sourceModel) {{
              throw new Error("Source Model is unavailable");
            }}
            sourceViewer.frameObject(sourceModel);
          }},
        }});

        setupLazyViewer({{
          buttonId: "load-edit-model",
          statusId: "status-edit-model",
          containerId: "viewer-edit",
          available: Boolean(editGlbHref),
          loadingText: "Loading final edit...",
          load: async () => {{
            const editViewer = createViewer("viewer-edit");
            const editModel = await loadGlb(editViewer, editGlbHref);
            if (!editModel) {{
              throw new Error("Final Edit is unavailable");
            }}
            editViewer.frameObject(editModel);
          }},
        }});

        setupLazyViewer({{
          buttonId: "load-source-voxelmesh",
          statusId: "status-source-voxelmesh",
          containerId: "viewer-source-voxelmesh",
          available: Boolean(sourceVoxelmeshHref),
          loadingText: "Loading voxel mesh...",
          load: async () => {{
            const sourceVoxelViewer = createViewer("viewer-source-voxelmesh");
            const sourceVoxelMesh = await loadGlb(sourceVoxelViewer, sourceVoxelmeshHref);
            if (!sourceVoxelMesh) {{
              throw new Error("Source VoxelMesh is unavailable");
            }}
            sourceVoxelViewer.frameObject(sourceVoxelMesh);
          }},
        }});

        setupLazyViewer({{
          buttonId: "load-combined",
          statusId: "status-combined",
          containerId: "viewer-combined",
          available: Boolean(sourceModelHref || maskGlbHref),
          loadingText: "Loading source model and mask...",
          load: async () => {{
            const combinedViewer = createViewer("viewer-combined");
            let combinedFrameTarget = null;
            const combinedSource = await loadGlb(combinedViewer, sourceModelHref);
            if (combinedSource) {{
              combinedFrameTarget = combinedSource;
            }}
            const combinedMask = await loadGlb(combinedViewer, maskGlbHref, (node) => {{
              node.material = new THREE.MeshPhongMaterial({{
                color: 0xcccccc,
                transparent: true,
                opacity: 0.7,
              }});
            }});
            if (!combinedFrameTarget) {{
              combinedFrameTarget = combinedMask;
            }}
            if (!combinedFrameTarget) {{
              throw new Error("Source + Mask viewer is unavailable");
            }}
            combinedViewer.frameObject(combinedFrameTarget);
          }},
        }});

        setupLazyViewer({{
          buttonId: "load-ss-voxelmesh",
          statusId: "status-ss-voxelmesh",
          containerId: "viewer-ss-voxelmesh",
          available: Boolean(ssVoxelmeshHref),
          loadingText: "Loading SS coords...",
          load: async () => {{
            const ssVoxelViewer = createViewer("viewer-ss-voxelmesh");
            const ssVoxelMesh = await loadGlb(ssVoxelViewer, ssVoxelmeshHref);
            if (!ssVoxelMesh) {{
              throw new Error("SS Coords are unavailable");
            }}
            ssVoxelViewer.frameObject(ssVoxelMesh);
          }},
        }});
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



__all__ = ['_render_daily_run_pages']
