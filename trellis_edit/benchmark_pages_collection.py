from __future__ import annotations

import html
import json
import shutil
from pathlib import Path
from typing import Any

from trellis_edit.common import ensure_dir, write_json

from .benchmark_pages_common import (
    SHOWCASE_VIEW_SPECS,
    _build_metric_rank_classes,
    _bundle_relpath,
    _comparison_table_css,
    _extract_scalar_metric,
    _format_delta,
    _format_metric_value,
    _format_summary_metric_value,
    _href_between_paths,
    _load_daily_runs,
    _load_json,
    _load_jsonl,
    _metric_label,
    _metric_table_class,
    _ordered_metric_names,
    _ordered_run_groups,
    _relative_href,
    _render_metric_sections,
    _run_name_sort_key,
    _slug,
    _table_metric_names,
)
from .benchmark_pages_run import _render_daily_run_pages
from .benchmark_pages_shared import _case_record_page_path, _load_case_payload

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
        (
            group_name,
            sorted(
                [run for run in runs if run["group_name"] == group_name],
                key=_run_name_sort_key,
            ),
        )
        for group_name in group_names
    ]
    ordered_runs = [run for _, group_runs in grouped_runs for run in group_runs]
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
                for run in ordered_runs
            ],
        },
    )
    return index_path


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
        config_page_relpath = str((run.get("manifest") or {}).get("experiment_config_page_path") or "").strip()
        config_href = (
            _href_between_paths(
                page_path,
                daily_root / run["path"] / config_page_relpath,
            )
            if config_page_relpath
            else None
        )
        config_link_html = (
            f"<div class='small'><a href='{html.escape(config_href)}'>config</a></div>"
            if config_href
            else ""
        )
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
            f"<td><a href='{html.escape(run_href)}'>{html.escape(run['label'])}</a><div class='muted small'>{html.escape(run['id'])}</div>{config_link_html}</td>"
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


def _select_showcase_runs(
    grouped_runs: list[tuple[str, list[dict[str, Any]]]],
    current_group: str | None,
) -> list[dict[str, Any]]:
    if current_group is None:
        all_runs = [run for _, runs in grouped_runs for run in runs]
        all_runs.sort(
            key=lambda item: (
                str(item.get("created_at") or ""),
                str(item.get("id") or ""),
            ),
            reverse=True,
        )
        selected: list[dict[str, Any]] = []
        seen_labels: set[str] = set()
        for run in all_runs:
            label = str(run.get("label") or "")
            if label in seen_labels:
                continue
            seen_labels.add(label)
            selected.append(run)
            if len(selected) >= 4:
                break
        return selected
    if not grouped_runs:
        return []
    return grouped_runs[0][1][:3]


def _build_collection_showcase_cards(
    *,
    page_path: Path,
    runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for run in runs:
        run_root = Path(run["root"])
        case_records = _load_jsonl(run_root / "cases.jsonl")
        for case_record in case_records:
            status = str(case_record.get("status") or "").strip().lower()
            if status == "failed":
                continue
            case_payload = _load_case_payload(run_root, case_record)
            artifacts = case_payload.get("artifacts") or {}
            source_relpath = artifacts.get("source_image")
            edit_relpath = artifacts.get("edit_image")
            images_relpath = artifacts.get("images")
            if not source_relpath or not edit_relpath or not images_relpath:
                continue

            source_href = _href_between_paths(page_path, run_root / str(source_relpath))
            edit_href = _href_between_paths(page_path, run_root / str(edit_relpath))
            if source_href is None or edit_href is None:
                continue

            views: list[dict[str, str]] = []
            all_present = True
            images_root = run_root / str(images_relpath)
            for label, filename in SHOWCASE_VIEW_SPECS:
                href = _href_between_paths(page_path, images_root / filename)
                if href is None:
                    all_present = False
                    break
                views.append({"label": label, "href": href})
            if not all_present:
                continue

            case_page_href = _href_between_paths(page_path, _case_record_page_path(run_root, case_record))
            cards.append(
                {
                    "run_label": str(run.get("label") or run_root.name),
                    "group_name": str(run.get("group_name") or ""),
                    "case_id": str(case_payload.get("case_id") or case_record.get("case_id") or ""),
                    "prompt_text": str(case_payload.get("prompt_text") or "").strip(),
                    "case_page_href": case_page_href,
                    "source_href": source_href,
                    "edit_href": edit_href,
                    "views": views,
                }
            )
            break
    return cards


def _render_collection_showcase(cards: list[dict[str, Any]]) -> str:
    if not cards:
        return ""
    card_html = []
    for card in cards:
        strip_items = [
            (
                "source",
                card["source_href"],
            ),
            (
                "edit",
                card["edit_href"],
            ),
            *[(item["label"], item["href"]) for item in card["views"]],
        ]
        thumbs = "".join(
            (
                "<figure class='showcase-thumb'>"
                f"<div class='showcase-label'>{html.escape(label)}</div>"
                f"<img src=\"{html.escape(href)}\" alt=\"{html.escape(label)}\" loading=\"lazy\">"
                "</figure>"
            )
            for label, href in strip_items
        )
        links = []
        if card.get("case_page_href"):
            links.append(f"<a href=\"{html.escape(str(card['case_page_href']))}\">case page</a>")
        prompt_line = ""
        if card.get("prompt_text"):
            prompt_line = f"<p class='muted small'>{html.escape(str(card['prompt_text']))}</p>"
        card_html.append(
            "<article class='showcase-card'>"
            "<div class='showcase-card-head'>"
            "<div>"
            f"<h3>{html.escape(card['run_label'])}</h3>"
            f"<p class='muted'>{html.escape(card['case_id'])}</p>"
            f"{prompt_line}"
            "</div>"
            f"<div class='section-links'>{' · '.join(links)}</div>"
            "</div>"
            f"<div class='showcase-strip'>{thumbs}</div>"
            "</article>"
        )
    return (
        "<section class='panel'>"
        "<div class='section-head'>"
        "<div><h2>Quick Showcase</h2><p class='muted'>One sample row per selected run. Layout: source, edit, front, right, back, left.</p></div>"
        "</div>"
        f"<div class='showcase-list'>{''.join(card_html)}</div>"
        "</section>"
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
        source_button_id = f"load-source-{index}"
        source_status_id = f"status-source-{index}"
        edit_button_id = f"load-edit-{index}"
        edit_status_id = f"status-edit-{index}"
        combined_button_id = f"load-combined-{index}"
        combined_status_id = f"status-combined-{index}"
        viewer_specs.append(
            {
                "sourceId": source_id,
                "sourceButtonId": source_button_id,
                "sourceStatusId": source_status_id,
                "combinedId": combined_id,
                "combinedButtonId": combined_button_id,
                "combinedStatusId": combined_status_id,
                "editId": edit_id,
                "editButtonId": edit_button_id,
                "editStatusId": edit_status_id,
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
            "<section class='viewer-card'>"
            f"<div class='viewer-card-head'><h3>Source Model</h3><button id=\"{html.escape(source_button_id)}\" class='load-viewer-btn'>{'Load' if source_model_href is not None else 'Unavailable'}</button></div>"
            f"<p id=\"{html.escape(source_status_id)}\" class='viewer-status muted'>{'Loads on demand.' if source_model_href is not None else 'Artifact unavailable.'}</p>"
            f"<div id=\"{html.escape(source_id)}\" class='viewer-3d is-hidden'></div>"
            "</section>"
            "<section class='viewer-card'>"
            f"<div class='viewer-card-head'><h3>Final Edit</h3><button id=\"{html.escape(edit_button_id)}\" class='load-viewer-btn'>{'Load' if run_entry.get('edit_glb_href') else 'Unavailable'}</button></div>"
            f"<p id=\"{html.escape(edit_status_id)}\" class='viewer-status muted'>{'Loads on demand.' if run_entry.get('edit_glb_href') else 'Artifact unavailable.'}</p>"
            f"<div id=\"{html.escape(edit_id)}\" class='viewer-3d is-hidden'></div>"
            "</section>"
            "</div>"
            "<div class='viewer-grid viewer-grid-secondary'>"
            "<section class='viewer-card'>"
            f"<div class='viewer-card-head'><h3>Source + Mask</h3><button id=\"{html.escape(combined_button_id)}\" class='load-viewer-btn'>{'Load' if (source_model_href is not None or mask_glb_href is not None) else 'Unavailable'}</button></div>"
            f"<p id=\"{html.escape(combined_status_id)}\" class='viewer-status muted'>{'Loads on demand.' if (source_model_href is not None or mask_glb_href is not None) else 'Artifact unavailable.'}</p>"
            f"<div id=\"{html.escape(combined_id)}\" class='viewer-3d is-hidden'></div>"
            "</section>"
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
    .viewer-grid-secondary {{ grid-template-columns: repeat(1, minmax(0, 1fr)); }}
    .metric-grid {{ grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); margin-top: 12px; }}
    .metric-sections {{ margin-top: 12px; }}
    .run-card, .viewer-card, .metric {{ background: #fafafa; border: 1px solid #e5e5e5; border-radius: 12px; }}
    .run-card {{ padding: 14px; }}
    .viewer-card {{ padding: 12px; }}
    .viewer-card-head {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 10px; }}
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
    .viewer-3d.is-hidden {{ display: none; }}
    .viewer-status {{ margin: 0 0 10px; font-size: 13px; }}
    .load-viewer-btn {{ border: none; color: #fff; cursor: pointer; font-size: 13px; font-weight: 600; background: linear-gradient(135deg, #475569 0%, #334155 100%); padding: 8px 12px; border-radius: 10px; transition: all 0.2s ease; }}
    .load-viewer-btn:hover {{ transform: translateY(-1px); }}
    .load-viewer-btn:disabled {{ background: #cbd5e1; cursor: not-allowed; transform: none; }}
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
      container.innerHTML = "";
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
          console.error("Failed to initialize lazy viewer", containerId, error);
          container.classList.add("is-hidden");
          button.disabled = false;
          button.textContent = "Load";
          status.textContent = "Failed to load. Click to retry.";
        }} finally {{
          loading = false;
        }}
      }});
    }}

    function initViewerSet(spec) {{
      setupLazyViewer({{
        buttonId: spec.sourceButtonId,
        statusId: spec.sourceStatusId,
        containerId: spec.sourceId,
        available: Boolean(spec.sourceHref),
        loadingText: "Loading source model...",
        load: async () => {{
          const sourceViewer = createViewer(spec.sourceId);
          const sourceModel = await loadGlb(sourceViewer, spec.sourceHref);
          if (!sourceModel) {{
            throw new Error("Source Model is unavailable");
          }}
          sourceViewer.frameObject(sourceModel);
        }},
      }});

      setupLazyViewer({{
        buttonId: spec.editButtonId,
        statusId: spec.editStatusId,
        containerId: spec.editId,
        available: Boolean(spec.editHref),
        loadingText: "Loading final edit...",
        load: async () => {{
          const editViewer = createViewer(spec.editId);
          const editModel = await loadGlb(editViewer, spec.editHref);
          if (!editModel) {{
            throw new Error("Final Edit is unavailable");
          }}
          editViewer.frameObject(editModel);
        }},
      }});

      setupLazyViewer({{
        buttonId: spec.combinedButtonId,
        statusId: spec.combinedStatusId,
        containerId: spec.combinedId,
        available: Boolean(spec.sourceHref || spec.maskHref),
        loadingText: "Loading source model and mask...",
        load: async () => {{
          const combinedViewer = createViewer(spec.combinedId);
          let combinedFrameTarget = null;
          const combinedSource = await loadGlb(combinedViewer, spec.sourceHref);
          if (combinedSource) {{
            combinedFrameTarget = combinedSource;
          }}
          const combinedMask = await loadGlb(combinedViewer, spec.maskHref, (node) => {{
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
    }}

    function initAllViewers() {{
      for (const spec of viewerSpecs) {{
        try {{
          initViewerSet(spec);
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

__all__ = ['build_daily_index', 'build_focus_index']
