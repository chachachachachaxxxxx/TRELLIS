#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "result_gallery" / "index.html"
DEFAULT_MANIFEST = REPO_ROOT / "outputs" / "result_gallery" / "manifest.json"
DEFAULT_SCAN_ROOTS = [
    REPO_ROOT / "outputs" / "five_methods",
    REPO_ROOT / "outputs" / "ablations",
]


@dataclass
class ExperimentResult:
    case_dir: Path
    collection: str
    pipeline: str
    name: str
    sample_glb: Path | None = None
    voxel_glb: Path | None = None
    sample_stage: str | None = None
    preprocess_meta: dict[str, Any] = field(default_factory=dict)
    ss_meta: dict[str, Any] = field(default_factory=dict)

    @property
    def relative_case_dir(self) -> Path:
        return self.case_dir.relative_to(REPO_ROOT)

    @property
    def complete_pair(self) -> bool:
        return self.sample_glb is not None and self.voxel_glb is not None


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def friendly_name(value: str) -> str:
    return value.replace("_", " ")


def _normalized_text(value: Any) -> str:
    text = str(value).strip()
    return " ".join(text.split())


def build_search_index(*values: Any) -> str:
    tokens: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in (None, "", []):
            continue
        raw = _normalized_text(value)
        if not raw:
            continue
        variants = {
            raw,
            raw.lower(),
            raw.replace("_", " "),
            raw.replace("-", " "),
            raw.replace("/", " "),
            friendly_name(raw),
        }
        for variant in variants:
            normalized = _normalized_text(variant).lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                tokens.append(normalized)
    return " ".join(tokens)


def rel_href(from_file: Path, to_file: Path | None) -> str | None:
    if to_file is None:
        return None
    return Path(os.path.relpath(to_file, start=from_file.parent)).as_posix()


def rel_dir_href(from_file: Path, to_dir: Path | None) -> str | None:
    href = rel_href(from_file, to_dir)
    if href is None:
        return None
    return href if href.endswith("/") else f"{href}/"


def find_case_dir(model_path: Path) -> tuple[Path, str | None]:
    relative = model_path.relative_to(REPO_ROOT)
    parts = relative.parts
    edit_idx = parts.index("edit")
    case_dir = REPO_ROOT.joinpath(*parts[:edit_idx])
    stage = parts[edit_idx + 1] if edit_idx + 1 < len(parts) - 1 else None
    return case_dir, stage


def discover_results(scan_roots: list[Path]) -> list[ExperimentResult]:
    experiments: dict[Path, ExperimentResult] = {}
    for root in scan_roots:
        if not root.exists():
            continue
        for model_path in sorted(root.rglob("sample_00.glb")) + sorted(root.rglob("voxel_mesh.glb")):
            case_dir, stage = find_case_dir(model_path)
            rel = case_dir.relative_to(REPO_ROOT)
            parts = rel.parts
            collection = parts[1] if len(parts) > 1 else "outputs"
            pipeline = parts[2] if len(parts) > 2 else "misc"
            name = "/".join(parts[3:]) if len(parts) > 3 else case_dir.name
            exp = experiments.setdefault(
                case_dir,
                ExperimentResult(
                    case_dir=case_dir,
                    collection=collection,
                    pipeline=pipeline,
                    name=name,
                ),
            )
            if model_path.name == "sample_00.glb":
                exp.sample_glb = model_path
                exp.sample_stage = stage
            elif model_path.name == "voxel_mesh.glb":
                exp.voxel_glb = model_path
        for exp in experiments.values():
            if not exp.preprocess_meta:
                exp.preprocess_meta = load_json(exp.case_dir / "edit" / "preprocess" / "input_preprocess.json")
            if not exp.ss_meta:
                exp.ss_meta = load_json(exp.case_dir / "edit" / "ss" / "ss_metadata.json")
    return sorted(
        experiments.values(),
        key=lambda item: (item.collection, item.pipeline, item.name),
    )


def build_manifest(results: list[ExperimentResult], output_html: Path) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for item in results:
        selection = item.preprocess_meta.get("foreground_scale_selection", {})
        probe_results = selection.get("probe_results", [])
        search_key = build_search_index(
            item.collection,
            item.pipeline,
            item.name,
            item.relative_case_dir.as_posix(),
            selection.get("strategy"),
            item.ss_meta.get("inversion_mode"),
            item.ss_meta.get("ss_denoise_solver_mode"),
        )
        probe_summary = [
            {
                "crop_scale": probe.get("crop_scale"),
                "bbox_volume_ratio": probe.get("bbox_volume_ratio"),
            }
            for probe in probe_results
        ]
        manifest.append(
            {
                "collection": item.collection,
                "pipeline": item.pipeline,
                "name": item.name,
                "case_dir": item.relative_case_dir.as_posix(),
                "case_href": rel_dir_href(output_html, item.case_dir),
                "sample_glb": rel_href(output_html, item.sample_glb),
                "voxel_glb": rel_href(output_html, item.voxel_glb),
                "sample_stage": item.sample_stage,
                "crop_scale": item.preprocess_meta.get("crop_scale"),
                "foreground_scale_strategy": selection.get("strategy"),
                "foreground_probe_mode": selection.get("mode"),
                "foreground_probe_reason": selection.get("reason"),
                "probe_image": selection.get("probe_image"),
                "probe_summary": probe_summary,
                "ss_voxel_count": item.ss_meta.get("voxel_count"),
                "inversion_mode": item.ss_meta.get("inversion_mode"),
                "ss_denoise_solver_mode": item.ss_meta.get("ss_denoise_solver_mode"),
                "complete_pair": item.complete_pair,
                "search_key": search_key,
            }
        )
    return manifest


def render_chip(label: str, value: Any) -> str:
    if value in (None, "", []):
        return ""
    return (
        f'<span class="chip"><span class="chip-label">{html.escape(label)}</span>'
        f"<strong>{html.escape(str(value))}</strong></span>"
    )


def render_meta_line(label: str, value: Any) -> str:
    if value in (None, "", []):
        return ""
    return (
        f'<div class="meta-row"><span class="meta-key">{html.escape(label)}</span>'
        f'<span class="meta-value">{html.escape(str(value))}</span></div>'
    )


def render_probe_summary(selection: dict[str, Any]) -> str:
    probe_results = selection.get("probe_results") or []
    if not probe_results:
        return ""
    chunks = []
    for probe in probe_results:
        scale = probe.get("crop_scale")
        ratio = probe.get("bbox_volume_ratio")
        if scale is None or ratio is None:
            continue
        chunks.append(f"{scale:.2f} -> {ratio:.4f}")
    return ", ".join(chunks)


def render_viewer(
    title: str,
    glb_href: str | None,
    stage_label: str | None,
    *,
    orientation: str | None = None,
    axis_label: str | None = None,
) -> str:
    if glb_href is None:
        return f"""
        <section class="viewer-card empty">
          <div class="viewer-head">
            <p class="viewer-title">{html.escape(title)}</p>
            <p class="viewer-stage">missing</p>
          </div>
          <div class="viewer-empty">No GLB found for this slot.</div>
        </section>
        """
    viewer_attrs = [
        f'src="{html.escape(glb_href, quote=True)}"',
        'camera-controls',
        'shadow-intensity="0.9"',
        'exposure="1"',
        'environment-image="neutral"',
        'interaction-prompt="none"',
        'touch-action="pan-y"',
        'loading="lazy"',
        'reveal="auto"',
        'camera-target="0m 0m 0m"',
        'camera-orbit="45deg 65deg auto"',
    ]
    if orientation is not None:
        viewer_attrs.append(f'orientation="{html.escape(orientation, quote=True)}"')
    stage_text = stage_label or ""
    if axis_label:
        stage_text = f"{stage_text} · {axis_label}" if stage_text else axis_label
    return f"""
    <section class="viewer-card">
      <div class="viewer-head">
        <p class="viewer-title">{html.escape(title)}</p>
        <p class="viewer-stage">{html.escape(stage_text)}</p>
      </div>
      <model-viewer
        {" ".join(viewer_attrs)}
      ></model-viewer>
      <div class="viewer-links">
        <a href="{html.escape(glb_href, quote=True)}" target="_blank" rel="noopener">open glb</a>
      </div>
    </section>
    """


def render_card(item: ExperimentResult, output_html: Path) -> str:
    selection = item.preprocess_meta.get("foreground_scale_selection", {})
    probe_summary = render_probe_summary(selection)
    card_search = build_search_index(
        item.collection,
        item.pipeline,
        item.name,
        item.relative_case_dir.as_posix(),
        selection.get("strategy"),
        item.ss_meta.get("inversion_mode"),
        item.ss_meta.get("ss_denoise_solver_mode"),
    )
    relative_case_path = item.relative_case_dir.as_posix()
    case_href = rel_dir_href(output_html, item.case_dir)
    sample_href = rel_href(output_html, item.sample_glb)
    voxel_href = rel_href(output_html, item.voxel_glb)

    meta_lines = "".join(
        [
            render_meta_line("path", relative_case_path),
            render_meta_line("crop_scale", item.preprocess_meta.get("crop_scale")),
            render_meta_line("foreground_strategy", selection.get("strategy")),
            render_meta_line("probe_image", selection.get("probe_image")),
            render_meta_line("probe_mode", selection.get("mode")),
            render_meta_line("probe_reason", selection.get("reason")),
            render_meta_line("probe_bbox", probe_summary),
            render_meta_line("inversion_mode", item.ss_meta.get("inversion_mode")),
            render_meta_line("denoise_solver", item.ss_meta.get("ss_denoise_solver_mode")),
            render_meta_line("ss_voxel_count", item.ss_meta.get("voxel_count")),
        ]
    )

    chips = "".join(
        [
            render_chip("collection", item.collection),
            render_chip("pipeline", item.pipeline),
            render_chip("crop", item.preprocess_meta.get("crop_scale")),
            render_chip("strategy", selection.get("strategy")),
            render_chip("solver", item.ss_meta.get("ss_denoise_solver_mode") or item.ss_meta.get("inversion_mode")),
            render_chip("voxels", item.ss_meta.get("voxel_count")),
        ]
    )

    return f"""
    <article
      class="result-card"
      data-complete="{str(item.complete_pair).lower()}"
      data-search="{html.escape(card_search.lower(), quote=True)}"
    >
      <div class="card-head">
        <div>
          <p class="eyebrow">{html.escape(friendly_name(item.collection))} / {html.escape(friendly_name(item.pipeline))}</p>
          <h2>{html.escape(item.name)}</h2>
          <p class="card-path">
            <a href="{html.escape(case_href, quote=True) if case_href else '#'}" target="_blank" rel="noopener">{html.escape(relative_case_path)}</a>
          </p>
        </div>
        <div class="chip-row">{chips}</div>
      </div>

      <div class="viewer-grid">
        {render_viewer("voxel_mesh.glb", voxel_href, "edit/ss", orientation="-90deg 0deg 0deg", axis_label="z-up")}
        {render_viewer("sample_00.glb", sample_href, f"edit/{item.sample_stage}" if item.sample_stage else "edit")}
      </div>

      <div class="meta-grid">{meta_lines}</div>
    </article>
    """


def render_group(group_key: str, items: list[ExperimentResult], output_html: Path) -> str:
    cards = "".join(render_card(item, output_html) for item in items)
    return f"""
    <section class="group-section">
      <div class="group-head">
        <h2>{html.escape(group_key)}</h2>
        <p>{len(items)} result{"s" if len(items) != 1 else ""}</p>
      </div>
      <div class="cards">{cards}</div>
    </section>
    """


def render_html(results: list[ExperimentResult], output_html: Path) -> str:
    groups: dict[str, list[ExperimentResult]] = {}
    for item in results:
        group_key = f"{friendly_name(item.collection)} / {friendly_name(item.pipeline)}"
        groups.setdefault(group_key, []).append(item)

    total = len(results)
    complete = sum(item.complete_pair for item in results)
    sample_only = sum(item.sample_glb is not None and item.voxel_glb is None for item in results)
    group_blocks = "".join(
        render_group(group_key, groups[group_key], output_html) for group_key in sorted(groups)
    )

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TRELLIS Edit Result Gallery</title>
  <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/4.0.0/model-viewer.min.js"></script>
  <script nomodule src="https://ajax.googleapis.com/ajax/libs/model-viewer/4.0.0/model-viewer-legacy.js"></script>
  <style>
    :root {{
      --bg: #f5efe2;
      --paper: rgba(255, 251, 242, 0.86);
      --ink: #1e2430;
      --muted: #5d6674;
      --line: rgba(30, 36, 48, 0.12);
      --accent: #9b4d30;
      --accent-2: #2b7a78;
      --shadow: 0 18px 60px rgba(42, 38, 32, 0.12);
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(155, 77, 48, 0.18), transparent 34%),
        radial-gradient(circle at top right, rgba(43, 122, 120, 0.17), transparent 28%),
        linear-gradient(180deg, #f9f5ea 0%, #f1eadc 100%);
    }}
    body::before {{
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background-image:
        linear-gradient(rgba(30, 36, 48, 0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(30, 36, 48, 0.02) 1px, transparent 1px);
      background-size: 32px 32px;
      mask-image: linear-gradient(180deg, rgba(0, 0, 0, 0.9), transparent 95%);
    }}
    .page {{
      width: min(1500px, calc(100vw - 40px));
      margin: 28px auto 60px;
      position: relative;
      z-index: 1;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1.4fr 1fr;
      gap: 18px;
      margin-bottom: 20px;
    }}
    .hero-card,
    .toolbar,
    .group-section,
    .result-card {{
      background: var(--paper);
      backdrop-filter: blur(14px);
      border: 1px solid rgba(255, 255, 255, 0.55);
      box-shadow: var(--shadow);
    }}
    .hero-card {{
      border-radius: 26px;
      padding: 26px 28px;
    }}
    .hero-card h1 {{
      margin: 0 0 10px;
      font-size: clamp(30px, 4vw, 52px);
      line-height: 0.96;
      letter-spacing: -0.04em;
    }}
    .hero-card p {{
      margin: 0;
      max-width: 55ch;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.6;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}
    .stat {{
      border-radius: 20px;
      padding: 20px 18px;
      background: linear-gradient(145deg, rgba(255,255,255,0.7), rgba(255,255,255,0.3));
      border: 1px solid rgba(30, 36, 48, 0.08);
    }}
    .stat strong {{
      display: block;
      font-size: clamp(28px, 3vw, 40px);
      line-height: 1;
      margin-bottom: 8px;
    }}
    .stat span {{
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      align-items: center;
      justify-content: space-between;
      border-radius: 22px;
      padding: 14px 16px;
      margin-bottom: 22px;
    }}
    .toolbar-left {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
    }}
    .search {{
      min-width: min(460px, 100%);
      flex: 1 1 420px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.72);
      padding: 14px 18px;
      font-size: 15px;
      color: var(--ink);
      outline: none;
    }}
    .toggle {{
      display: inline-flex;
      gap: 10px;
      align-items: center;
      color: var(--muted);
      font-size: 14px;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(255,255,255,0.58);
      border: 1px solid var(--line);
    }}
    .toggle input {{
      accent-color: var(--accent);
    }}
    .hint {{
      color: var(--muted);
      font-size: 13px;
    }}
    .group-section {{
      border-radius: 28px;
      padding: 18px;
      margin-bottom: 22px;
    }}
    .group-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      padding: 6px 8px 14px;
      border-bottom: 1px solid var(--line);
      margin-bottom: 16px;
    }}
    .group-head h2 {{
      margin: 0;
      font-size: 22px;
      letter-spacing: -0.03em;
      text-transform: capitalize;
    }}
    .group-head p {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .cards {{
      display: grid;
      gap: 18px;
    }}
    .result-card {{
      border-radius: 24px;
      padding: 18px;
    }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: start;
      margin-bottom: 14px;
    }}
    .eyebrow {{
      margin: 0 0 8px;
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 0.1em;
      font-size: 12px;
      font-weight: 700;
    }}
    .card-head h2 {{
      margin: 0;
      font-size: 26px;
      line-height: 1.05;
      letter-spacing: -0.03em;
    }}
    .chip-row {{
      display: flex;
      flex-wrap: wrap;
      justify-content: end;
      gap: 8px;
      max-width: 54%;
    }}
    .chip {{
      display: inline-flex;
      gap: 7px;
      align-items: center;
      border-radius: 999px;
      padding: 7px 10px;
      background: rgba(255,255,255,0.72);
      border: 1px solid rgba(30, 36, 48, 0.08);
      font-size: 12px;
      color: var(--muted);
      white-space: nowrap;
    }}
    .chip strong {{
      color: var(--ink);
      font-weight: 700;
    }}
    .viewer-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 14px;
    }}
    .viewer-card {{
      border-radius: 20px;
      overflow: hidden;
      border: 1px solid rgba(30, 36, 48, 0.08);
      background: linear-gradient(180deg, rgba(255,255,255,0.86), rgba(247, 242, 231, 0.9));
      min-height: 360px;
      display: flex;
      flex-direction: column;
    }}
    .viewer-card.empty {{
      justify-content: center;
      background: linear-gradient(180deg, rgba(240,235,224,0.92), rgba(232,225,211,0.94));
    }}
    .viewer-head {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: baseline;
      padding: 12px 14px;
      border-bottom: 1px solid rgba(30, 36, 48, 0.08);
    }}
    .viewer-title {{
      margin: 0;
      font-size: 14px;
      font-weight: 700;
      letter-spacing: 0.02em;
    }}
    .viewer-stage {{
      margin: 0;
      color: var(--muted);
      font-size: 12px;
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
    }}
    .card-path {{
      margin: 8px 0 0;
      font-size: 12px;
      line-height: 1.45;
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      word-break: break-word;
    }}
    .card-path a {{
      color: var(--muted);
      text-decoration: none;
    }}
    .card-path a:hover {{
      color: var(--accent-2);
      text-decoration: underline;
    }}
    model-viewer {{
      width: 100%;
      flex: 1;
      min-height: 300px;
      background:
        radial-gradient(circle at 30% 20%, rgba(43, 122, 120, 0.12), transparent 26%),
        radial-gradient(circle at 70% 80%, rgba(155, 77, 48, 0.12), transparent 28%),
        linear-gradient(180deg, #f7f1e4 0%, #efe6d6 100%);
    }}
    .viewer-empty {{
      padding: 28px 20px;
      color: var(--muted);
      font-size: 15px;
      text-align: center;
    }}
    .viewer-links {{
      padding: 10px 14px 14px;
      border-top: 1px solid rgba(30, 36, 48, 0.08);
    }}
    .viewer-links a {{
      color: var(--accent-2);
      text-decoration: none;
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .viewer-links a:hover {{
      text-decoration: underline;
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px 16px;
      border-top: 1px solid var(--line);
      padding-top: 14px;
    }}
    .meta-row {{
      display: flex;
      flex-direction: column;
      gap: 4px;
      min-width: 0;
    }}
    .meta-key {{
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .meta-value {{
      font-size: 14px;
      line-height: 1.45;
      word-break: break-word;
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
    }}
    .result-card.is-hidden,
    .group-section.is-hidden {{
      display: none;
    }}
    @media (max-width: 1100px) {{
      .hero {{
        grid-template-columns: 1fr;
      }}
      .card-head,
      .group-head {{
        flex-direction: column;
      }}
      .chip-row {{
        justify-content: start;
        max-width: none;
      }}
      .meta-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}
    @media (max-width: 760px) {{
      .page {{
        width: min(100vw - 20px, 100%);
        margin: 14px auto 30px;
      }}
      .toolbar {{
        align-items: stretch;
      }}
      .search {{
        min-width: 0;
      }}
      .viewer-grid,
      .meta-grid,
      .stats {{
        grid-template-columns: 1fr;
      }}
      .hero-card,
      .group-section,
      .result-card {{
        border-radius: 20px;
      }}
      .card-head h2 {{
        font-size: 22px;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div class="hero-card">
        <p class="eyebrow">TRELLIS Edit Browser</p>
        <h1>Experiment Gallery</h1>
        <p>集中浏览每个实验的 <code>edit/ss/voxel_mesh.glb</code> 和 <code>sample_00.glb</code>。页面会自动扫已有结果目录，适合快速比较不同方法、不同前景尺度策略和一阶/二阶版本。</p>
      </div>
      <div class="stats">
        <div class="stat"><strong>{total}</strong><span>experiments</span></div>
        <div class="stat"><strong>{complete}</strong><span>complete pairs</span></div>
        <div class="stat"><strong>{sample_only}</strong><span>sample only</span></div>
      </div>
    </section>

    <section class="toolbar">
      <div class="toolbar-left">
        <input id="search" class="search" type="search" placeholder="搜索 case 名、相对路径、pipeline、strategy、solver...">
        <label class="toggle">
          <input id="only-complete" type="checkbox">
          <span>只看同时有 voxel 和 final 的结果</span>
        </label>
      </div>
      <p class="hint">建议通过本地 HTTP 打开，不要直接双击 HTML。</p>
    </section>

    {group_blocks}
  </main>

  <script>
    const searchInput = document.getElementById("search");
    const onlyComplete = document.getElementById("only-complete");
    const cards = Array.from(document.querySelectorAll(".result-card"));
    const groups = Array.from(document.querySelectorAll(".group-section"));

    function applyFilters() {{
      const query = searchInput.value.trim().toLowerCase();
      for (const card of cards) {{
        const haystack = card.dataset.search || "";
        const complete = card.dataset.complete === "true";
        const matchesText = !query || haystack.includes(query);
        const matchesComplete = !onlyComplete.checked || complete;
        card.classList.toggle("is-hidden", !(matchesText && matchesComplete));
      }}
      for (const group of groups) {{
        const visible = group.querySelector(".result-card:not(.is-hidden)");
        group.classList.toggle("is-hidden", !visible);
      }}
    }}

    searchInput.addEventListener("input", applyFilters);
    onlyComplete.addEventListener("change", applyFilters);
    applyFilters();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a static HTML gallery for TRELLIS edit results.")
    parser.add_argument(
        "--scan-root",
        action="append",
        dest="scan_roots",
        help="Directory to scan. Can be passed multiple times. Defaults to outputs/five_methods and outputs/ablations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output HTML path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Output manifest path. Default: {DEFAULT_MANIFEST}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scan_roots = [Path(root).resolve() for root in args.scan_roots] if args.scan_roots else DEFAULT_SCAN_ROOTS
    results = discover_results(scan_roots)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(results, args.output))
    args.manifest.write_text(json.dumps(build_manifest(results, args.output), indent=2))
    print(f"Wrote gallery HTML to {args.output}")
    print(f"Wrote manifest to {args.manifest}")
    print(f"Discovered {len(results)} experiment results.")


if __name__ == "__main__":
    main()
