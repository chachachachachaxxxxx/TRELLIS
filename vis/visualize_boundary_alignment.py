#!/usr/bin/env python3
"""Generate a boundary-focused voxel visualization report.

This tool compares source/edit voxel occupancies against a mask voxel region and
produces a standalone HTML report focused on:
1. Mask boundary occupancy
2. In-mask overlap between source and edit
3. Whether edited voxels stay aligned with the boundary
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Set, Tuple

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import torch


Coord = Tuple[int, int, int]
NEIGHBOR_OFFSETS: Tuple[Coord, ...] = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)


ROLE_DEFAULTS = {
    "source": ("coords.pt", "coords.ply", "voxels.ply", "features.npz"),
    "edit": ("coords.pt", "coords.ply", "voxels.ply"),
    "mask": ("coords.pt", "coords.ply", "voxels_delete.ply", "mask_coords.pt", "mask_coords.ply"),
}


@dataclass(frozen=True)
class LayerStyle:
    name: str
    color: str
    size: float
    opacity: float
    visible: bool
    description: str


LAYER_STYLES: Dict[str, LayerStyle] = {
    "mask_boundary": LayerStyle(
        name="Mask Boundary",
        color="#ff5a5f",
        size=3.2,
        opacity=0.22,
        visible=True,
        description="Mask itself, restricted to boundary voxels only.",
    ),
    "boundary_aligned": LayerStyle(
        name="Boundary Aligned",
        color="#00c2a8",
        size=4.0,
        opacity=0.95,
        visible=True,
        description="Source and edit both occupy the same mask-boundary voxels.",
    ),
    "boundary_missing": LayerStyle(
        name="Boundary Missing",
        color="#ffb347",
        size=4.4,
        opacity=0.95,
        visible=True,
        description="Source occupied these boundary voxels, but edit did not.",
    ),
    "boundary_added": LayerStyle(
        name="Boundary Added",
        color="#9d6cff",
        size=4.4,
        opacity=0.95,
        visible=True,
        description="Edit occupies these mask-boundary voxels while source did not.",
    ),
    "in_mask_overlap": LayerStyle(
        name="In-Mask Overlap",
        color="#2f90ff",
        size=3.4,
        opacity=0.9,
        visible=True,
        description="Source and edit overlap inside the mask.",
    ),
    "in_mask_source_only": LayerStyle(
        name="In-Mask Source Only",
        color="#ffd166",
        size=3.6,
        opacity=0.78,
        visible=False,
        description="Mask voxels kept only by the source, absent from the edit.",
    ),
    "in_mask_edit_only": LayerStyle(
        name="In-Mask Edit Only",
        color="#06d6a0",
        size=3.8,
        opacity=0.84,
        visible=True,
        description="Mask voxels newly introduced by the edit.",
    ),
    "edit_leak_near_boundary": LayerStyle(
        name="Edit Leak Near Boundary",
        color="#ffffff",
        size=4.5,
        opacity=0.96,
        visible=True,
        description="Edited voxels outside the mask but directly adjacent to the mask boundary.",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Source voxel file or directory")
    parser.add_argument("--edit", required=True, help="Edited voxel file or directory")
    parser.add_argument("--mask", required=True, help="Mask voxel file or directory")
    parser.add_argument("--resolution", type=int, default=64, help="Voxel grid resolution")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/boundary_alignment_vis"),
        help="Output directory for HTML + JSON report",
    )
    parser.add_argument(
        "--report-name",
        default="boundary_alignment",
        help="Base filename for generated report files",
    )
    return parser.parse_args()


def resolve_input_path(raw_path: str, role: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"{role} path not found: {path}")

    for candidate in ROLE_DEFAULTS[role]:
        candidate_path = path / candidate
        if candidate_path.is_file():
            return candidate_path

    searched = ", ".join(ROLE_DEFAULTS[role])
    raise FileNotFoundError(f"No supported {role} voxel file found under {path}. Tried: {searched}")


def normalize_coords_array(coords: np.ndarray | torch.Tensor, resolution: int) -> np.ndarray:
    array = coords.detach().cpu().numpy() if isinstance(coords, torch.Tensor) else np.asarray(coords)
    if array.ndim != 2 or array.shape[1] not in (3, 4):
        raise ValueError(f"Expected coords shape [N, 3] or [N, 4], got {array.shape}")

    if array.shape[1] == 4:
        array = array[:, 1:]

    if array.size == 0:
        return np.zeros((0, 3), dtype=np.int32)

    array = np.rint(array).astype(np.int32, copy=False)
    valid = np.all((array >= 0) & (array < resolution), axis=1)
    array = array[valid]
    if len(array) == 0:
        return np.zeros((0, 3), dtype=np.int32)

    return np.unique(array, axis=0)


def load_coords(path: Path, resolution: int) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix == ".ply":
        import utils3d  # type: ignore

        positions = np.asarray(utils3d.io.read_ply(str(path))[0], dtype=np.float32)
        coords = ((positions + 0.5) * resolution).astype(np.int32)
        coords = np.clip(coords, 0, resolution - 1)
        return normalize_coords_array(coords, resolution)

    if suffix == ".pt":
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            for key in ("coords", "indices"):
                if key in payload:
                    return normalize_coords_array(payload[key], resolution)
            raise ValueError(f"Unsupported .pt payload keys in {path}")
        return normalize_coords_array(payload, resolution)

    if suffix == ".npz":
        payload = np.load(path)
        for key in ("indices", "coords"):
            if key in payload:
                return normalize_coords_array(payload[key], resolution)
        raise ValueError(f"Unsupported .npz payload keys in {path}")

    raise ValueError(f"Unsupported voxel input format: {path}")


def array_to_coord_set(coords: np.ndarray) -> Set[Coord]:
    return set(map(tuple, coords.astype(np.int32, copy=False)))


def coord_set_to_array(coords: Iterable[Coord]) -> np.ndarray:
    coords_list = sorted(set(coords))
    if not coords_list:
        return np.zeros((0, 3), dtype=np.int32)
    return np.asarray(coords_list, dtype=np.int32)


def in_bounds(coord: Coord, resolution: int) -> bool:
    x, y, z = coord
    return 0 <= x < resolution and 0 <= y < resolution and 0 <= z < resolution


def find_boundary_voxels(coord_set: Set[Coord], resolution: int) -> Set[Coord]:
    boundary: Set[Coord] = set()
    for coord in coord_set:
        x, y, z = coord
        for dx, dy, dz in NEIGHBOR_OFFSETS:
            neighbor = (x + dx, y + dy, z + dz)
            if not in_bounds(neighbor, resolution) or neighbor not in coord_set:
                boundary.add(coord)
                break
    return boundary


def find_outer_shell(mask_boundary: Set[Coord], mask_set: Set[Coord], resolution: int) -> Set[Coord]:
    shell: Set[Coord] = set()
    for x, y, z in mask_boundary:
        for dx, dy, dz in NEIGHBOR_OFFSETS:
            neighbor = (x + dx, y + dy, z + dz)
            if in_bounds(neighbor, resolution) and neighbor not in mask_set:
                shell.add(neighbor)
    return shell


def safe_ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def compute_regions(
    source_coords: np.ndarray,
    edit_coords: np.ndarray,
    mask_coords: np.ndarray,
    resolution: int,
) -> tuple[Dict[str, np.ndarray], Dict[str, object]]:
    source_set = array_to_coord_set(source_coords)
    edit_set = array_to_coord_set(edit_coords)
    mask_set = array_to_coord_set(mask_coords)

    mask_boundary = find_boundary_voxels(mask_set, resolution)
    outer_shell = find_outer_shell(mask_boundary, mask_set, resolution)

    source_in_mask = source_set & mask_set
    edit_in_mask = edit_set & mask_set
    in_mask_overlap = source_in_mask & edit_in_mask
    in_mask_source_only = source_in_mask - edit_in_mask
    in_mask_edit_only = edit_in_mask - source_in_mask

    source_boundary = source_set & mask_boundary
    edit_boundary = edit_set & mask_boundary
    boundary_aligned = source_boundary & edit_boundary
    boundary_missing = source_boundary - edit_boundary
    boundary_added = edit_boundary - source_boundary
    edit_leak_near_boundary = (edit_set - mask_set) & outer_shell

    regions = {
        "mask_boundary": coord_set_to_array(mask_boundary),
        "boundary_aligned": coord_set_to_array(boundary_aligned),
        "boundary_missing": coord_set_to_array(boundary_missing),
        "boundary_added": coord_set_to_array(boundary_added),
        "in_mask_overlap": coord_set_to_array(in_mask_overlap),
        "in_mask_source_only": coord_set_to_array(in_mask_source_only),
        "in_mask_edit_only": coord_set_to_array(in_mask_edit_only),
        "edit_leak_near_boundary": coord_set_to_array(edit_leak_near_boundary),
    }

    in_mask_union = source_in_mask | edit_in_mask
    boundary_union = source_boundary | edit_boundary

    stats = {
        "resolution": resolution,
        "totals": {
            "source": len(source_set),
            "edit": len(edit_set),
            "mask": len(mask_set),
        },
        "mask_overlap": {
            "source_in_mask": len(source_in_mask),
            "edit_in_mask": len(edit_in_mask),
            "overlap": len(in_mask_overlap),
            "source_only": len(in_mask_source_only),
            "edit_only": len(in_mask_edit_only),
            "overlap_vs_source_ratio": safe_ratio(len(in_mask_overlap), len(source_in_mask)),
            "overlap_vs_edit_ratio": safe_ratio(len(in_mask_overlap), len(edit_in_mask)),
            "iou": safe_ratio(len(in_mask_overlap), len(in_mask_union)),
        },
        "boundary_alignment": {
            "mask_boundary_voxels": len(mask_boundary),
            "source_on_mask_boundary": len(source_boundary),
            "edit_on_mask_boundary": len(edit_boundary),
            "aligned": len(boundary_aligned),
            "missing": len(boundary_missing),
            "added": len(boundary_added),
            "edit_leak_near_boundary": len(edit_leak_near_boundary),
            "source_boundary_preservation_ratio": safe_ratio(len(boundary_aligned), len(source_boundary)),
            "mask_boundary_coverage_by_edit_ratio": safe_ratio(len(edit_boundary), len(mask_boundary)),
            "boundary_iou": safe_ratio(len(boundary_aligned), len(boundary_union)),
        },
    }

    return regions, stats


def make_trace(layer_key: str, coords: np.ndarray) -> go.Scatter3d | None:
    if len(coords) == 0:
        return None

    style = LAYER_STYLES[layer_key]
    return go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode="markers",
        name=f"{style.name} ({len(coords)})",
        visible=True if style.visible else "legendonly",
        marker=dict(
            size=style.size,
            color=style.color,
            opacity=style.opacity,
        ),
        hovertemplate=(
            f"{style.name}<br>"
            "x=%{x:.0f}, y=%{y:.0f}, z=%{z:.0f}"
            "<extra></extra>"
        ),
    )


def build_figure(regions: Mapping[str, np.ndarray], stats: Mapping[str, object], title: str) -> go.Figure:
    fig = go.Figure()
    for layer_key in LAYER_STYLES:
        trace = make_trace(layer_key, regions[layer_key])
        if trace is not None:
            fig.add_trace(trace)

    boundary_stats = stats["boundary_alignment"]  # type: ignore[index]
    overlap_stats = stats["mask_overlap"]  # type: ignore[index]

    fig.update_layout(
        title=(
            f"{title}<br>"
            f"<sup>Boundary IoU {boundary_stats['boundary_iou']:.1%} · "
            f"Mask Overlap IoU {overlap_stats['iou']:.1%}</sup>"
        ),
        template="plotly_dark",
        paper_bgcolor="#10141a",
        plot_bgcolor="#10141a",
        width=1280,
        height=920,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=0, r=0, t=88, b=0),
        scene=dict(
            bgcolor="#10141a",
            aspectmode="data",
            xaxis=dict(title="X", backgroundcolor="#10141a", gridcolor="#23303f", zerolinecolor="#23303f"),
            yaxis=dict(title="Y", backgroundcolor="#10141a", gridcolor="#23303f", zerolinecolor="#23303f"),
            zaxis=dict(title="Z", backgroundcolor="#10141a", gridcolor="#23303f", zerolinecolor="#23303f"),
            camera=dict(eye=dict(x=1.55, y=1.45, z=1.2)),
        ),
    )
    return fig


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_summary_cards(stats: Mapping[str, object]) -> str:
    totals = stats["totals"]  # type: ignore[index]
    overlap = stats["mask_overlap"]  # type: ignore[index]
    boundary = stats["boundary_alignment"]  # type: ignore[index]

    cards = [
        ("Source Voxels", f"{totals['source']:,}", "Original occupancy input"),
        ("Edit Voxels", f"{totals['edit']:,}", "Edited occupancy input"),
        ("Mask Voxels", f"{totals['mask']:,}", "Mask occupancy input"),
        ("Mask Overlap IoU", pct(overlap["iou"]), "Source/edit overlap inside mask"),
        ("Boundary IoU", pct(boundary["boundary_iou"]), "Agreement on mask boundary occupancy"),
        ("Boundary Leaks", f"{boundary['edit_leak_near_boundary']:,}", "Edited voxels spilling outside mask near boundary"),
    ]

    return "\n".join(
        (
            '<div class="card">'
            f'<div class="card-label">{escape(label)}</div>'
            f'<div class="card-value">{escape(value)}</div>'
            f'<div class="card-note">{escape(note)}</div>'
            "</div>"
        )
        for label, value, note in cards
    )


def build_metric_table(stats: Mapping[str, object]) -> str:
    overlap = stats["mask_overlap"]  # type: ignore[index]
    boundary = stats["boundary_alignment"]  # type: ignore[index]

    rows = [
        ("Source in mask", f"{overlap['source_in_mask']:,}"),
        ("Edit in mask", f"{overlap['edit_in_mask']:,}"),
        ("In-mask overlap", f"{overlap['overlap']:,}"),
        ("In-mask source only", f"{overlap['source_only']:,}"),
        ("In-mask edit only", f"{overlap['edit_only']:,}"),
        ("Overlap / source-in-mask", pct(overlap["overlap_vs_source_ratio"])),
        ("Overlap / edit-in-mask", pct(overlap["overlap_vs_edit_ratio"])),
        ("Mask boundary voxels", f"{boundary['mask_boundary_voxels']:,}"),
        ("Source on mask boundary", f"{boundary['source_on_mask_boundary']:,}"),
        ("Edit on mask boundary", f"{boundary['edit_on_mask_boundary']:,}"),
        ("Boundary aligned", f"{boundary['aligned']:,}"),
        ("Boundary missing", f"{boundary['missing']:,}"),
        ("Boundary added", f"{boundary['added']:,}"),
        ("Source boundary preservation", pct(boundary["source_boundary_preservation_ratio"])),
        ("Mask boundary coverage by edit", pct(boundary["mask_boundary_coverage_by_edit_ratio"])),
    ]

    return "\n".join(
        f"<tr><td>{escape(label)}</td><td>{escape(value)}</td></tr>"
        for label, value in rows
    )


def build_legend(regions: Mapping[str, np.ndarray]) -> str:
    blocks = []
    for layer_key, style in LAYER_STYLES.items():
        blocks.append(
            '<div class="legend-item">'
            f'<span class="swatch" style="background:{style.color};"></span>'
            f'<div class="legend-copy"><strong>{escape(style.name)}</strong>'
            f'<span>{escape(style.description)}</span>'
            f'<em>{len(regions[layer_key]):,} voxels</em></div>'
            "</div>"
        )
    return "\n".join(blocks)


def build_html_report(
    fig: go.Figure,
    stats: Mapping[str, object],
    regions: Mapping[str, np.ndarray],
    input_paths: Mapping[str, Path],
) -> str:
    plot_html = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=True,
        config={"displaylogo": False, "scrollZoom": True, "responsive": True},
    )

    paths_html = "\n".join(
        f"<li><span>{escape(role.title())}</span><code>{escape(str(path))}</code></li>"
        for role, path in input_paths.items()
    )

    stats_json = escape(json.dumps(stats, indent=2))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Voxel Boundary Alignment Report</title>
  <style>
    :root {{
      --bg: #f1eadf;
      --panel: rgba(255, 252, 246, 0.82);
      --ink: #181716;
      --muted: #5f5a53;
      --line: rgba(24, 23, 22, 0.12);
      --accent: #10141a;
      --shadow: 0 24px 60px rgba(43, 36, 30, 0.16);
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(255, 155, 104, 0.18), transparent 34%),
        radial-gradient(circle at bottom right, rgba(34, 157, 220, 0.12), transparent 38%),
        linear-gradient(160deg, #ece2d2 0%, #f8f4ee 48%, #e8ded0 100%);
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(320px, 420px) 1fr;
      gap: 20px;
      min-height: 100vh;
      padding: 20px;
    }}
    .panel {{
      background: var(--panel);
      backdrop-filter: blur(18px);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .sidebar {{
      padding: 26px 24px 22px;
    }}
    .eyebrow {{
      margin: 0;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: #8b5e34;
    }}
    h1 {{
      margin: 8px 0 6px;
      font-size: 31px;
      line-height: 1.06;
      letter-spacing: -0.03em;
    }}
    .lead {{
      margin: 0 0 22px;
      color: var(--muted);
      line-height: 1.55;
      font-size: 14px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-bottom: 18px;
    }}
    .card {{
      background: rgba(255, 255, 255, 0.55);
      border: 1px solid rgba(24, 23, 22, 0.08);
      border-radius: 18px;
      padding: 14px 14px 13px;
    }}
    .card-label {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #7a736b;
    }}
    .card-value {{
      margin-top: 7px;
      font-size: 28px;
      font-weight: 700;
      letter-spacing: -0.04em;
    }}
    .card-note {{
      margin-top: 5px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }}
    .section {{
      margin-top: 20px;
      padding-top: 20px;
      border-top: 1px solid var(--line);
    }}
    .section h2 {{
      margin: 0 0 12px;
      font-size: 15px;
      letter-spacing: -0.01em;
    }}
    .paths {{
      list-style: none;
      padding: 0;
      margin: 0;
      display: grid;
      gap: 8px;
    }}
    .paths li {{
      display: grid;
      gap: 4px;
      padding: 10px 12px;
      background: rgba(255, 255, 255, 0.56);
      border-radius: 14px;
      border: 1px solid rgba(24, 23, 22, 0.08);
    }}
    .paths span {{
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #7a736b;
    }}
    code, pre {{
      font-family: "IBM Plex Mono", "JetBrains Mono", monospace;
      font-size: 12px;
    }}
    code {{
      word-break: break-all;
      color: #2b2b2b;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    td {{
      padding: 8px 0;
      border-bottom: 1px solid rgba(24, 23, 22, 0.08);
    }}
    td:last-child {{
      text-align: right;
      font-weight: 700;
    }}
    .legend-item {{
      display: grid;
      grid-template-columns: 14px 1fr;
      gap: 10px;
      align-items: start;
      padding: 9px 0;
    }}
    .swatch {{
      width: 14px;
      height: 14px;
      border-radius: 999px;
      margin-top: 2px;
      box-shadow: 0 0 0 1px rgba(24, 23, 22, 0.16);
    }}
    .legend-copy {{
      display: grid;
      gap: 2px;
    }}
    .legend-copy strong {{
      font-size: 13px;
    }}
    .legend-copy span {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }}
    .legend-copy em {{
      font-style: normal;
      color: #6d675f;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .plot-wrap {{
      padding: 10px;
    }}
    .plot-panel {{
      height: calc(100vh - 40px);
      overflow: hidden;
      background: linear-gradient(180deg, #131922 0%, #0b1017 100%);
      border-radius: 24px;
    }}
    .plot-panel > div {{
      height: 100%;
    }}
    details {{
      margin-top: 12px;
      background: rgba(255, 255, 255, 0.45);
      border: 1px solid rgba(24, 23, 22, 0.08);
      border-radius: 14px;
      padding: 10px 12px;
    }}
    summary {{
      cursor: pointer;
      font-weight: 700;
    }}
    pre {{
      white-space: pre-wrap;
      margin: 10px 0 0;
      max-height: 260px;
      overflow: auto;
      color: #403d39;
    }}
    @media (max-width: 1180px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}
      .plot-panel {{
        height: 72vh;
      }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="panel sidebar">
      <p class="eyebrow">Voxel Boundary Report</p>
      <h1>Boundary-first inspection for source, edit, and mask voxels</h1>
      <p class="lead">
        The report suppresses unrelated outer geometry and keeps the view centered on:
        mask boundary occupancy, in-mask overlap, and edited voxels that fail to stay aligned to the boundary.
      </p>

      <div class="cards">
        {build_summary_cards(stats)}
      </div>

      <section class="section">
        <h2>Inputs</h2>
        <ul class="paths">
          {paths_html}
        </ul>
      </section>

      <section class="section">
        <h2>Metrics</h2>
        <table>
          {build_metric_table(stats)}
        </table>
      </section>

      <section class="section">
        <h2>Layers</h2>
        {build_legend(regions)}
      </section>

      <details>
        <summary>Raw Stats JSON</summary>
        <pre>{stats_json}</pre>
      </details>
    </aside>

    <main class="panel plot-wrap">
      <div class="plot-panel">
        {plot_html}
      </div>
    </main>
  </div>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    input_paths = {
        "source": resolve_input_path(args.source, "source"),
        "edit": resolve_input_path(args.edit, "edit"),
        "mask": resolve_input_path(args.mask, "mask"),
    }

    source_coords = load_coords(input_paths["source"], args.resolution)
    edit_coords = load_coords(input_paths["edit"], args.resolution)
    mask_coords = load_coords(input_paths["mask"], args.resolution)

    regions, stats = compute_regions(source_coords, edit_coords, mask_coords, args.resolution)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    html_path = args.output_dir / f"{args.report_name}.html"
    json_path = args.output_dir / f"{args.report_name}_stats.json"

    title = (
        f"Boundary Alignment | "
        f"source={input_paths['source'].name} · "
        f"edit={input_paths['edit'].name} · "
        f"mask={input_paths['mask'].name}"
    )
    fig = build_figure(regions, stats, title)
    html = build_html_report(fig, stats, regions, input_paths)

    html_path.write_text(html, encoding="utf-8")
    json_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Saved HTML report to {html_path}")
    print(f"Saved stats JSON to {json_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
