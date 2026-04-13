#!/usr/bin/env python3
"""Generate a boundary-focused voxel visualization report.

This tool compares source/edit voxel occupancies against a mask voxel region and
produces a standalone HTML report focused on:
1. Mask boundary occupancy
2. In-mask overlap between source and edit
3. Whether edited voxels stay aligned with the boundary

python visualize_boundary_alignment.py --source /home/wangxinxing/3dlocaledit/TRELLIS_EDIT/assets/edit_example/voxels.ply --edit /home/wangxinxing/3dlocaledit/TRELLIS_EDIT/outputs/p2p_latent_blend_ss/test/edit/ss/coords.ply --mask /home/wangxinxing/3dlocaledit/TRELLIS_EDIT/assets/edit_example/voxels_delete.ply --output-dir /home/wangxinxing/3dlocaledit/TRELLIS_EDIT/outputs/boundary_alignment_vis

python visualize_boundary_alignment.py --source /home/wangxinxing/3dlocaledit/TRELLIS_EDIT/assets/Avengers_Gamma_Green_Smash_Fists_prompt_3/voxels.ply --edit /home/wangxinxing/3dlocaledit/TRELLIS_EDIT/outputs/p2p_latent_blend_ss/Avengers_Gamma_Green_Smash_Fists_prompt_3/edit/ss/coords.ply --mask /home/wangxinxing/3dlocaledit/TRELLIS_EDIT/assets/Avengers_Gamma_Green_Smash_Fists_prompt_3/voxels_delete.ply --output-dir /home/wangxinxing/3dlocaledit/TRELLIS_EDIT/outputs/boundary_alignment_vis/Avengers_Gamma_Green_Smash_Fists_prompt_3

python visualize_boundary_alignment.py --source /home/wangxinxing/3dlocaledit/TRELLIS_EDIT/assets/CHICKEN_RACER_prompt_1/voxels.ply --edit /home/wangxinxing/3dlocaledit/TRELLIS_EDIT/outputs/p2p_latent_blend_ss/CHICKEN_RACER_prompt_1/edit/ss/coords.ply --mask /home/wangxinxing/3dlocaledit/TRELLIS_EDIT/assets/CHICKEN_RACER_prompt_1/voxels_delete.ply --output-dir /home/wangxinxing/3dlocaledit/TRELLIS_EDIT/outputs/boundary_alignment_vis/CHICKEN_RACER_prompt_1

"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Dict, Iterable, Mapping, Set, Tuple

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


UNIT_CUBE_VERTICES = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ],
    dtype=np.float32,
)

UNIT_CUBE_FACES = np.asarray(
    [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 7, 3],
        [0, 4, 7],
        [1, 6, 5],
        [1, 2, 6],
        [0, 5, 4],
        [0, 1, 5],
        [3, 6, 2],
        [3, 7, 6],
    ],
    dtype=np.int32,
)


@dataclass(frozen=True)
class LayerStyle:
    name: str
    color: str
    size: float
    opacity: float
    visible: bool
    description: str


LAYER_STYLES: Dict[str, LayerStyle] = {
    "boundary": LayerStyle(
        name="Boundary",
        color="#ff5a5f",
        size=1.0,
        opacity=1.0,
        visible=True,
        description="Source voxels outside the mask that are adjacent to source voxels inside the mask.",
    ),
    "boundary_isolated": LayerStyle(
        name="Boundary Isolated",
        color="#8b0000",
        size=1.2,
        opacity=1.0,
        visible=True,
        description="Boundary voxels with NO edit voxels nearby (neither occupied nor adjacent).",
    ),
    "interface": LayerStyle(
        name="Interface",
        color="#ff9500",
        size=1.0,
        opacity=1.0,
        visible=True,
        description="Source voxels inside the mask that are adjacent to boundary voxels.",
    ),
    "source_only_in_mask": LayerStyle(
        name="Source Only In Mask",
        color="#4c78ff",
        size=1.0,
        opacity=1.0,
        visible=True,
        description="Source-only voxels inside the mask.",
    ),
    "edit_only_in_mask": LayerStyle(
        name="Edit Only In Mask",
        color="#00d27a",
        size=1.0,
        opacity=1.0,
        visible=True,
        description="Edit-only voxels inside the mask.",
    ),
    "overlap_in_mask": LayerStyle(
        name="Overlap In Mask",
        color="#ffd166",
        size=1.0,
        opacity=1.0,
        visible=True,
        description="Source/edit overlapping voxels inside the mask.",
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

    source_in_mask = source_set & mask_set
    edit_in_mask = edit_set & mask_set
    in_mask_overlap = source_in_mask & edit_in_mask
    in_mask_source_only = source_in_mask - edit_in_mask
    in_mask_edit_only = edit_in_mask - source_in_mask

    # Find source voxels outside the mask
    source_outside_mask = source_set - mask_set

    # Boundary voxels = source voxels OUTSIDE mask that are adjacent to source voxels INSIDE mask
    boundary_ring: Set[Coord] = set()
    for x, y, z in source_outside_mask:
        for dx, dy, dz in NEIGHBOR_OFFSETS:
            neighbor = (x + dx, y + dy, z + dz)
            if neighbor in source_in_mask:
                boundary_ring.add((x, y, z))
                break

    # Interface voxels = source voxels INSIDE mask that are adjacent to boundary voxels
    interface_voxels: Set[Coord] = set()
    for x, y, z in boundary_ring:
        for dx, dy, dz in NEIGHBOR_OFFSETS:
            neighbor = (x + dx, y + dy, z + dz)
            if neighbor in source_in_mask:
                interface_voxels.add(neighbor)

    # Boundary alignment metrics
    source_touch_boundary = interface_voxels
    edit_touch_boundary = edit_in_mask & interface_voxels
    boundary_aligned = source_touch_boundary & edit_touch_boundary
    boundary_missing = source_touch_boundary - edit_touch_boundary
    boundary_added = edit_touch_boundary - source_touch_boundary

    # Edit leaks: edit voxels outside mask
    edit_outside_mask = edit_set - mask_set
    edit_leak_near_boundary = edit_outside_mask & boundary_ring

    # Find boundary voxels that have NO edit voxels nearby (neither occupied nor adjacent)
    boundary_isolated_from_edit: Set[Coord] = set()
    for x, y, z in boundary_ring:
        # Check if this boundary voxel itself is occupied by edit
        if (x, y, z) in edit_set:
            continue
        # Check if any neighbor is occupied by edit
        has_edit_neighbor = False
        for dx, dy, dz in NEIGHBOR_OFFSETS:
            neighbor = (x + dx, y + dy, z + dz)
            if neighbor in edit_set:
                has_edit_neighbor = True
                break
        if not has_edit_neighbor:
            boundary_isolated_from_edit.add((x, y, z))

    regions = {
        "boundary": coord_set_to_array(boundary_ring),
        "interface": coord_set_to_array(interface_voxels),
        "source_only_in_mask": coord_set_to_array(in_mask_source_only),
        "edit_only_in_mask": coord_set_to_array(in_mask_edit_only),
        "overlap_in_mask": coord_set_to_array(in_mask_overlap),
        "boundary_isolated": coord_set_to_array(boundary_isolated_from_edit),
    }

    in_mask_union = source_in_mask | edit_in_mask
    boundary_union = source_touch_boundary | edit_touch_boundary

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
            "boundary_voxels": len(boundary_ring),
            "interface_voxels": len(interface_voxels),
            "source_touch_boundary": len(source_touch_boundary),
            "edit_touch_boundary": len(edit_touch_boundary),
            "aligned": len(boundary_aligned),
            "missing": len(boundary_missing),
            "added": len(boundary_added),
            "edit_leak_near_boundary": len(edit_leak_near_boundary),
            "boundary_isolated_from_edit": len(boundary_isolated_from_edit),
            "source_boundary_preservation_ratio": safe_ratio(len(boundary_aligned), len(source_touch_boundary)),
            "boundary_coverage_by_edit_ratio": safe_ratio(len(edit_touch_boundary), len(interface_voxels)),
            "boundary_iou": safe_ratio(len(boundary_aligned), len(boundary_union)),
        },
    }

    return regions, stats


def build_voxel_mesh(coords: np.ndarray, cube_size: float) -> tuple[np.ndarray, np.ndarray]:
    if len(coords) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)

    inset = (1.0 - cube_size) * 0.5
    vertices = np.empty((len(coords) * 8, 3), dtype=np.float32)
    faces = np.empty((len(coords) * 12, 3), dtype=np.int32)

    for idx, coord in enumerate(coords.astype(np.float32, copy=False)):
        base_vertex = idx * 8
        base_face = idx * 12
        offset = coord + inset
        vertices[base_vertex:base_vertex + 8] = UNIT_CUBE_VERTICES * cube_size + offset
        faces[base_face:base_face + 12] = UNIT_CUBE_FACES + base_vertex

    return vertices, faces


def build_voxel_edge_trace(coords: np.ndarray, cube_size: float, visible: bool) -> go.Scatter3d | None:
    if len(coords) == 0:
        return None

    inset = (1.0 - cube_size) * 0.5
    edge_pairs = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )

    xs = []
    ys = []
    zs = []
    for coord in coords.astype(np.float32, copy=False):
        cube_vertices = UNIT_CUBE_VERTICES * cube_size + coord + inset
        for start_idx, end_idx in edge_pairs:
            start_vertex = cube_vertices[start_idx]
            end_vertex = cube_vertices[end_idx]
            xs.extend((float(start_vertex[0]), float(end_vertex[0]), None))
            ys.extend((float(start_vertex[1]), float(end_vertex[1]), None))
            zs.extend((float(start_vertex[2]), float(end_vertex[2]), None))

    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        visible=True if visible else "legendonly",
        showlegend=False,
        line=dict(color="#f8f4ea", width=2),
        hoverinfo="skip",
    )


def make_traces(layer_key: str, coords: np.ndarray) -> list[go.BaseTraceType]:
    if len(coords) == 0:
        return []

    style = LAYER_STYLES[layer_key]
    vertices, faces = build_voxel_mesh(coords, style.size)
    mesh_trace = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        name=f"{style.name} ({len(coords)})",
        visible=True if style.visible else "legendonly",
        color=style.color,
        opacity=style.opacity,
        flatshading=True,
        lighting=dict(
            ambient=0.52,
            diffuse=0.48,
            fresnel=0.0,
            roughness=1.0,
            specular=0.0,
        ),
        lightposition=dict(x=140, y=120, z=220),
        hoverinfo="skip",
    )
    edge_trace = build_voxel_edge_trace(coords, style.size, style.visible)
    traces = [mesh_trace]
    if edge_trace is not None:
        traces.append(edge_trace)
    return traces


def build_figure(regions: Mapping[str, np.ndarray], stats: Mapping[str, object], title: str) -> go.Figure:
    fig = go.Figure()
    for layer_key in LAYER_STYLES:
        for trace in make_traces(layer_key, regions[layer_key]):
            fig.add_trace(trace)

    boundary_stats = stats["boundary_alignment"]  # type: ignore[index]
    overlap_stats = stats["mask_overlap"]  # type: ignore[index]

    fig.update_layout(
        title=(
            f"{title}<br>"
            f"<span style=\"font-size:13px;font-weight:400;opacity:0.88\">"
            f"Boundary IoU {boundary_stats['boundary_iou']:.1%} · "
            f"Mask Overlap IoU {overlap_stats['iou']:.1%}"
            f"</span>"
        ),
        template="plotly_dark",
        paper_bgcolor="#10141a",
        plot_bgcolor="#10141a",
        width=1280,
        height=920,
        showlegend=False,
        margin=dict(l=4, r=4, t=100, b=108),
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
    overlap = stats["mask_overlap"]  # type: ignore[index]
    boundary = stats["boundary_alignment"]  # type: ignore[index]

    cards = [
        ("Boundary Voxels", f"{boundary['boundary_voxels']:,}", "Source voxels outside mask adjacent to source voxels inside mask"),
        ("Boundary Isolated", f"{boundary['boundary_isolated_from_edit']:,}", "Boundary voxels with NO edit voxels nearby"),
        ("Overlap In Mask", f"{overlap['overlap']:,}", "Source/edit overlapping voxels inside the mask"),
        ("Source Only In Mask", f"{overlap['source_only']:,}", "Source-only voxels inside the mask"),
        ("Edit Only In Mask", f"{overlap['edit_only']:,}", "Edit-only voxels inside the mask"),
        ("Boundary IoU", pct(boundary["boundary_iou"]), "Agreement on interior voxels that touch the outer boundary ring"),
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
        ("Total source voxels", f"{stats['totals']['source']:,}"),  # type: ignore[index]
        ("Total edit voxels", f"{stats['totals']['edit']:,}"),  # type: ignore[index]
        ("Source in mask", f"{overlap['source_in_mask']:,}"),
        ("Edit in mask", f"{overlap['edit_in_mask']:,}"),
        ("In-mask overlap", f"{overlap['overlap']:,}"),
        ("Source only in mask", f"{overlap['source_only']:,}"),
        ("Edit only in mask", f"{overlap['edit_only']:,}"),
        ("Overlap / source-in-mask", pct(overlap["overlap_vs_source_ratio"])),
        ("Overlap / edit-in-mask", pct(overlap["overlap_vs_edit_ratio"])),
        ("Boundary voxels", f"{boundary['boundary_voxels']:,}"),
        ("Boundary isolated from edit", f"{boundary['boundary_isolated_from_edit']:,}"),
        ("Boundary interface voxels", f"{boundary['interface_voxels']:,}"),
        ("Source touching boundary", f"{boundary['source_touch_boundary']:,}"),
        ("Edit touching boundary", f"{boundary['edit_touch_boundary']:,}"),
        ("Boundary aligned", f"{boundary['aligned']:,}"),
        ("Boundary missing", f"{boundary['missing']:,}"),
        ("Boundary added", f"{boundary['added']:,}"),
        ("Boundary leaks", f"{boundary['edit_leak_near_boundary']:,}"),
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


def build_toggle_controls(regions: Mapping[str, np.ndarray]) -> str:
    controls = []
    for layer_key, style in LAYER_STYLES.items():
        checked = " checked" if style.visible else ""
        controls.append(
            '<label class="toggle-row">'
            f'<input type="checkbox" data-layer="{escape(layer_key)}"{checked}>'
            f'<span class="toggle-swatch" style="background:{style.color};"></span>'
            f'<span class="toggle-label">{escape(style.name)}</span>'
            f'<span class="toggle-count">{len(regions[layer_key]):,}</span>'
            "</label>"
        )
    return "\n".join(controls)


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
    trace_indices = {}
    current_idx = 0
    for layer_key in LAYER_STYLES:
        if len(regions[layer_key]) > 0:
            trace_indices[layer_key] = [current_idx, current_idx + 1]
            current_idx += 2
    trace_indices_json = json.dumps(trace_indices)

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
    .toggle-panel {{
      display: grid;
      gap: 10px;
    }}
    .toggle-row {{
      display: grid;
      grid-template-columns: 18px 14px 1fr auto;
      align-items: center;
      gap: 10px;
      padding: 10px 12px;
      background: rgba(255, 255, 255, 0.56);
      border-radius: 14px;
      border: 1px solid rgba(24, 23, 22, 0.08);
      cursor: pointer;
    }}
    .toggle-row input {{
      margin: 0;
    }}
    .toggle-swatch {{
      width: 14px;
      height: 14px;
      border-radius: 4px;
      box-shadow: 0 0 0 1px rgba(24, 23, 22, 0.16);
    }}
    .toggle-label {{
      font-size: 13px;
      font-weight: 600;
    }}
    .toggle-count {{
      font-size: 12px;
      color: #6d675f;
      font-variant-numeric: tabular-nums;
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
        The report focuses on boundary alignment between source and edit voxels.
        Boundary voxels (red) are source voxels outside the mask that are adjacent to source voxels inside the mask.
        The visualization shows how well edit voxels preserve the boundary interface.
      </p>

      <section class="section" style="margin-top: 16px; padding-top: 0; border-top: none;">
        <h2>Switches</h2>
        <div class="toggle-panel">
          {build_toggle_controls(regions)}
        </div>
      </section>

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
  <script>
    const TRACE_INDEX = JSON.parse('{trace_indices_json}');
    const plotDiv = document.querySelector('.plot-panel .js-plotly-plot');
    for (const input of document.querySelectorAll('.toggle-row input[type="checkbox"]')) {{
      input.addEventListener('change', () => {{
        const layer = input.dataset.layer;
        const traceIndices = TRACE_INDEX[layer] || [];
        if (plotDiv && traceIndices.length > 0) {{
          Plotly.restyle(plotDiv, {{ visible: input.checked }}, traceIndices);
        }}
      }});
    }}
  </script>
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
