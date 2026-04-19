#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import trimesh

from trellis_edit.common import ensure_dir, utc_now_iso, write_json


Coord = tuple[int, int, int]
NEIGHBOR_OFFSETS: tuple[Coord, ...] = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)

ROLE_DEFAULTS = {
    "source": ("coords.pt", "coords.ply", "voxels.ply"),
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

EXPOSED_FACE_TRIANGLES: dict[Coord, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    (0, 0, -1): ((0, 2, 1), (0, 3, 2)),
    (0, 0, 1): ((4, 5, 6), (4, 6, 7)),
    (-1, 0, 0): ((0, 7, 3), (0, 4, 7)),
    (1, 0, 0): ((1, 6, 5), (1, 2, 6)),
    (0, -1, 0): ((0, 5, 4), (0, 1, 5)),
    (0, 1, 0): ((3, 6, 2), (3, 7, 6)),
}

METRIC_DESCRIPTIONS = {
    "outside_change_ratio": (
        "Mask 外改动比例。公式为 |(S_out Δ E_out)| / |S_out|，主要作为 sanity check。"
    ),
    "boundary_isolation_ratio": (
        "Mask 内 interface 体素中，没有被 edit in-mask 体素占据或 6 邻接接住的比例。越低越好。"
    ),
    "in_mask_change_ratio": (
        "Mask 内总体改动比例。公式为 |(S_in Δ E_in)| / |M|。"
    ),
    "in_mask_largest_component_ratio": (
        "Mask 内 edit 体素最大连通块占 edit_in_mask 的比例。越高说明越不碎。"
    ),
    "in_mask_isolated_voxel_ratio": (
        "Mask 内 edit 体素里 6 邻接度为 0 的孤立点比例。越低越好。"
    ),
    "largest_component_is_watertight": (
        "取完整 edit 体素集合 E 的最大 6 邻接连通体，只保留其外表面暴露面生成 cubic surface mesh，"
        "再检查 trimesh.is_watertight。True 表示这个最大主体表面是闭合水密的。"
    ),
}


@dataclass(frozen=True)
class SparseVoxelSSEvaluatorInputs:
    source: Path
    edit: Path
    mask: Path


def resolve_voxel_input_path(raw_path: str | Path, role: str) -> Path:
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


def _coords3d(coords: torch.Tensor) -> torch.Tensor:
    if coords.ndim != 2:
        raise RuntimeError(f"Expected coords tensor with shape [N, 3] or [N, 4], got {tuple(coords.shape)}")
    if coords.shape[1] == 4:
        coords = coords[:, 1:]
    if coords.shape[1] != 3:
        raise RuntimeError(f"Expected coords tensor with shape [N, 3] or [N, 4], got {tuple(coords.shape)}")
    return torch.unique(coords.int().cpu(), dim=0)


def _load_coords_from_file(coords_path: Path, device: torch.device, resolution: int = 64) -> torch.Tensor:
    if coords_path.suffix == ".pt":
        coords = torch.load(coords_path, map_location=device)
        return _coords3d(coords)

    if coords_path.suffix == ".npz":
        payload = np.load(coords_path)
        for key in ("coords", "indices"):
            if key in payload:
                return _coords3d(torch.from_numpy(payload[key]).to(device=device))
        raise ValueError(f"Unsupported npz keys in {coords_path}")

    if coords_path.suffix == ".ply":
        geometry = trimesh.load(str(coords_path), process=False)
        if not hasattr(geometry, "vertices"):
            raise ValueError(f"PLY file does not expose vertices: {coords_path}")
        positions = np.asarray(geometry.vertices, dtype=np.float32)
        coords = ((torch.from_numpy(positions) + 0.5) * resolution).int()
        coords = torch.clamp(coords, min=0, max=resolution - 1)
        return _coords3d(coords.to(device=device))

    raise ValueError(f"Unsupported coords file format: {coords_path.suffix}")


def _coord_set(coords: torch.Tensor) -> set[Coord]:
    if coords.numel() == 0:
        return set()
    return set(map(tuple, coords.tolist()))


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _neighbor_coords(coord: Coord) -> Iterable[Coord]:
    x, y, z = coord
    for dx, dy, dz in NEIGHBOR_OFFSETS:
        yield x + dx, y + dy, z + dz


def _build_boundary_ring(source_inside_mask: set[Coord], source_outside_mask: set[Coord]) -> set[Coord]:
    boundary_ring: set[Coord] = set()
    for coord in source_outside_mask:
        if any(neighbor in source_inside_mask for neighbor in _neighbor_coords(coord)):
            boundary_ring.add(coord)
    return boundary_ring


def _build_interface_voxels(boundary_ring: set[Coord], source_inside_mask: set[Coord]) -> set[Coord]:
    interface_voxels: set[Coord] = set()
    for coord in boundary_ring:
        for neighbor in _neighbor_coords(coord):
            if neighbor in source_inside_mask:
                interface_voxels.add(neighbor)
    return interface_voxels


def _has_in_mask_edit_support(coord: Coord, edit_inside_mask: set[Coord]) -> bool:
    if coord in edit_inside_mask:
        return True
    return any(neighbor in edit_inside_mask for neighbor in _neighbor_coords(coord))


def _largest_component_size(coords: set[Coord]) -> int:
    return len(_largest_component_coords(coords))


def _largest_component_coords(coords: set[Coord]) -> set[Coord]:
    remaining = set(coords)
    largest: set[Coord] = set()
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = {start}
        while stack:
            current = stack.pop()
            for neighbor in _neighbor_coords(current):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    component.add(neighbor)
        if len(component) > len(largest):
            largest = component
    return largest


def _isolated_voxel_count(coords: set[Coord]) -> int:
    isolated = 0
    for coord in coords:
        if not any(neighbor in coords for neighbor in _neighbor_coords(coord)):
            isolated += 1
    return isolated


def _build_exposed_surface_mesh(coords: set[Coord]) -> trimesh.Trimesh:
    vertex_map: dict[tuple[float, float, float], int] = {}
    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    for coord in coords:
        coord_offset = np.asarray(coord, dtype=np.float32)
        cube_vertices = UNIT_CUBE_VERTICES + coord_offset
        for face_dir, triangles in EXPOSED_FACE_TRIANGLES.items():
            neighbor = (coord[0] + face_dir[0], coord[1] + face_dir[1], coord[2] + face_dir[2])
            if neighbor in coords:
                continue

            for triangle in triangles:
                face: list[int] = []
                for vertex_idx in triangle:
                    vertex = tuple(float(value) for value in cube_vertices[vertex_idx])
                    mapped = vertex_map.get(vertex)
                    if mapped is None:
                        mapped = len(vertices)
                        vertex_map[vertex] = mapped
                        vertices.append(list(vertex))
                    face.append(mapped)
                faces.append(face)

    if not faces:
        return trimesh.Trimesh(
            vertices=np.zeros((0, 3), dtype=np.float32),
            faces=np.zeros((0, 3), dtype=np.int64),
            process=False,
        )

    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    mesh.remove_unreferenced_vertices()
    return mesh


def _is_component_watertight(coords: set[Coord]) -> bool:
    if not coords:
        return False
    mesh = _build_exposed_surface_mesh(coords)
    if len(mesh.faces) == 0:
        return False
    return bool(mesh.is_watertight)


def compute_sparse_voxel_ss_metrics(
    source_coords: torch.Tensor,
    edit_coords: torch.Tensor,
    mask_coords: torch.Tensor,
) -> dict[str, object]:
    source_set = _coord_set(_coords3d(source_coords))
    edit_set = _coord_set(_coords3d(edit_coords))
    mask_set = _coord_set(_coords3d(mask_coords))

    source_inside_mask = source_set & mask_set
    source_outside_mask = source_set - mask_set
    edit_inside_mask = edit_set & mask_set
    edit_outside_mask = edit_set - mask_set

    outside_change = source_outside_mask.symmetric_difference(edit_outside_mask)
    boundary_ring = _build_boundary_ring(source_inside_mask, source_outside_mask)
    interface_voxels = _build_interface_voxels(boundary_ring, source_inside_mask)
    isolated_interface = {
        coord for coord in interface_voxels if not _has_in_mask_edit_support(coord, edit_inside_mask)
    }

    in_mask_changed = source_inside_mask.symmetric_difference(edit_inside_mask)
    largest_in_mask_component = _largest_component_coords(edit_inside_mask) if edit_inside_mask else set()
    largest_component_size = len(largest_in_mask_component)
    isolated_voxel_count = _isolated_voxel_count(edit_inside_mask) if edit_inside_mask else 0
    largest_edit_component = _largest_component_coords(edit_set) if edit_set else set()

    metrics = {
        "outside_change_ratio": _safe_ratio(len(outside_change), len(source_outside_mask)),
        "boundary_isolation_ratio": _safe_ratio(len(isolated_interface), len(interface_voxels)),
        "in_mask_change_ratio": _safe_ratio(len(in_mask_changed), len(mask_set)),
        "in_mask_largest_component_ratio": _safe_ratio(largest_component_size, len(edit_inside_mask)),
        "in_mask_isolated_voxel_ratio": _safe_ratio(isolated_voxel_count, len(edit_inside_mask)),
        "largest_component_is_watertight": _is_component_watertight(largest_edit_component),
    }

    counts = {
        "source_voxels": len(source_set),
        "edit_voxels": len(edit_set),
        "mask_voxels": len(mask_set),
        "source_inside_mask": len(source_inside_mask),
        "source_outside_mask": len(source_outside_mask),
        "edit_inside_mask": len(edit_inside_mask),
        "edit_outside_mask": len(edit_outside_mask),
        "outside_changed_voxels": len(outside_change),
        "boundary_ring_voxels": len(boundary_ring),
        "interface_voxels": len(interface_voxels),
        "isolated_interface_voxels": len(isolated_interface),
        "in_mask_changed_voxels": len(in_mask_changed),
        "largest_in_mask_component_voxels": largest_component_size,
        "isolated_in_mask_voxels": isolated_voxel_count,
        "largest_edit_component_voxels": len(largest_edit_component),
    }

    return {
        "metrics": metrics,
        "counts": counts,
    }


def evaluate_sparse_voxel_ss(
    inputs: SparseVoxelSSEvaluatorInputs,
    *,
    output_dir: Path | None = None,
) -> dict[str, object]:
    device = torch.device("cpu")
    source_path = resolve_voxel_input_path(inputs.source, role="source")
    edit_path = resolve_voxel_input_path(inputs.edit, role="edit")
    mask_path = resolve_voxel_input_path(inputs.mask, role="mask")

    source_coords = _load_coords_from_file(source_path, device)
    edit_coords = _load_coords_from_file(edit_path, device)
    mask_coords = _load_coords_from_file(mask_path, device)

    payload = {
        "timestamp_utc": utc_now_iso(),
        "input_paths": {
            "source": str(source_path),
            "edit": str(edit_path),
            "mask": str(mask_path),
        },
        "metric_descriptions": dict(METRIC_DESCRIPTIONS),
        **compute_sparse_voxel_ss_metrics(source_coords, edit_coords, mask_coords),
    }

    if output_dir is not None:
        output_root = ensure_dir(output_dir)
        write_json(output_root / "ss_evaluation.json", payload)

    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate sparse-voxel SS editing outputs with local geometry metrics including watertightness. "
            "Example: python run_sparse_voxel_ss_evaluator.py "
            "--source assets/edit_example/voxels.ply "
            "--mask assets/edit_example/voxels_delete.ply "
            "--edit outputs/p2p_latent_blend_ss/test/edit/ss/coords.ply"
        )
    )
    parser.add_argument("--source", required=True, help="Source voxel file or directory.")
    parser.add_argument("--edit", required=True, help="Edited voxel file or directory.")
    parser.add_argument("--mask", required=True, help="Mask voxel file or directory.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory. Defaults to <edit_parent>/sparse_voxel_ss_eval.",
    )
    return parser


def default_output_dir(edited_coords: str | Path) -> Path:
    path = Path(edited_coords).expanduser().resolve()
    base_dir = path if path.is_dir() else path.parent
    return base_dir / "sparse_voxel_ss_eval"


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output_dir(args.edit)
    result = evaluate_sparse_voxel_ss(
        SparseVoxelSSEvaluatorInputs(
            source=Path(args.source),
            edit=Path(args.edit),
            mask=Path(args.mask),
        ),
        output_dir=output_dir,
    )

    metrics = result["metrics"]
    print(f"Saved SS evaluation to {output_dir / 'ss_evaluation.json'}")
    print(f"outside_change_ratio: {format_pct(metrics['outside_change_ratio'])}")
    print(f"boundary_isolation_ratio: {format_pct(metrics['boundary_isolation_ratio'])}")
    print(f"in_mask_change_ratio: {format_pct(metrics['in_mask_change_ratio'])}")
    print(f"in_mask_largest_component_ratio: {format_pct(metrics['in_mask_largest_component_ratio'])}")
    print(f"in_mask_isolated_voxel_ratio: {format_pct(metrics['in_mask_isolated_voxel_ratio'])}")
    print(f"largest_component_is_watertight: {'yes' if metrics['largest_component_is_watertight'] else 'no'}")


if __name__ == "__main__":
    main()
