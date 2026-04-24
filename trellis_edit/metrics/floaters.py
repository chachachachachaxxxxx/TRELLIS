from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.csgraph import connected_components


FLOATER_METRIC_NAME = "floaters"
LOCAL_MESH_METRICS = {FLOATER_METRIC_NAME}
DEFAULT_MERGE_PRECISION = 1e-6
DEFAULT_FLOATER_AREA_RATIO_THRESHOLD = 1e-3
DEFAULT_MICRO_FLOATER_AREA_RATIO_THRESHOLD = 1e-4


def split_local_mesh_metrics(metrics: Iterable[str]) -> tuple[list[str], list[str]]:
    """Return (external_metrics, local_mesh_metrics) while preserving user order."""
    external_metrics: list[str] = []
    local_metrics: list[str] = []
    seen_local: set[str] = set()
    for raw_metric in metrics:
        metric = str(raw_metric).strip()
        if not metric:
            continue
        metric_key = metric.lower()
        if metric_key in LOCAL_MESH_METRICS:
            if metric_key not in seen_local:
                local_metrics.append(metric_key)
                seen_local.add(metric_key)
            continue
        external_metrics.append(metric)
    return external_metrics, local_metrics


def compute_local_mesh_metric_payloads(
    *,
    gt_root: Path,
    pred_root: Path,
    metrics: Iterable[str],
) -> dict[str, dict[str, Any]]:
    requested = {str(metric).strip().lower() for metric in metrics}
    payloads: dict[str, dict[str, Any]] = {}
    if FLOATER_METRIC_NAME in requested:
        payloads[FLOATER_METRIC_NAME] = compute_floaters_payload(
            gt_root=gt_root,
            pred_root=pred_root,
        )
    return payloads


def compute_floaters_payload(
    *,
    gt_root: Path,
    pred_root: Path,
    merge_precision: float = DEFAULT_MERGE_PRECISION,
    floater_area_ratio_threshold: float = DEFAULT_FLOATER_AREA_RATIO_THRESHOLD,
    micro_floater_area_ratio_threshold: float = DEFAULT_MICRO_FLOATER_AREA_RATIO_THRESHOLD,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    scores: list[float] = []
    eval_ids: list[str] = []
    errors: list[dict[str, str]] = []

    for case in _iter_predicted_cases(gt_root=gt_root, pred_root=pred_root):
        try:
            row = analyze_mesh_floaters(
                case["edit_glb"],
                merge_precision=merge_precision,
                floater_area_ratio_threshold=floater_area_ratio_threshold,
                micro_floater_area_ratio_threshold=micro_floater_area_ratio_threshold,
            )
        except Exception as exc:
            errors.append(
                {
                    "eval_id": case["eval_id"],
                    "case_id": case["case_id"],
                    "path": str(case["edit_glb"]),
                    "error": str(exc),
                }
            )
            continue

        row.update(
            {
                "eval_id": case["eval_id"],
                "case_id": case["case_id"],
                "path": str(case["edit_glb"]),
            }
        )
        rows.append(row)
        eval_ids.append(case["eval_id"])
        scores.append(float(row["floater_count_lt_0p1pct"]))

    score_array = np.asarray(scores, dtype=np.float64)
    payload: dict[str, Any] = {
        "mean": float(score_array.mean()) if len(score_array) else None,
        "std": float(score_array.std()) if len(score_array) else None,
        "count": int(len(score_array)),
        "all_scores": scores,
        "all_ids": eval_ids,
        "details": rows,
        "definition": {
            "score": "candidate floater count per edit.glb",
            "main_component": "largest connected component by surface area after coordinate merge",
            "candidate_floater": (
                f"non-main component with area_ratio < {floater_area_ratio_threshold:g}"
            ),
            "micro_floater": (
                f"non-main component with area_ratio < {micro_floater_area_ratio_threshold:g}"
            ),
            "merge_precision": merge_precision,
        },
    }
    if errors:
        payload["errors"] = errors
    return payload


def analyze_mesh_floaters(
    mesh_path: Path,
    *,
    merge_precision: float = DEFAULT_MERGE_PRECISION,
    floater_area_ratio_threshold: float = DEFAULT_FLOATER_AREA_RATIO_THRESHOLD,
    micro_floater_area_ratio_threshold: float = DEFAULT_MICRO_FLOATER_AREA_RATIO_THRESHOLD,
) -> dict[str, Any]:
    mesh = _load_mesh(mesh_path)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    base_row: dict[str, Any] = {
        "faces": int(len(faces)),
        "vertices": int(len(vertices)),
    }
    if len(vertices) == 0 or len(faces) == 0:
        return {
            **base_row,
            **_empty_floaters_stats(),
        }

    merged_vertices, remapped_faces = _merge_mesh_vertices(
        vertices=vertices,
        faces=faces,
        precision=merge_precision,
    )
    if len(remapped_faces) == 0:
        return {
            **base_row,
            **_empty_floaters_stats(),
        }

    face_area = _face_areas(merged_vertices, remapped_faces)
    valid_faces = face_area > 0
    remapped_faces = remapped_faces[valid_faces]
    face_area = face_area[valid_faces]
    if len(remapped_faces) == 0:
        return {
            **base_row,
            **_empty_floaters_stats(),
        }

    component_rows = _surface_components(
        vertices=merged_vertices,
        faces=remapped_faces,
        face_area=face_area,
    )
    total_area = float(face_area.sum())
    total_component_faces = sum(row["faces"] for row in component_rows)
    if total_area <= 0 or not component_rows:
        return {
            **base_row,
            **_empty_floaters_stats(),
        }

    main_index = max(range(len(component_rows)), key=lambda index: component_rows[index]["area"])
    non_main = [row for index, row in enumerate(component_rows) if index != main_index]
    non_main_area = sum(row["area"] for row in non_main)
    non_main_faces = sum(row["faces"] for row in non_main)
    floaters = [
        row
        for row in non_main
        if row["area"] / total_area < floater_area_ratio_threshold
    ]
    micro_floaters = [
        row
        for row in non_main
        if row["area"] / total_area < micro_floater_area_ratio_threshold
    ]
    largest_non_main = max(non_main, key=lambda row: row["area"], default={"area": 0.0, "faces": 0})

    return {
        **base_row,
        "component_count": int(len(component_rows)),
        "non_main_component_count": int(len(non_main)),
        "non_main_area_ratio": _safe_ratio(non_main_area, total_area),
        "non_main_face_ratio": _safe_ratio(non_main_faces, total_component_faces),
        "floater_count_lt_0p1pct": int(len(floaters)),
        "floater_area_ratio_lt_0p1pct": _safe_ratio(
            sum(row["area"] for row in floaters),
            total_area,
        ),
        "micro_floater_count_lt_0p01pct": int(len(micro_floaters)),
        "micro_floater_area_ratio_lt_0p01pct": _safe_ratio(
            sum(row["area"] for row in micro_floaters),
            total_area,
        ),
        "largest_non_main_area_ratio": _safe_ratio(largest_non_main["area"], total_area),
        "largest_non_main_faces": int(largest_non_main["faces"]),
    }


def _iter_predicted_cases(*, gt_root: Path, pred_root: Path) -> list[dict[str, Any]]:
    metadata_path = gt_root / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        cases: list[dict[str, Any]] = []
        for item in metadata:
            dataset = item.get("dataset")
            object_name = item.get("source_model")
            if not dataset or not object_name:
                continue
            for prompt_id in (1, 2, 3):
                prompt_key = f"prompt_{prompt_id}"
                edit_glb = pred_root / dataset / object_name / prompt_key / "edit.glb"
                if not edit_glb.is_file():
                    continue
                cases.append(
                    {
                        "eval_id": f"{dataset}_{object_name}_{prompt_key}",
                        "case_id": f"{dataset}/{object_name}/{prompt_key}",
                        "edit_glb": edit_glb,
                    }
                )
        if cases:
            return cases

    cases = []
    for edit_glb in sorted(pred_root.glob("*/**/prompt_*/edit.glb")):
        rel = edit_glb.relative_to(pred_root)
        if not rel.parts or rel.parts[0].startswith("_") or len(rel.parts) < 4:
            continue
        prompt_key = rel.parts[-2]
        if not prompt_key.startswith("prompt_"):
            continue
        dataset = rel.parts[0]
        object_name = "/".join(rel.parts[1:-2])
        cases.append(
            {
                "eval_id": f"{dataset}_{object_name}_{prompt_key}",
                "case_id": f"{dataset}/{object_name}/{prompt_key}",
                "edit_glb": edit_glb,
            }
        )
    return cases


def _load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    if isinstance(loaded, list):
        loaded = trimesh.util.concatenate(loaded)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh object {type(loaded)!r}: {mesh_path}")
    return loaded


def _merge_mesh_vertices(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    precision: float,
) -> tuple[np.ndarray, np.ndarray]:
    vertex_keys = np.round(vertices / precision).astype(np.int64)
    merged_keys, inverse = np.unique(vertex_keys, axis=0, return_inverse=True)
    merged_vertices = merged_keys.astype(np.float64) * precision
    remapped_faces = inverse[faces]
    valid = (
        (remapped_faces[:, 0] != remapped_faces[:, 1])
        & (remapped_faces[:, 1] != remapped_faces[:, 2])
        & (remapped_faces[:, 0] != remapped_faces[:, 2])
    )
    return merged_vertices, remapped_faces[valid]


def _face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    return 0.5 * np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )


def _surface_components(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_area: np.ndarray,
) -> list[dict[str, Any]]:
    edges = np.vstack(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ]
    )
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    graph = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
        shape=(len(vertices), len(vertices)),
    ).tocsr()
    _, labels = connected_components(graph, directed=False, return_labels=True)
    face_labels = labels[faces[:, 0]]
    component_ids, inverse = np.unique(face_labels, return_inverse=True)
    component_area = np.bincount(inverse, weights=face_area)
    component_faces = np.bincount(inverse)
    return [
        {
            "component_id": int(component_ids[index]),
            "area": float(component_area[index]),
            "faces": int(component_faces[index]),
        }
        for index in range(len(component_ids))
        if component_area[index] > 0
    ]


def _empty_floaters_stats() -> dict[str, Any]:
    return {
        "component_count": 0,
        "non_main_component_count": 0,
        "non_main_area_ratio": 0.0,
        "non_main_face_ratio": 0.0,
        "floater_count_lt_0p1pct": 0,
        "floater_area_ratio_lt_0p1pct": 0.0,
        "micro_floater_count_lt_0p01pct": 0,
        "micro_floater_area_ratio_lt_0p01pct": 0.0,
        "largest_non_main_area_ratio": 0.0,
        "largest_non_main_faces": 0,
    }


def _safe_ratio(numerator: float | int, denominator: float | int) -> float:
    denominator_float = float(denominator)
    if denominator_float == 0:
        return 0.0
    return float(numerator) / denominator_float

