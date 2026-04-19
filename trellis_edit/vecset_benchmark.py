from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


DEFAULT_GT_ROOT = Path("/cache/wangxinxing/data/trellis_edit_benchmark/edit3d_data/data")
DEFAULT_BENCHMARK_ROOT = Path("/cache/wangxinxing/data/trellis_edit_benchmark")
DEFAULT_METRICS = [
    "psnr",
    "ssim",
    "lpips",
    "fid",
    "dino_if_max",
    "dino_if_mean",
    "chamfer",
    "clip_t",
]
VECSET_EDIT_ROOT = Path("/home/wangxinxing/3dlocaledit/VecSet-Edit")


def ensure_vecset_edit_root() -> Path:
    repo_root = VECSET_EDIT_ROOT.expanduser().resolve()
    if not repo_root.is_dir():
        raise RuntimeError(f"Cannot locate VecSet-Edit repo: {repo_root}")
    return repo_root


def ensure_vecset_import_path() -> Path:
    repo_root = ensure_vecset_edit_root()
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


def load_benchmark_metadata(gt_root: Path) -> list[dict[str, Any]]:
    metadata_path = gt_root / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def case_id(dataset: str, object_name: str, prompt_id: int) -> str:
    return f"{dataset}/{object_name}/prompt_{prompt_id}"


def select_prompt_cases(
    metadata: list[dict[str, Any]],
    *,
    gt_root: Path,
    limit: int,
    dataset: str | None = None,
    object_name: str | None = None,
    prompt_id: int | None = None,
    require_source_glb: bool = True,
    require_edit_image: bool = True,
    require_render_image: bool = False,
    require_mask_image: bool = False,
) -> list[tuple[str, str, int]]:
    cases: list[tuple[str, str, int]] = []
    for row in metadata:
        row_dataset = str(row.get("dataset") or "")
        row_object_name = str(row.get("source_model") or "")
        if not row_dataset or not row_object_name:
            continue
        if dataset and row_dataset != dataset:
            continue
        if object_name and row_object_name != object_name:
            continue

        source_glb = gt_root / row_dataset / row_object_name / "source_model" / "model.glb"
        if require_source_glb and not source_glb.is_file():
            continue

        for candidate_prompt_id in (1, 2, 3):
            if prompt_id is not None and candidate_prompt_id != prompt_id:
                continue
            prompt_dir = gt_root / row_dataset / row_object_name / f"prompt_{candidate_prompt_id}"
            if not prompt_dir.is_dir():
                continue
            if not row.get(f"prompt_{candidate_prompt_id}"):
                continue
            if require_edit_image and not (prompt_dir / "2d_edit.png").is_file():
                continue
            if require_render_image and not (prompt_dir / "2d_render.png").is_file():
                continue
            if require_mask_image and not (prompt_dir / "2d_mask.png").is_file():
                continue
            cases.append((row_dataset, row_object_name, candidate_prompt_id))
            if len(cases) >= limit:
                return cases
    return cases


def group_cases_by_object(cases: list[tuple[str, str, int]]) -> dict[tuple[str, str], list[int]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for dataset, object_name, prompt_id in cases:
        grouped[(dataset, object_name)].append(prompt_id)
    return dict(grouped)


def select_first_prompt_case(
    metadata: list[dict[str, Any]],
    *,
    gt_root: Path,
    require_render_image: bool,
    require_mask_image: bool,
) -> tuple[str, str, int]:
    cases = select_prompt_cases(
        metadata,
        gt_root=gt_root,
        limit=1,
        require_source_glb=True,
        require_edit_image=True,
        require_render_image=require_render_image,
        require_mask_image=require_mask_image,
    )
    if not cases:
        raise RuntimeError("No valid prompt-level case found in benchmark metadata.")
    return cases[0]


def select_first_object_cases(
    metadata: list[dict[str, Any]],
    *,
    gt_root: Path,
    require_render_image: bool,
) -> tuple[str, str, list[int]]:
    cases = select_prompt_cases(
        metadata,
        gt_root=gt_root,
        limit=300,
        require_source_glb=True,
        require_edit_image=False,
        require_render_image=require_render_image,
        require_mask_image=False,
    )
    grouped = group_cases_by_object(cases)
    if not grouped:
        raise RuntimeError("No valid object-level case found in benchmark metadata.")
    (dataset, object_name), prompt_ids = next(iter(grouped.items()))
    return dataset, object_name, sorted(prompt_ids)


def default_render_gpu_ids(device: str) -> list[str]:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_devices:
        return [item.strip() for item in visible_devices.split(",") if item.strip()]
    if device.startswith("cuda:"):
        return [device.split(":", 1)[1]]
    return []


def parse_render_gpu_ids(raw_value: str, *, device: str) -> list[str]:
    if not raw_value.strip():
        return default_render_gpu_ids(device)
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def run_command(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, env=env, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def load_total_seconds(pred_root: Path) -> float:
    summary_path = pred_root / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return float(summary.get("total_seconds") or 0.0)

    manifest_path = pred_root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return float(manifest.get("total_elapsed_seconds") or 0.0)

    return 0.0


def load_scene_bounds(glb_path: Path | str) -> dict[str, Any]:
    scene = trimesh.load(Path(glb_path), force="scene")
    bounds = np.asarray(scene.bounds, dtype=np.float64)
    center = (bounds[0] + bounds[1]) * 0.5
    extent = bounds[1] - bounds[0]
    return {
        "min": bounds[0].tolist(),
        "max": bounds[1].tolist(),
        "center": center.tolist(),
        "extent": extent.tolist(),
        "max_extent": float(extent.max()),
    }


def compare_bbox_alignment(
    *,
    reference_glb: Path | str,
    candidate_glb: Path | str,
    max_center_delta_ratio: float = 0.05,
    max_extent_ratio_deviation: float = 0.15,
) -> dict[str, Any]:
    reference_bounds = load_scene_bounds(reference_glb)
    candidate_bounds = load_scene_bounds(candidate_glb)

    reference_center = np.asarray(reference_bounds["center"], dtype=np.float64)
    candidate_center = np.asarray(candidate_bounds["center"], dtype=np.float64)
    center_delta = candidate_center - reference_center
    center_delta_norm = float(np.linalg.norm(center_delta))

    reference_max_extent = max(float(reference_bounds["max_extent"]), 1e-8)
    candidate_max_extent = float(candidate_bounds["max_extent"])
    center_delta_ratio = center_delta_norm / reference_max_extent
    extent_ratio = candidate_max_extent / reference_max_extent
    extent_ratio_deviation = abs(extent_ratio - 1.0)

    passed = (
        center_delta_ratio <= max_center_delta_ratio
        and extent_ratio_deviation <= max_extent_ratio_deviation
    )
    return {
        "reference_glb": str(Path(reference_glb).expanduser().resolve()),
        "candidate_glb": str(Path(candidate_glb).expanduser().resolve()),
        "reference_bounds": reference_bounds,
        "candidate_bounds": candidate_bounds,
        "center_delta": center_delta.tolist(),
        "center_delta_norm": center_delta_norm,
        "center_delta_ratio": center_delta_ratio,
        "extent_ratio": extent_ratio,
        "extent_ratio_deviation": extent_ratio_deviation,
        "thresholds": {
            "max_center_delta_ratio": float(max_center_delta_ratio),
            "max_extent_ratio_deviation": float(max_extent_ratio_deviation),
        },
        "passed": bool(passed),
    }
