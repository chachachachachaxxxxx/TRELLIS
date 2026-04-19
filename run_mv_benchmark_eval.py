#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from trellis_edit.benchmarking import (
    _bundle_relpath,
    _case_id,
    _link_or_copy,
    _render_daily_run_pages,
    _summarize_statuses,
    _write_jsonl,
    build_daily_index,
    build_focus_index,
    build_run_id,
)
from trellis_edit.common import ensure_dir, write_json


DEFAULT_MV_DATASET_ROOT = Path(
    "/cache/wangxinxing/data/trellis_edit_benchmark/edit3d_mv_pseudosource_micro10"
)
DEFAULT_DAILY_ROOT = Path("/cache/wangxinxing/data/trellis_edit_benchmark/daily")
DEFAULT_IMAGE_SIZE = 512
DEFAULT_RENDER_SAMPLES = 64
MV_METRIC_NAMES = (
    "lpips_in_seen",
    "ssim_in_seen",
    "lpips_in_novel",
    "ssim_in_novel",
    "dino_if_seen",
    "dino_if_novel",
    "chamfer_target",
    "cross_seed_novel_lpips",
)


def _bootstrap_edit3d_bench() -> Path:
    repo_root = Path(__file__).resolve().parent
    candidates = [
        (repo_root / "../VoxHammer/Edit3D-Bench").resolve(),
        (repo_root / "VoxHammer/Edit3D-Bench").resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    raise RuntimeError("Cannot locate Edit3D-Bench directory.")


EDIT3D_BENCH_ROOT = _bootstrap_edit3d_bench()


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


eval_modules_pkg = types.ModuleType("eval_modules")
eval_modules_pkg.__path__ = [str(EDIT3D_BENCH_ROOT / "eval_modules")]
sys.modules.setdefault("eval_modules", eval_modules_pkg)
bench_render = None
GeometryMetricsEvaluator = None
ImageMetricsEvaluator = None
build_include_mask = None
load_prepared_rgb_image = None


def _ensure_mv_modules_loaded() -> None:
    global bench_render
    global GeometryMetricsEvaluator
    global ImageMetricsEvaluator
    global build_include_mask
    global load_prepared_rgb_image

    if (
        bench_render is not None
        and GeometryMetricsEvaluator is not None
        and ImageMetricsEvaluator is not None
        and build_include_mask is not None
        and load_prepared_rgb_image is not None
    ):
        return

    mask_utils_module = _load_module(
        "eval_modules.mask_utils",
        EDIT3D_BENCH_ROOT / "eval_modules" / "mask_utils.py",
    )
    _load_module(
        "eval_modules.data_loader",
        EDIT3D_BENCH_ROOT / "eval_modules" / "data_loader.py",
    )
    image_metrics_module = _load_module(
        "eval_modules.image_metrics",
        EDIT3D_BENCH_ROOT / "eval_modules" / "image_metrics.py",
    )
    geometry_metrics_module = _load_module(
        "eval_modules.geometry_metrics",
        EDIT3D_BENCH_ROOT / "eval_modules" / "geometry_metrics.py",
    )
    bench_render = _load_module("edit3d_bench_render", EDIT3D_BENCH_ROOT / "render.py")

    GeometryMetricsEvaluator = geometry_metrics_module.GeometryMetricsEvaluator
    ImageMetricsEvaluator = image_metrics_module.ImageMetricsEvaluator
    build_include_mask = mask_utils_module.build_include_mask
    load_prepared_rgb_image = mask_utils_module.load_prepared_rgb_image


@dataclass(frozen=True)
class ViewSpec:
    view_id: str
    gt_path: Path
    mask_path: Path | None


@dataclass(frozen=True)
class MvCase:
    case_id: str
    dataset: str
    object_name: str
    prompt_id: int
    case_root: Path
    payload: dict[str, Any]

    @property
    def prompt_key(self) -> str:
        return f"prompt_{self.prompt_id}"


@dataclass(frozen=True)
class EvaluatedMvCase:
    case: MvCase
    pred_model: Path | None
    metrics: dict[str, float | None]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_daily_run_roots(daily_root: Path) -> list[Path]:
    run_roots = []
    for child in daily_root.iterdir():
        if not child.is_dir() or child.name == "groups":
            continue
        if (child / "manifest.json").is_file():
            run_roots.append(child)
    return sorted(run_roots)


def _load_mv_cases(dataset_root: Path) -> list[MvCase]:
    cases_dir = dataset_root / "cases"
    cases: list[MvCase] = []
    for case_root in sorted(cases_dir.iterdir()):
        if not case_root.is_dir():
            continue
        payload = _read_json(case_root / "case.json")
        dataset = str(payload.get("dataset") or "").strip()
        object_name = str(payload.get("source_model_name") or "").strip()
        prompt_id = int(payload.get("prompt_id") or 0)
        case_id = str(payload.get("case_id") or case_root.name)
        if not dataset or not object_name or prompt_id <= 0:
            continue
        cases.append(
            MvCase(
                case_id=case_id,
                dataset=dataset,
                object_name=object_name,
                prompt_id=prompt_id,
                case_root=case_root,
                payload=payload,
            )
        )
    return cases


def _build_angle_grid(camera_payload: dict[str, Any], key: str) -> tuple[list[float], list[float]]:
    views = list(camera_payload.get(key) or [])
    elevations: list[float] = []
    azimuths: list[float] = []
    for view in views:
        elevation = float(view["elevation"])
        azimuth = float(view["azimuth"])
        if elevation not in elevations:
            elevations.append(elevation)
        if azimuth not in azimuths:
            azimuths.append(azimuth)
    return elevations, azimuths


def _resolve_pred_model(pred_root: Path, case: MvCase) -> Path | None:
    candidates = [
        pred_root / case.dataset / case.object_name / case.prompt_key / "edit.glb",
        pred_root / "cases" / case.case_id / "edit.glb",
        pred_root / case.case_id / "edit.glb",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _render_sequence_complete(output_dir: Path, expected_count: int) -> bool:
    return len(list(output_dir.glob("render_*.png"))) == expected_count


def _render_angle_grid(
    scene_manager,
    *,
    output_dir: Path,
    elevations: list[float],
    azimuths: list[float],
    image_size: int,
    quiet_blender: bool,
) -> None:
    ensure_dir(output_dir)
    bench_render.reset_render_pass(scene_manager)
    bench_render.clear_render_sequence(str(output_dir))
    _, cam_mats, _, _ = bench_render.get_camera_positions_on_sphere(
        center=(0, 0, 0),
        radius=1.8,
        elevations=elevations,
        azimuths=azimuths,
    )
    bench_render.add_camera_path(cam_mats)
    bench_render.enable_color_output(
        image_size,
        image_size,
        str(output_dir),
        file_format="PNG",
        mode="IMAGE",
        film_transparent=True,
    )
    with bench_render.suppress_blender_output(enabled=quiet_blender):
        scene_manager.render()


def _render_case_views(
    *,
    model_path: Path,
    seen_dir: Path,
    novel_dir: Path,
    seen_grid: tuple[list[float], list[float]],
    novel_grid: tuple[list[float], list[float]],
    image_size: int,
    quiet_blender: bool,
    cycles_backend: str,
    overwrite: bool,
) -> None:
    need_seen = overwrite or not _render_sequence_complete(seen_dir, expected_count=4)
    need_novel = overwrite or not _render_sequence_complete(novel_dir, expected_count=16)
    if not need_seen and not need_novel:
        return

    scene_manager = bench_render.prepare_scene(
        str(model_path),
        engine="CYCLES",
        samples=DEFAULT_RENDER_SAMPLES,
        cycles_backend=cycles_backend,
    )
    try:
        if need_seen:
            _render_angle_grid(
                scene_manager,
                output_dir=seen_dir,
                elevations=seen_grid[0],
                azimuths=seen_grid[1],
                image_size=image_size,
                quiet_blender=quiet_blender,
            )
        if need_novel:
            _render_angle_grid(
                scene_manager,
                output_dir=novel_dir,
                elevations=novel_grid[0],
                azimuths=novel_grid[1],
                image_size=image_size,
                quiet_blender=quiet_blender,
            )
    finally:
        scene_manager.clear(reset_keyframes=True)
        scene_manager.gc()


def _build_seen_specs(case: MvCase) -> list[ViewSpec]:
    specs = []
    input_views = case.payload.get("input_views") or {}
    for view in sorted(input_views.values(), key=lambda item: str(item["render_id"])):
        specs.append(
            ViewSpec(
                view_id=str(view["render_id"]),
                gt_path=case.case_root / str(view["target_path"]),
                mask_path=case.case_root / str(view["mask_path"]),
            )
        )
    return specs


def _build_novel_specs(case: MvCase, camera_pool: dict[str, Any]) -> list[ViewSpec]:
    specs = []
    for view in camera_pool.get("views") or []:
        view_id = str(view["view_id"])
        specs.append(
            ViewSpec(
                view_id=view_id,
                gt_path=case.case_root / "eval_gt" / "target_views" / f"render_{view_id}.png",
                mask_path=case.case_root / "eval_gt" / "masks" / f"mask_{view_id}.png",
            )
        )
    return specs


def _load_mask_array(mask_path: Path | None, image_size: int, *, inside: bool, all_pixels: bool = False) -> np.ndarray:
    if all_pixels or mask_path is None:
        return np.ones((image_size, image_size), dtype=np.float32)
    outside_mask = build_include_mask(
        str(mask_path),
        (image_size, image_size),
        dilation_px=0,
        ignore_mask=False,
    )
    if inside:
        return 1.0 - outside_mask.astype(np.float32)
    return outside_mask.astype(np.float32)


def _compute_lpips_ssim(
    evaluator: ImageMetricsEvaluator,
    *,
    pairs: list[tuple[Path, Path, Path | None]],
    image_size: int,
    inside_mask: bool,
    all_pixels: bool = False,
) -> tuple[float | None, float | None]:
    if not pairs:
        return None, None

    to_tensor = transforms.ToTensor()
    gt_tensors = []
    pred_tensors = []
    masks = []
    for gt_path, pred_path, mask_path in pairs:
        gt_tensors.append(to_tensor(load_prepared_rgb_image(str(gt_path), (image_size, image_size))))
        pred_tensors.append(to_tensor(load_prepared_rgb_image(str(pred_path), (image_size, image_size))))
        masks.append(
            torch.from_numpy(
                _load_mask_array(
                    mask_path,
                    image_size,
                    inside=inside_mask,
                    all_pixels=all_pixels,
                )
            )
        )

    batch_gt = torch.stack(gt_tensors, dim=0)
    batch_pred = torch.stack(pred_tensors, dim=0)
    batch_mask = torch.stack(masks, dim=0)
    ssim_scores = [
        score
        for score in evaluator._calculate_ssim_batch_torch(batch_gt, batch_pred, batch_mask)
        if score is not None
    ]
    lpips_scores = [
        score
        for score in evaluator._calculate_lpips_batch_masked(
            batch_gt,
            batch_pred,
            batch_mask.unsqueeze(1),
        )
        if score is not None
    ]
    ssim_mean = float(np.mean(ssim_scores)) if ssim_scores else None
    lpips_mean = float(np.mean(lpips_scores)) if lpips_scores else None
    return lpips_mean, ssim_mean


def _compute_dino_similarity(
    evaluator: ImageMetricsEvaluator,
    *,
    pairs: list[tuple[Path, Path]],
) -> float | None:
    if not pairs:
        return None
    gt_images = [evaluator._load_prepared_pil(str(gt_path)) for gt_path, _ in pairs]
    pred_images = [evaluator._load_prepared_pil(str(pred_path)) for _, pred_path in pairs]
    embeddings = evaluator._encode_dino_embeddings(gt_images + pred_images)
    gt_embeddings = embeddings[: len(gt_images)]
    pred_embeddings = embeddings[len(gt_images) :]
    similarities = (gt_embeddings * pred_embeddings).sum(dim=1)
    return float(similarities.mean().item())


def _compute_target_chamfer(
    evaluator: GeometryMetricsEvaluator,
    *,
    target_model: Path,
    pred_model: Path,
    mask_glb: Path,
    case_id: str,
) -> float | None:
    scores, _, _, _ = evaluator._compute_chamfer_with_dataloader(
        [(str(target_model), str(pred_model), str(mask_glb), case_id)]
    )
    return float(scores[0]) if scores else None


def _compute_cross_seed_novel_lpips(
    evaluator: ImageMetricsEvaluator,
    *,
    novel_dir: Path,
    peer_novel_dirs: list[Path],
    image_size: int,
) -> float | None:
    if not peer_novel_dirs:
        return None

    scores = []
    for peer_dir in peer_novel_dirs:
        current_pairs = []
        for pred_path in sorted(novel_dir.glob("render_*.png")):
            peer_path = peer_dir / pred_path.name
            if peer_path.is_file():
                current_pairs.append((pred_path, peer_path, None))
        if not current_pairs:
            continue
        lpips_mean, _ = _compute_lpips_ssim(
            evaluator,
            pairs=current_pairs,
            image_size=image_size,
            inside_mask=False,
            all_pixels=True,
        )
        if lpips_mean is not None:
            scores.append(lpips_mean)
    return float(np.mean(scores)) if scores else None


def _build_case_metric_summary(case_metrics: dict[str, dict[str, float | None]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric_name in MV_METRIC_NAMES:
        values = [
            float(metric_value)
            for metric_value in (
                case_payload.get(metric_name)
                for case_payload in case_metrics.values()
            )
            if metric_value is not None
        ]
        if not values:
            continue
        values_array = np.array(values, dtype=np.float32)
        summary[metric_name] = {
            "mean": float(np.mean(values_array)),
            "std": float(np.std(values_array)),
            "count": int(values_array.shape[0]),
        }
    return summary


def _case_key(case: MvCase) -> tuple[str, str, int]:
    return (case.dataset, case.object_name, case.prompt_id)


def _case_record_key(record: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(record.get("dataset") or ""),
        str(record.get("object_name") or ""),
        int(record.get("prompt_id") or 0),
    )


def _resolve_peer_pred_roots_from_runs(peer_run_roots: list[Path]) -> list[Path]:
    peer_pred_roots: list[Path] = []
    for peer_run_root in peer_run_roots:
        manifest_path = peer_run_root / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest_payload = _read_json(manifest_path)
        pred_root = manifest_payload.get("pred_root")
        if not pred_root:
            continue
        peer_pred_roots.append(Path(str(pred_root)).expanduser().resolve())
    return peer_pred_roots


def _build_mv_metrics_payload(
    *,
    dataset_root: Path,
    image_size: int,
    device: str,
    metric_summary: dict[str, Any],
    raw_case_metrics: dict[str, dict[str, float | None]],
) -> dict[str, Any]:
    return {
        "dataset_root": str(dataset_root),
        "image_size": image_size,
        "device": device,
        "metrics": metric_summary,
        "cases": raw_case_metrics,
    }


def _select_case_prompt_text(case: MvCase) -> str:
    edit_instruction = str(case.payload.get("edit_instruction") or "").strip()
    prompt_text = str(case.payload.get("prompt_text") or "").strip()
    source_prompt = str(case.payload.get("source_prompt") or "").strip()
    if edit_instruction:
        if prompt_text and prompt_text.lower() not in edit_instruction.lower():
            return f"{prompt_text} | {edit_instruction}"
        return edit_instruction
    return prompt_text or source_prompt


def _materialize_mv_case_artifacts(
    *,
    bundle_root: Path,
    case: MvCase,
    pred_model: Path | None,
) -> dict[str, str]:
    artifact_dir = ensure_dir(bundle_root / "artifacts" / case.dataset / case.object_name / case.prompt_key)
    relpaths: dict[str, str] = {}

    anchor_payload = case.payload.get("anchor_edit") or {}
    candidates: list[tuple[str, Path, Path]] = []

    for key, filename in (
        ("source_path", "source_image.png"),
        ("target_path", "edit_image.png"),
        ("mask_path", "mask_image.png"),
    ):
        relative_path = anchor_payload.get(key)
        if relative_path:
            label = {
                "source_path": "source_image",
                "target_path": "edit_image",
                "mask_path": "mask_image",
            }[key]
            candidates.append((label, case.case_root / str(relative_path), artifact_dir / filename))

    candidates.extend(
        [
            ("source_model_glb", case.case_root / "source_model.glb", artifact_dir / "source_model.glb"),
            ("target_model_glb", case.case_root / "target_model.glb", artifact_dir / "target_model.glb"),
            ("mask_glb", case.case_root / "edit_region.glb", artifact_dir / "mask.glb"),
            ("input_views", case.case_root / "input_views", artifact_dir / "input_views"),
            ("eval_target_views", case.case_root / "eval_gt" / "target_views", artifact_dir / "eval_target_views"),
            ("eval_masks", case.case_root / "eval_gt" / "masks", artifact_dir / "eval_masks"),
        ]
    )

    if pred_model is not None:
        pred_case_dir = pred_model.parent
        mv_eval_dir = pred_case_dir / "mv_eval"
        candidates.extend(
            [
                ("edit_glb", pred_model, artifact_dir / "edit.glb"),
                ("images", mv_eval_dir, artifact_dir / "images"),
                ("seen_views", mv_eval_dir / "seen_views", artifact_dir / "seen_views"),
                ("novel_views", mv_eval_dir / "novel_views", artifact_dir / "novel_views"),
            ]
        )

    for name, src, dst in candidates:
        if _link_or_copy(src, dst):
            relpaths[name] = _bundle_relpath(bundle_root, dst)
    return relpaths


def _create_daily_bundle_from_pred_root(
    *,
    daily_root: Path,
    pred_root: Path,
    dataset_root: Path,
    cases: list[MvCase],
    case_metrics_by_key: dict[tuple[str, str, int], dict[str, float | None]],
    metric_summary: dict[str, Any],
    requested_metrics: list[str],
    total_time_seconds: float,
    run_group: str | None,
) -> Path:
    raw_manifest = _read_json(pred_root / "manifest.json") if (pred_root / "manifest.json").is_file() else {}
    config_name = str(raw_manifest.get("run_name") or raw_manifest.get("config_name") or pred_root.name)
    entrypoint_name = str(raw_manifest.get("method") or raw_manifest.get("entrypoint") or "mv_edit")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = build_run_id(entrypoint_name, config_name, timestamp)
    bundle_root = (daily_root / run_id).resolve()
    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    ensure_dir(bundle_root)

    case_records: list[dict[str, Any]] = []
    cases_dir = ensure_dir(bundle_root / "cases")
    pages_dir = ensure_dir(bundle_root / "pages")
    for case in cases:
        case_key = _case_key(case)
        metrics_payload = {
            metric_name: case_metrics_by_key.get(case_key, {}).get(metric_name)
            for metric_name in requested_metrics
        }
        pred_model = _resolve_pred_model(pred_root, case)
        artifact_relpaths = _materialize_mv_case_artifacts(
            bundle_root=bundle_root,
            case=case,
            pred_model=pred_model,
        )

        has_edit = "edit_glb" in artifact_relpaths
        has_render = "images" in artifact_relpaths or "novel_views" in artifact_relpaths or "seen_views" in artifact_relpaths
        status = "failed" if not has_edit else ("ok" if has_render else "partial")
        case_id = _case_id(case.dataset, case.object_name, case.prompt_id)
        case_payload = {
            "case_id": case_id,
            "dataset": case.dataset,
            "object_name": case.object_name,
            "prompt_id": case.prompt_id,
            "status": status,
            "prompt_text": _select_case_prompt_text(case),
            "source_prompt": case.payload.get("source_prompt"),
            "edit_instruction": case.payload.get("edit_instruction"),
            "raw_case_id": case.case_id,
            "metrics": metrics_payload,
            "artifacts": artifact_relpaths,
        }
        case_json_path = cases_dir / case.dataset / case.object_name / f"prompt_{case.prompt_id}.json"
        write_json(case_json_path, case_payload)

        page_path = pages_dir / case.dataset / case.object_name / f"prompt_{case.prompt_id}.html"
        ensure_dir(page_path.parent)
        case_records.append(
            {
                "case_id": case_id,
                "dataset": case.dataset,
                "object_name": case.object_name,
                "prompt_id": case.prompt_id,
                "status": status,
                "metrics": metrics_payload,
                "detail_path": _bundle_relpath(bundle_root, case_json_path),
                "page_path": _bundle_relpath(bundle_root, page_path),
                "artifact_dir": f"artifacts/{case.dataset}/{case.object_name}/{case.prompt_key}",
            }
        )

    summary_payload = {
        "totals": _summarize_statuses(case_records),
        "metrics": metric_summary,
    }
    manifest_payload = {
        "run_id": run_id,
        "entrypoint": entrypoint_name,
        "config_name": config_name,
        "group": run_group if run_group is not None else raw_manifest.get("group"),
        "created_at": datetime.now().astimezone().isoformat(),
        "metrics": list(requested_metrics),
        "pred_root": str(pred_root),
        "dataset_root": str(dataset_root),
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


def _update_daily_bundle(
    *,
    run_root: Path,
    case_metrics: dict[tuple[str, str, int], dict[str, float | None]],
    metric_summary: dict[str, Any],
) -> None:
    summary_path = run_root / "summary.json"
    manifest_path = run_root / "manifest.json"
    cases_jsonl_path = run_root / "cases.jsonl"

    summary_payload = _read_json(summary_path)
    summary_payload.setdefault("metrics", {}).update(metric_summary)
    write_json(summary_path, summary_payload)

    manifest_payload = _read_json(manifest_path)
    existing_metrics = list(manifest_payload.get("metrics") or [])
    for metric_name in MV_METRIC_NAMES:
        if metric_name in metric_summary and metric_name not in existing_metrics:
            existing_metrics.append(metric_name)
    manifest_payload["metrics"] = existing_metrics
    write_json(manifest_path, manifest_payload)

    case_records = []
    for line in cases_jsonl_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        metrics = dict(record.get("metrics") or {})
        metrics.update(case_metrics.get(_case_record_key(record), {}))
        record["metrics"] = metrics
        case_records.append(record)
    cases_jsonl_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in case_records) + "\n",
        encoding="utf-8",
    )

    for record in case_records:
        detail_path = run_root / str(record.get("detail_path") or "")
        if not detail_path.is_file():
            continue
        detail_payload = _read_json(detail_path)
        metrics = dict(detail_payload.get("metrics") or {})
        metrics.update(case_metrics.get(_case_record_key(record), {}))
        detail_payload["metrics"] = metrics
        write_json(detail_path, detail_payload)


def _rebuild_indexes_for_daily_root(daily_root: Path) -> None:
    if daily_root.name != "daily":
        return
    build_daily_index(daily_root.parent)
    build_focus_index(daily_root.parent)


def _evaluate_pred_root(
    *,
    pred_root: Path,
    dataset_root: Path,
    image_size: int,
    device: str,
    overwrite_render: bool,
    quiet_blender: bool,
    cycles_backend: str,
    peer_pred_roots: list[Path],
) -> dict[str, Any]:
    _ensure_mv_modules_loaded()
    pred_root = pred_root.expanduser().resolve()
    print(f"[PRED] {pred_root.name}")
    print(f"       pred_root={pred_root}")

    camera_pool = _read_json(dataset_root / "camera_pool.json")
    input_cameras = _read_json(dataset_root / "input_view_cameras.json")
    seen_grid = _build_angle_grid(input_cameras, "views")
    novel_grid = _build_angle_grid(camera_pool, "views")
    cases = _load_mv_cases(dataset_root)

    image_evaluator = ImageMetricsEvaluator(
        device=device,
        image_size=(image_size, image_size),
        batch_size=32,
        num_workers=0,
        ignore_mask=False,
    )
    geometry_evaluator = GeometryMetricsEvaluator(
        device=device,
        batch_size=1,
        num_workers=0,
        ignore_mask=True,
    )

    case_metrics_by_key: dict[tuple[str, str, int], dict[str, float | None]] = {}
    raw_case_metrics: dict[str, dict[str, float | None]] = {}
    evaluated_cases: dict[tuple[str, str, int], EvaluatedMvCase] = {}

    for case in cases:
        pred_model = _resolve_pred_model(pred_root, case)
        if pred_model is None:
            evaluated_cases[_case_key(case)] = EvaluatedMvCase(
                case=case,
                pred_model=None,
                metrics={metric_name: None for metric_name in MV_METRIC_NAMES},
            )
            continue

        pred_case_dir = pred_model.parent
        mv_eval_dir = ensure_dir(pred_case_dir / "mv_eval")
        seen_dir = mv_eval_dir / "seen_views"
        novel_dir = mv_eval_dir / "novel_views"
        _render_case_views(
            model_path=pred_model,
            seen_dir=seen_dir,
            novel_dir=novel_dir,
            seen_grid=seen_grid,
            novel_grid=novel_grid,
            image_size=image_size,
            quiet_blender=quiet_blender,
            cycles_backend=cycles_backend,
            overwrite=overwrite_render,
        )

        seen_pairs = []
        seen_dino_pairs = []
        for spec in _build_seen_specs(case):
            pred_path = seen_dir / f"render_{spec.view_id}.png"
            if not pred_path.is_file():
                continue
            seen_pairs.append((spec.gt_path, pred_path, spec.mask_path))
            seen_dino_pairs.append((spec.gt_path, pred_path))

        novel_pairs = []
        novel_dino_pairs = []
        for spec in _build_novel_specs(case, camera_pool):
            pred_path = novel_dir / f"render_{spec.view_id}.png"
            if not pred_path.is_file():
                continue
            novel_pairs.append((spec.gt_path, pred_path, spec.mask_path))
            novel_dino_pairs.append((spec.gt_path, pred_path))

        lpips_in_seen, ssim_in_seen = _compute_lpips_ssim(
            image_evaluator,
            pairs=seen_pairs,
            image_size=image_size,
            inside_mask=True,
        )
        lpips_in_novel, ssim_in_novel = _compute_lpips_ssim(
            image_evaluator,
            pairs=novel_pairs,
            image_size=image_size,
            inside_mask=True,
        )
        dino_if_seen = _compute_dino_similarity(
            image_evaluator,
            pairs=seen_dino_pairs,
        )
        dino_if_novel = _compute_dino_similarity(
            image_evaluator,
            pairs=novel_dino_pairs,
        )
        chamfer_target = _compute_target_chamfer(
            geometry_evaluator,
            target_model=case.case_root / "target_model.glb",
            pred_model=pred_model,
            mask_glb=case.case_root / "edit_region.glb",
            case_id=case.case_id,
        )

        peer_novel_dirs = []
        for peer_pred_root in peer_pred_roots:
            peer_model = _resolve_pred_model(peer_pred_root, case)
            if peer_model is None:
                continue
            peer_mv_eval_dir = ensure_dir(peer_model.parent / "mv_eval")
            peer_seen_dir = peer_mv_eval_dir / "seen_views"
            peer_novel_dir = peer_mv_eval_dir / "novel_views"
            _render_case_views(
                model_path=peer_model,
                seen_dir=peer_seen_dir,
                novel_dir=peer_novel_dir,
                seen_grid=seen_grid,
                novel_grid=novel_grid,
                image_size=image_size,
                quiet_blender=quiet_blender,
                cycles_backend=cycles_backend,
                overwrite=overwrite_render,
            )
            if peer_novel_dir.is_dir():
                peer_novel_dirs.append(peer_novel_dir)

        cross_seed_novel_lpips = _compute_cross_seed_novel_lpips(
            image_evaluator,
            novel_dir=novel_dir,
            peer_novel_dirs=peer_novel_dirs,
            image_size=image_size,
        )

        metric_payload = {
            "lpips_in_seen": lpips_in_seen,
            "ssim_in_seen": ssim_in_seen,
            "lpips_in_novel": lpips_in_novel,
            "ssim_in_novel": ssim_in_novel,
            "dino_if_seen": dino_if_seen,
            "dino_if_novel": dino_if_novel,
            "chamfer_target": chamfer_target,
            "cross_seed_novel_lpips": cross_seed_novel_lpips,
        }
        case_metrics_by_key[_case_key(case)] = metric_payload
        raw_case_metrics[case.case_id] = metric_payload
        evaluated_cases[_case_key(case)] = EvaluatedMvCase(
            case=case,
            pred_model=pred_model,
            metrics=metric_payload,
        )
        print(
            f"      case={case.case_id} "
            f"lpips_in_novel={lpips_in_novel} "
            f"dino_if_novel={dino_if_novel} "
            f"chamfer_target={chamfer_target}"
        )

    metric_summary = _build_case_metric_summary(raw_case_metrics)
    return {
        "pred_root": str(pred_root),
        "metric_summary": metric_summary,
        "case_metrics_by_key": case_metrics_by_key,
        "raw_case_metrics": raw_case_metrics,
        "evaluated_cases": evaluated_cases,
        "cases": cases,
        "case_count": len(raw_case_metrics),
        "report_payload": _build_mv_metrics_payload(
            dataset_root=dataset_root,
            image_size=image_size,
            device=device,
            metric_summary=metric_summary,
            raw_case_metrics=raw_case_metrics,
        ),
    }


def _evaluate_run(
    *,
    run_root: Path,
    dataset_root: Path,
    image_size: int,
    device: str,
    overwrite_render: bool,
    quiet_blender: bool,
    cycles_backend: str,
    peer_run_roots: list[Path],
) -> dict[str, Any]:
    manifest_payload = _read_json(run_root / "manifest.json")
    pred_root = Path(str(manifest_payload["pred_root"])).expanduser().resolve()
    print(f"[RUN] {run_root.name}")
    print(f"      pred_root={pred_root}")
    evaluation_payload = _evaluate_pred_root(
        pred_root=pred_root,
        dataset_root=dataset_root,
        image_size=image_size,
        device=device,
        overwrite_render=overwrite_render,
        quiet_blender=quiet_blender,
        cycles_backend=cycles_backend,
        peer_pred_roots=_resolve_peer_pred_roots_from_runs(peer_run_roots),
    )
    write_json(run_root / "mv_metrics.json", evaluation_payload["report_payload"])
    _update_daily_bundle(
        run_root=run_root,
        case_metrics=evaluation_payload["case_metrics_by_key"],
        metric_summary=evaluation_payload["metric_summary"],
    )
    return {
        "run_root": str(run_root),
        "pred_root": str(pred_root),
        "summary": evaluation_payload["metric_summary"],
        "case_count": evaluation_payload["case_count"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate multi-view editing metrics and either augment or create benchmark daily bundles."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_MV_DATASET_ROOT,
        help="Multi-view benchmark dataset root.",
    )
    parser.add_argument(
        "--daily-root",
        type=Path,
        default=DEFAULT_DAILY_ROOT,
        help="Benchmark daily root.",
    )
    parser.add_argument(
        "--pred-root",
        type=Path,
        action="append",
        help="Raw multi-view prediction root(s), e.g. pred_mv/<run_name>.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        action="append",
        help="Specific daily run root(s). Defaults to all runs under --daily-root.",
    )
    parser.add_argument(
        "--peer-run-root",
        type=Path,
        action="append",
        help="Optional peer run root(s) for cross-seed novel-view LPIPS.",
    )
    parser.add_argument(
        "--peer-pred-root",
        type=Path,
        action="append",
        help="Optional peer raw prediction root(s) for cross-seed novel-view LPIPS.",
    )
    parser.add_argument(
        "--run-group",
        type=str,
        default=None,
        help="Optional run group override when creating a new daily bundle from --pred-root.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--cycles-backend", type=str, default="AUTO")
    parser.add_argument("--overwrite-render", action="store_true")
    parser.add_argument("--quiet-blender", action="store_true", default=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    daily_root = args.daily_root.expanduser().resolve()
    if not dataset_root.exists():
        raise RuntimeError(f"Dataset root does not exist: {dataset_root}")
    ensure_dir(daily_root)

    pred_roots = [path.expanduser().resolve() for path in (args.pred_root or [])]
    peer_pred_roots = [path.expanduser().resolve() for path in (args.peer_pred_root or [])]
    peer_run_roots = [path.expanduser().resolve() for path in (args.peer_run_root or [])]
    run_roots: list[Path] = []
    if args.run_root:
        run_roots = [path.expanduser().resolve() for path in args.run_root]
    elif not pred_roots:
        run_roots = _iter_daily_run_roots(daily_root)

    results = []
    for pred_root in pred_roots:
        if not pred_root.exists():
            raise RuntimeError(f"Prediction root does not exist: {pred_root}")
        evaluation_payload = _evaluate_pred_root(
            pred_root=pred_root,
            dataset_root=dataset_root,
            image_size=int(args.image_size),
            device=args.device,
            overwrite_render=bool(args.overwrite_render),
            quiet_blender=bool(args.quiet_blender),
            cycles_backend=args.cycles_backend,
            peer_pred_roots=peer_pred_roots if len(pred_roots) == 1 else [],
        )
        write_json(pred_root / "mv_metrics.json", evaluation_payload["report_payload"])
        bundle_root = _create_daily_bundle_from_pred_root(
            daily_root=daily_root,
            pred_root=pred_root,
            dataset_root=dataset_root,
            cases=evaluation_payload["cases"],
            case_metrics_by_key=evaluation_payload["case_metrics_by_key"],
            metric_summary=evaluation_payload["metric_summary"],
            requested_metrics=list(MV_METRIC_NAMES),
            total_time_seconds=0.0,
            run_group=args.run_group,
        )
        write_json(bundle_root / "mv_metrics.json", evaluation_payload["report_payload"])
        results.append(
            {
                "mode": "pred_root",
                "run_root": str(bundle_root),
                "pred_root": str(pred_root),
                "summary": evaluation_payload["metric_summary"],
                "case_count": evaluation_payload["case_count"],
            }
        )

    for run_root in run_roots:
        if not run_root.exists():
            raise RuntimeError(f"Daily run root does not exist: {run_root}")
        results.append(
            _evaluate_run(
                run_root=run_root,
                dataset_root=dataset_root,
                image_size=int(args.image_size),
                device=args.device,
                overwrite_render=bool(args.overwrite_render),
                quiet_blender=bool(args.quiet_blender),
                cycles_backend=args.cycles_backend,
                peer_run_roots=peer_run_roots if len(run_roots) == 1 else [],
            )
        )

    _rebuild_indexes_for_daily_root(daily_root)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
