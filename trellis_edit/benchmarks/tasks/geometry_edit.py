from __future__ import annotations

import json
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from trellis_edit.benchmarks.single_view_common import (
    build_single_view_case_id,
    build_single_view_prompt_text_map,
    load_single_view_metadata,
)
from trellis_edit.common import ensure_dir
from trellis_edit.metrics.geometry_edit import (
    GEOMETRY_EDIT_METRICS,
    ClipImageSimilarityEvaluator,
    FloaterMeshEvaluator,
    Uni3DPointCloudEvaluator,
    build_mask_sdf,
    load_trimesh,
    sample_outside_mask_point_cloud,
    source_normalization,
    symmetric_chamfer_distance,
)

from ..base import BenchmarkContext, BenchmarkTask, CaseIdentity, CaseResult
from ..vendors import render_module


DEFAULT_GEOMETRY_DATASET_ROOT = Path(
    "/cache/wangxinxing/data/trellis_edit_benchmark/edit3d_data/data"
)
DEFAULT_GEOMETRY_IMAGE_SIZE = 512
DEFAULT_GEOMETRY_RENDER_SAMPLES = 32
GEOMETRY_METRIC_NAMES = GEOMETRY_EDIT_METRICS

_VIEW_ID_COUNT = len(render_module.IMAGE_ELEVATIONS) * len(render_module.IMAGE_AZIMUTHS)
_ALL_VIEW_IDS = tuple(f"{view_index:04d}" for view_index in range(_VIEW_ID_COUNT))
_LOCAL_MV_TOPK = 3


@dataclass(frozen=True)
class GeometryEditCase:
    identity: CaseIdentity
    prompt_dir: Path
    pred_dir: Path
    prompt_text: str | None


@dataclass(frozen=True)
class PreparedGeometryEditCase:
    case: GeometryEditCase
    pred_model: Path | None
    source_model: Path
    source_render_dir: Path
    edit_image: Path
    source_image: Path
    mask_image: Path
    mask_model: Path
    anchor_view_id: str | None


def _normalize_case_tuples(raw_cases: Sequence[Any] | None) -> list[tuple[str, str, int]]:
    if raw_cases is None:
        return []

    resolved: list[tuple[str, str, int]] = []
    seen: set[tuple[str, str, int]] = set()
    for item in raw_cases:
        if isinstance(item, (tuple, list)) and len(item) == 3:
            dataset = str(item[0]).strip()
            object_name = str(item[1]).strip()
            prompt_id = int(item[2])
        elif isinstance(item, dict):
            dataset = str(item.get("dataset") or "").strip()
            object_name = str(item.get("object_name") or item.get("source_model") or "").strip()
            prompt_id = int(item.get("prompt_id"))
        else:
            raise RuntimeError(
                "Geometry-edit task cases must be tuples like "
                "(dataset, object_name, prompt_id) or mappings with the same fields."
            )
        case_tuple = (dataset, object_name, prompt_id)
        if case_tuple in seen:
            continue
        seen.add(case_tuple)
        resolved.append(case_tuple)
    return resolved


def _discover_case_tuples(
    *,
    gt_root: Path,
    pred_root: Path,
    metadata: list[dict[str, Any]],
    explicit_cases: Sequence[Any] | None,
) -> list[tuple[str, str, int]]:
    selected_cases = _normalize_case_tuples(explicit_cases)
    if selected_cases:
        return selected_cases

    cases: list[tuple[str, str, int]] = []
    for row in metadata:
        dataset = str(row.get("dataset") or "").strip()
        object_name = str(row.get("source_model") or "").strip()
        if not dataset or not object_name:
            continue
        for prompt_id in (1, 2, 3):
            prompt_key = f"prompt_{prompt_id}"
            prompt_dir = gt_root / dataset / object_name / prompt_key
            pred_dir = pred_root / dataset / object_name / prompt_key
            if not prompt_dir.is_dir():
                continue
            if not (pred_dir / "edit.glb").is_file():
                continue
            cases.append((dataset, object_name, prompt_id))
    return cases


def _build_cases(
    *,
    gt_root: Path,
    pred_root: Path,
    metadata: list[dict[str, Any]],
    case_tuples: list[tuple[str, str, int]],
) -> list[GeometryEditCase]:
    prompt_text_map = build_single_view_prompt_text_map(metadata)
    cases: list[GeometryEditCase] = []
    for dataset, object_name, prompt_id in case_tuples:
        identity = CaseIdentity(
            case_id=build_single_view_case_id(dataset, object_name, prompt_id),
            dataset=dataset,
            object_name=object_name,
            prompt_id=prompt_id,
        )
        prompt_key = f"prompt_{prompt_id}"
        cases.append(
            GeometryEditCase(
                identity=identity,
                prompt_dir=gt_root / dataset / object_name / prompt_key,
                pred_dir=pred_root / dataset / object_name / prompt_key,
                prompt_text=prompt_text_map.get((dataset, object_name, prompt_id)),
            )
        )
    return cases


def _resolve_anchor_view_id(source_image: Path, source_render_dir: Path) -> str | None:
    candidate_paths = sorted(source_render_dir.glob("render_*.png"))
    if not candidate_paths:
        return None

    source_rgba = Image.open(source_image).convert("RGBA")
    source_array = np.asarray(source_rgba, dtype=np.int16)

    best_view_id: str | None = None
    best_score: int | None = None
    for candidate_path in candidate_paths:
        candidate_rgba = Image.open(candidate_path).convert("RGBA")
        candidate_array = np.asarray(candidate_rgba, dtype=np.int16)
        score = int(np.abs(source_array - candidate_array).sum())
        if best_score is None or score < best_score:
            best_score = score
            best_view_id = candidate_path.stem.split("_", 1)[-1]
            if score == 0:
                break
    return best_view_id


def _view_file_name(prefix: str, view_id: str) -> str:
    return f"{prefix}_{view_id}.png"


def _sequence_complete(output_dir: Path, prefix: str) -> bool:
    return len(list(output_dir.glob(f"{prefix}_*.png"))) == _VIEW_ID_COUNT


def _clear_prefixed_sequence(output_dir: Path, prefix: str) -> None:
    for file_path in glob(str(output_dir / f"{prefix}_*.png")):
        Path(file_path).unlink()


def _apply_gray_material_override() -> None:
    import bpy

    material = bpy.data.materials.get("GeometryEvalGray")
    if material is None:
        material = bpy.data.materials.new(name="GeometryEvalGray")
        material.use_nodes = True
        node_tree = material.node_tree
        node_tree.nodes.clear()
        diffuse = node_tree.nodes.new("ShaderNodeBsdfDiffuse")
        diffuse.inputs[0].default_value = (0.5, 0.5, 0.5, 1.0)
        output = node_tree.nodes.new("ShaderNodeOutputMaterial")
        node_tree.links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])
    bpy.context.view_layer.material_override = material


def _clear_material_override() -> None:
    import bpy

    bpy.context.view_layer.material_override = None


def _render_view_grid(
    *,
    model_path: Path,
    gray_dir: Path | None,
    normal_dir: Path | None,
    image_size: int,
    render_samples: int,
    quiet_blender: bool,
    cycles_backend: str,
    overwrite: bool,
) -> None:
    need_gray = gray_dir is not None and (overwrite or not _sequence_complete(gray_dir, "render"))
    need_normal = normal_dir is not None and (overwrite or not _sequence_complete(normal_dir, "normal"))
    if not need_gray and not need_normal:
        return

    from bpyrenderer.render_output import enable_normals_output

    if gray_dir is not None:
        gray_dir.mkdir(parents=True, exist_ok=True)
    if normal_dir is not None:
        normal_dir.mkdir(parents=True, exist_ok=True)

    scene_manager = render_module.prepare_scene(
        str(model_path),
        engine="CYCLES",
        samples=render_samples,
        cycles_backend=cycles_backend,
    )
    try:
        _apply_gray_material_override()
        render_module.reset_render_pass(scene_manager)
        if need_gray and gray_dir is not None:
            _clear_prefixed_sequence(gray_dir, "render")
        if need_normal and normal_dir is not None:
            _clear_prefixed_sequence(normal_dir, "normal")

        _, camera_mats, _, _ = render_module.get_camera_positions_on_sphere(
            center=(0, 0, 0),
            radius=1.8,
            elevations=render_module.IMAGE_ELEVATIONS,
            azimuths=render_module.IMAGE_AZIMUTHS,
        )
        render_module.add_camera_path(camera_mats)

        if need_gray and gray_dir is not None:
            render_module.enable_color_output(
                image_size,
                image_size,
                str(gray_dir),
                file_format="PNG",
                mode="IMAGE",
                film_transparent=True,
            )
        if need_normal and normal_dir is not None:
            enable_normals_output(
                output_dir=str(normal_dir),
                file_prefix="normal_",
                file_format="PNG",
            )

        with render_module.suppress_blender_output(enabled=quiet_blender):
            scene_manager.render()
    finally:
        _clear_material_override()
        scene_manager.clear(reset_keyframes=True)
        scene_manager.gc()


def _neighbor_view_ids(anchor_view_id: str) -> list[str]:
    anchor_index = int(anchor_view_id)
    azimuth_count = len(render_module.IMAGE_AZIMUTHS)
    elevation_count = len(render_module.IMAGE_ELEVATIONS)
    row = anchor_index // azimuth_count
    col = anchor_index % azimuth_count

    candidates = {
        (row, col),
        (row, (col - 1) % azimuth_count),
        (row, (col + 1) % azimuth_count),
    }
    for other_row in range(elevation_count):
        if other_row == row:
            continue
        candidates.add((other_row, col))
        candidates.add((other_row, (col - 1) % azimuth_count))
        candidates.add((other_row, (col + 1) % azimuth_count))

    return [
        f"{candidate_row * azimuth_count + candidate_col:04d}"
        for candidate_row, candidate_col in sorted(candidates)
    ]


def _composite_rgba_on_gray(path: Path, *, gray_value: int = 127) -> Image.Image:
    rgba = Image.open(path).convert("RGBA")
    background = Image.new("RGBA", rgba.size, (gray_value, gray_value, gray_value, 255))
    background.alpha_composite(rgba)
    return background.convert("RGB")


def _masked_crop(image: Image.Image, mask_alpha: np.ndarray, *, gray_value: int = 127) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask = np.asarray(mask_alpha, dtype=np.uint8) > 0
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        filled = np.full_like(rgb, gray_value)
        return Image.fromarray(filled, mode="RGB")

    y0 = max(int(ys.min()), 0)
    y1 = min(int(ys.max()) + 1, rgb.shape[0])
    x0 = max(int(xs.min()), 0)
    x1 = min(int(xs.max()) + 1, rgb.shape[1])

    filled = np.full_like(rgb, gray_value)
    filled[mask] = rgb[mask]
    return Image.fromarray(filled[y0:y1, x0:x1], mode="RGB")


def _compute_masked_similarity(
    evaluator: ClipImageSimilarityEvaluator,
    *,
    edit_image: Image.Image,
    render_path: Path,
    mask_path: Path,
) -> float | None:
    if not render_path.is_file() or not mask_path.is_file():
        return None
    render_image = _composite_rgba_on_gray(render_path)
    mask_alpha = np.asarray(Image.open(mask_path).convert("RGBA"))[:, :, 3]
    aligned_edit_image = edit_image
    if aligned_edit_image.size != render_image.size:
        aligned_edit_image = aligned_edit_image.resize(render_image.size, Image.BILINEAR)
    edit_crop = _masked_crop(aligned_edit_image, mask_alpha)
    render_crop = _masked_crop(render_image, mask_alpha)
    return evaluator.similarity(edit_crop, render_crop)


def _topk_mean(scores: Sequence[float | None], *, k: int) -> float | None:
    valid_scores = [float(score) for score in scores if score is not None]
    if not valid_scores:
        return None
    k = max(1, min(int(k), len(valid_scores)))
    return float(np.mean(sorted(valid_scores, reverse=True)[:k]))


def _case_artifacts(case: GeometryEditCase) -> dict[str, Path]:
    geo_eval_dir = case.pred_dir / "geometry_eval"
    return {
        "source_image": case.prompt_dir / "2d_render.png",
        "edit_image": case.prompt_dir / "2d_edit.png",
        "mask_image": case.prompt_dir / "2d_mask.png",
        "source_model_glb": case.prompt_dir.parent / "source_model" / "model.glb",
        "mask_glb": case.prompt_dir / "3d_edit_region.glb",
        "edit_glb": case.pred_dir / "edit.glb",
        "geo_gray_views": geo_eval_dir / "gray_views",
        "geo_normal_views": geo_eval_dir / "normal_views",
        "geo_mask_views": geo_eval_dir / "mask_views",
        "geo_case_metrics": geo_eval_dir / "case_metrics.json",
    }


def _summarize_case_metrics(case_results: Sequence[CaseResult]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric_name in GEOMETRY_METRIC_NAMES:
        values = [
            float(metric_value)
            for metric_value in (item.metrics.get(metric_name) for item in case_results)
            if metric_value is not None
        ]
        if not values:
            continue
        value_array = np.asarray(values, dtype=np.float32)
        summary[metric_name] = {
            "mean": float(np.mean(value_array)),
            "std": float(np.std(value_array)),
            "count": int(value_array.shape[0]),
        }
    return summary


class GeometryEditTask(BenchmarkTask):
    name = "geometry_edit"

    def __init__(self, gt_root: Path):
        self.gt_root = gt_root.expanduser().resolve()
        self._metadata = load_single_view_metadata(self.gt_root)
        self._summary_metrics: dict[str, Any] = {}
        self._report_payload: dict[str, Any] = {}

    def discover_cases(self, context: BenchmarkContext) -> Sequence[GeometryEditCase]:
        pred_root = context.pred_root.expanduser().resolve()
        case_tuples = _discover_case_tuples(
            gt_root=self.gt_root,
            pred_root=pred_root,
            metadata=self._metadata,
            explicit_cases=context.task_config.get("cases"),
        )
        return _build_cases(
            gt_root=self.gt_root,
            pred_root=pred_root,
            metadata=self._metadata,
            case_tuples=case_tuples,
        )

    def prepare_cases(
        self,
        cases: Sequence[GeometryEditCase],
        context: BenchmarkContext,
    ) -> Sequence[PreparedGeometryEditCase]:
        prepared_cases: list[PreparedGeometryEditCase] = []
        for case in cases:
            source_model = case.prompt_dir.parent / "source_model" / "model.glb"
            source_render_dir = case.prompt_dir.parent / "source_model" / "render"
            source_image = case.prompt_dir / "2d_render.png"
            anchor_view_id = _resolve_anchor_view_id(source_image, source_render_dir)
            prepared_cases.append(
                PreparedGeometryEditCase(
                    case=case,
                    pred_model=case.pred_dir / "edit.glb" if (case.pred_dir / "edit.glb").is_file() else None,
                    source_model=source_model,
                    source_render_dir=source_render_dir,
                    edit_image=case.prompt_dir / "2d_edit.png",
                    source_image=source_image,
                    mask_image=case.prompt_dir / "2d_mask.png",
                    mask_model=case.prompt_dir / "3d_edit_region.glb",
                    anchor_view_id=anchor_view_id,
                )
            )
        return prepared_cases

    def evaluate_cases(
        self,
        prepared_cases: Sequence[PreparedGeometryEditCase],
        context: BenchmarkContext,
    ) -> Sequence[CaseResult]:
        task_config = dict(context.task_config)
        requested_metrics = tuple(
            metric_name
            for metric_name in context.requested_metrics
            if metric_name in GEOMETRY_METRIC_NAMES
        ) or GEOMETRY_METRIC_NAMES
        image_size = int(task_config.get("image_size", DEFAULT_GEOMETRY_IMAGE_SIZE))
        render_samples = int(task_config.get("render_samples", DEFAULT_GEOMETRY_RENDER_SAMPLES))
        overwrite_render = bool(task_config.get("overwrite_render", False))
        quiet_blender = bool(task_config.get("quiet_blender", True))
        cycles_backend = str(task_config.get("cycles_backend", "AUTO"))
        outside_points = int(task_config.get("outside_points", 10000))
        outside_initial_samples = int(task_config.get("outside_initial_samples", 40000))
        uni3d_model = str(task_config.get("uni3d_model", "uni3d-g"))
        floater_kwargs: dict[str, float] = {}
        if "floater_merge_precision" in task_config:
            floater_kwargs["merge_precision"] = float(task_config["floater_merge_precision"])
        if "floater_area_ratio_threshold" in task_config:
            floater_kwargs["floater_area_ratio_threshold"] = float(
                task_config["floater_area_ratio_threshold"]
            )
        if "micro_floater_area_ratio_threshold" in task_config:
            floater_kwargs["micro_floater_area_ratio_threshold"] = float(
                task_config["micro_floater_area_ratio_threshold"]
            )

        clip_evaluator = ClipImageSimilarityEvaluator(device=context.device)
        uni3d_evaluator = (
            Uni3DPointCloudEvaluator(device=context.device, model_name=uni3d_model)
            if any(metric_name == "uni3d" for metric_name in requested_metrics)
            else None
        )
        floater_evaluator = (
            FloaterMeshEvaluator(**floater_kwargs)
            if any(metric_name == "floaters" for metric_name in requested_metrics)
            else None
        )

        case_results: list[CaseResult] = []
        raw_case_metrics: dict[str, dict[str, Any]] = {}

        for case_index, prepared_case in enumerate(prepared_cases):
            case = prepared_case.case
            pred_model = prepared_case.pred_model
            metrics = {metric_name: None for metric_name in requested_metrics}
            case_meta: dict[str, Any] = {
                "anchor_view_id": prepared_case.anchor_view_id,
                "source_prompt": None,
                "edit_instruction": case.prompt_text,
            }

            if pred_model is None:
                case_results.append(
                    CaseResult(
                        identity=case.identity,
                        status="failed",
                        metrics=metrics,
                        artifacts=_case_artifacts(case),
                        prompt_text=case.prompt_text,
                        task_meta=case_meta,
                    )
                )
                raw_case_metrics[case.identity.case_id] = {
                    "metrics": metrics,
                    "task_meta": case_meta,
                    "status": "failed",
                }
                continue

            gray_scores_by_view: dict[str, float | None] = {}
            normal_scores_by_view: dict[str, float | None] = {}
            local_view_ids: list[str] = []
            geo_eval_dir = ensure_dir(case.pred_dir / "geometry_eval")
            status = "failed"
            try:
                gray_dir = ensure_dir(geo_eval_dir / "gray_views")
                normal_dir = ensure_dir(geo_eval_dir / "normal_views")
                mask_dir = ensure_dir(geo_eval_dir / "mask_views")

                _render_view_grid(
                    model_path=pred_model,
                    gray_dir=gray_dir,
                    normal_dir=normal_dir,
                    image_size=image_size,
                    render_samples=render_samples,
                    quiet_blender=quiet_blender,
                    cycles_backend=cycles_backend,
                    overwrite=overwrite_render,
                )
                _render_view_grid(
                    model_path=prepared_case.mask_model,
                    gray_dir=mask_dir,
                    normal_dir=None,
                    image_size=image_size,
                    render_samples=render_samples,
                    quiet_blender=quiet_blender,
                    cycles_backend=cycles_backend,
                    overwrite=overwrite_render,
                )

                edit_image = Image.open(prepared_case.edit_image).convert("RGB")
                anchor_view_id = prepared_case.anchor_view_id
                if anchor_view_id is None:
                    search_view_ids = list(_ALL_VIEW_IDS)
                else:
                    search_view_ids = [anchor_view_id]
                local_view_ids = (
                    _neighbor_view_ids(anchor_view_id)
                    if anchor_view_id is not None
                    else list(_ALL_VIEW_IDS)
                )
                case_meta["local_view_ids"] = list(local_view_ids)

                eval_view_ids = sorted(set(search_view_ids) | set(local_view_ids))
                for view_id in eval_view_ids:
                    mask_path = mask_dir / _view_file_name("render", view_id)
                    gray_scores_by_view[view_id] = _compute_masked_similarity(
                        clip_evaluator,
                        edit_image=edit_image,
                        render_path=gray_dir / _view_file_name("render", view_id),
                        mask_path=mask_path,
                    )
                    normal_scores_by_view[view_id] = _compute_masked_similarity(
                        clip_evaluator,
                        edit_image=edit_image,
                        render_path=normal_dir / _view_file_name("normal", view_id),
                        mask_path=mask_path,
                    )

                if "clipi" in requested_metrics:
                    if anchor_view_id is not None:
                        metrics["clipi"] = gray_scores_by_view.get(anchor_view_id)
                    else:
                        metrics["clipi"] = _topk_mean(gray_scores_by_view.values(), k=1)
                if "clipi_n" in requested_metrics:
                    if anchor_view_id is not None:
                        metrics["clipi_n"] = normal_scores_by_view.get(anchor_view_id)
                    else:
                        metrics["clipi_n"] = _topk_mean(normal_scores_by_view.values(), k=1)
                if "clipi_mv" in requested_metrics:
                    metrics["clipi_mv"] = _topk_mean(
                        [gray_scores_by_view.get(view_id) for view_id in local_view_ids],
                        k=_LOCAL_MV_TOPK,
                    )
                if "clipi_mv_n" in requested_metrics:
                    metrics["clipi_mv_n"] = _topk_mean(
                        [normal_scores_by_view.get(view_id) for view_id in local_view_ids],
                        k=_LOCAL_MV_TOPK,
                    )

                if "uni3d" in requested_metrics or "cd" in requested_metrics:
                    source_mesh = load_trimesh(prepared_case.source_model)
                    pred_mesh = load_trimesh(pred_model)
                    mask_mesh = load_trimesh(prepared_case.mask_model)
                    source_center, source_extent = source_normalization(source_mesh)
                    mask_sdf = build_mask_sdf(mask_mesh)
                    source_points, source_diag = sample_outside_mask_point_cloud(
                        source_mesh,
                        mask_sdf=mask_sdf,
                        center=source_center,
                        extent=source_extent,
                        num_points=outside_points,
                        initial_sample_count=outside_initial_samples,
                        seed=case_index * 17,
                    )
                    pred_points, pred_diag = sample_outside_mask_point_cloud(
                        pred_mesh,
                        mask_sdf=mask_sdf,
                        center=source_center,
                        extent=source_extent,
                        num_points=outside_points,
                        initial_sample_count=outside_initial_samples,
                        seed=case_index * 17 + 1,
                    )
                    case_meta["outside_sampling"] = {
                        "mask_mode": "sdf",
                        "source": source_diag,
                        "pred": pred_diag,
                    }

                    if source_points is not None and pred_points is not None:
                        if "uni3d" in requested_metrics and uni3d_evaluator is not None:
                            metrics["uni3d"] = uni3d_evaluator.similarity(source_points, pred_points)
                        if "cd" in requested_metrics:
                            metrics["cd"] = symmetric_chamfer_distance(
                                source_points[:, :3],
                                pred_points[:, :3],
                            )

                if "floaters" in requested_metrics:
                    if floater_evaluator is not None:
                        floater_stats = floater_evaluator.analyze(pred_model)
                        case_meta["floaters"] = floater_stats
                        metrics["floaters"] = float(
                            floater_stats["floater_count_lt_0p1pct"]
                        )

                status = "ok"
                if any(metric_value is None for metric_value in metrics.values()):
                    status = "partial"
            except Exception as exc:
                case_meta["error"] = f"{type(exc).__name__}: {exc}"

            case_metrics_payload = {
                "case_id": case.identity.case_id,
                "anchor_view_id": prepared_case.anchor_view_id,
                "local_view_ids": local_view_ids,
                "metrics": metrics,
                "gray_scores_by_view": gray_scores_by_view,
                "normal_scores_by_view": normal_scores_by_view,
                "task_meta": case_meta,
            }
            (geo_eval_dir / "case_metrics.json").write_text(
                json.dumps(case_metrics_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            case_results.append(
                CaseResult(
                    identity=case.identity,
                    status=status,
                    metrics=metrics,
                    artifacts=_case_artifacts(case),
                    prompt_text=case.prompt_text,
                    task_meta=case_meta,
                )
            )
            raw_case_metrics[case.identity.case_id] = {
                "metrics": metrics,
                "task_meta": case_meta,
                "status": status,
            }

        self._summary_metrics = _summarize_case_metrics(case_results)
        self._report_payload = {
            "task": self.name,
            "metrics": self._summary_metrics,
            "cases": raw_case_metrics,
        }
        return case_results

    def summarize_run(
        self,
        case_results: Sequence[CaseResult],
        context: BenchmarkContext,
    ) -> dict[str, Any]:
        return dict(self._summary_metrics)

    def build_manifest(
        self,
        case_results: Sequence[CaseResult],
        context: BenchmarkContext,
    ) -> dict[str, Any]:
        manifest = super().build_manifest(case_results, context)
        manifest.update(
            {
                "gt_root": str(self.gt_root),
                "case_count": len(case_results),
            }
        )
        return manifest

    def collect_run_artifacts(
        self,
        case_results: Sequence[CaseResult],
        context: BenchmarkContext,
    ) -> dict[str, Any]:
        return {
            "geometry_metrics": dict(self._report_payload),
        }
