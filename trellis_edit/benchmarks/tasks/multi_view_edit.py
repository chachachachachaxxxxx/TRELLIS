from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ..base import BenchmarkContext, BenchmarkTask, CaseIdentity, CaseResult
from ..vendors import (
    GeometryMetricsEvaluator,
    ImageMetricsEvaluator,
    build_include_mask,
    load_prepared_rgb_image,
    render_module,
)


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


@dataclass(frozen=True)
class ViewSpec:
    view_id: str
    gt_path: Path
    mask_path: Path | None


@dataclass(frozen=True)
class MultiViewCase:
    identity: CaseIdentity
    case_root: Path
    payload: dict[str, Any]


@dataclass(frozen=True)
class PreparedMultiViewCase:
    case: MultiViewCase
    pred_model: Path | None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_daily_run_roots(daily_root: Path) -> list[Path]:
    run_roots = []
    for child in daily_root.iterdir():
        if not child.is_dir() or child.name == "groups":
            continue
        if (child / "manifest.json").is_file():
            run_roots.append(child)
    return sorted(run_roots)


def _prompt_text(case_payload: dict[str, Any]) -> str | None:
    edit_instruction = str(case_payload.get("edit_instruction") or "").strip()
    prompt_text = str(case_payload.get("prompt_text") or "").strip()
    source_prompt = str(case_payload.get("source_prompt") or "").strip()
    if edit_instruction:
        if prompt_text and prompt_text.lower() not in edit_instruction.lower():
            return f"{prompt_text} | {edit_instruction}"
        return edit_instruction
    return prompt_text or source_prompt or None


def _build_case_identity(case_root: Path, payload: dict[str, Any]) -> CaseIdentity:
    dataset = str(payload.get("dataset") or "").strip() or None
    object_name = str(payload.get("source_model_name") or "").strip() or None
    prompt_id_raw = payload.get("prompt_id")
    prompt_id: int | None = None
    if prompt_id_raw not in (None, ""):
        try:
            prompt_id = int(prompt_id_raw)
        except (TypeError, ValueError):
            prompt_id = None

    case_id = str(payload.get("case_id") or case_root.name).strip() or case_root.name
    sample_name = str(payload.get("sample_name") or case_root.name).strip() or case_root.name
    display_name = str(payload.get("display_name") or "").strip() or sample_name

    if dataset and object_name and prompt_id is not None:
        path_tokens = (
            dataset,
            *[part for part in object_name.split("/") if part],
            f"prompt_{prompt_id}",
        )
    else:
        path_tokens = ("cases", sample_name)

    return CaseIdentity(
        case_id=case_id,
        dataset=dataset,
        object_name=object_name,
        prompt_id=prompt_id,
        sample_name=sample_name,
        display_name=display_name,
        path_tokens=tuple(path_tokens),
    )


def _load_mv_cases(dataset_root: Path) -> list[MultiViewCase]:
    cases_dir = dataset_root / "cases"
    cases: list[MultiViewCase] = []
    for case_root in sorted(cases_dir.iterdir()):
        if not case_root.is_dir():
            continue
        payload = _read_json(case_root / "case.json")
        identity = _build_case_identity(case_root, payload)
        cases.append(
            MultiViewCase(
                identity=identity,
                case_root=case_root,
                payload=payload,
            )
        )
    return cases


def _build_angle_grid(
    camera_payload: dict[str, Any],
    key: str,
) -> tuple[list[float], list[float]]:
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


def _resolve_pred_model(pred_root: Path, case: MultiViewCase) -> Path | None:
    prompt_id = case.identity.prompt_id
    object_name = case.identity.object_name
    dataset = case.identity.dataset

    candidates: list[Path] = []
    if dataset and object_name and prompt_id is not None:
        candidates.append(pred_root / dataset / object_name / f"prompt_{prompt_id}" / "edit.glb")
    candidates.append(pred_root / "cases" / case.identity.sample_name / "edit.glb")
    candidates.append(pred_root / case.identity.case_id / "edit.glb")
    candidates.append(pred_root / case.identity.sample_name / "edit.glb")
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
    output_dir.mkdir(parents=True, exist_ok=True)
    render_module.reset_render_pass(scene_manager)
    render_module.clear_render_sequence(str(output_dir))
    _, cam_mats, _, _ = render_module.get_camera_positions_on_sphere(
        center=(0, 0, 0),
        radius=1.8,
        elevations=elevations,
        azimuths=azimuths,
    )
    render_module.add_camera_path(cam_mats)
    render_module.enable_color_output(
        image_size,
        image_size,
        str(output_dir),
        file_format="PNG",
        mode="IMAGE",
        film_transparent=True,
    )
    with render_module.suppress_blender_output(enabled=quiet_blender):
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

    scene_manager = render_module.prepare_scene(
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


def _build_seen_specs(case: MultiViewCase) -> list[ViewSpec]:
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


def _build_novel_specs(case: MultiViewCase, camera_pool: dict[str, Any]) -> list[ViewSpec]:
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


def _load_mask_array(
    mask_path: Path | None,
    image_size: int,
    *,
    inside: bool,
    all_pixels: bool = False,
) -> np.ndarray:
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


def _summarize_case_metrics(case_results: list[CaseResult]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric_name in MV_METRIC_NAMES:
        values = [
            float(metric_value)
            for metric_value in (
                item.metrics.get(metric_name)
                for item in case_results
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


class MultiViewEditTask(BenchmarkTask):
    name = "multi_view_edit"

    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root.expanduser().resolve()
        self._summary_metrics: dict[str, Any] = {}
        self._report_payload: dict[str, Any] = {}

    def discover_cases(self, context: BenchmarkContext) -> list[MultiViewCase]:
        return _load_mv_cases(self.dataset_root)

    def prepare_cases(
        self,
        cases: list[MultiViewCase],
        context: BenchmarkContext,
    ) -> list[PreparedMultiViewCase]:
        pred_root = context.pred_root.expanduser().resolve()
        return [
            PreparedMultiViewCase(case=case, pred_model=_resolve_pred_model(pred_root, case))
            for case in cases
        ]

    def evaluate_cases(
        self,
        prepared_cases: list[PreparedMultiViewCase],
        context: BenchmarkContext,
    ) -> list[CaseResult]:
        task_config = dict(context.task_config)
        image_size = int(task_config.get("image_size", DEFAULT_IMAGE_SIZE))
        overwrite_render = bool(task_config.get("overwrite_render", False))
        quiet_blender = bool(task_config.get("quiet_blender", True))
        cycles_backend = str(task_config.get("cycles_backend", "AUTO"))
        peer_pred_roots = [
            Path(path).expanduser().resolve()
            for path in task_config.get("peer_pred_roots", [])
        ]

        camera_pool = _read_json(self.dataset_root / "camera_pool.json")
        input_cameras = _read_json(self.dataset_root / "input_view_cameras.json")
        seen_grid = _build_angle_grid(input_cameras, "views")
        novel_grid = _build_angle_grid(camera_pool, "views")

        image_evaluator = ImageMetricsEvaluator(
            device=context.device,
            image_size=(image_size, image_size),
            batch_size=32,
            num_workers=0,
            ignore_mask=False,
        )
        geometry_evaluator = GeometryMetricsEvaluator(
            device=context.device,
            batch_size=1,
            num_workers=0,
            ignore_mask=True,
        )

        case_results: list[CaseResult] = []
        raw_case_metrics: dict[str, dict[str, float | None]] = {}

        for prepared_case in prepared_cases:
            case = prepared_case.case
            pred_model = prepared_case.pred_model
            if pred_model is None:
                case_results.append(
                    CaseResult(
                        identity=case.identity,
                        status="failed",
                        metrics={metric_name: None for metric_name in MV_METRIC_NAMES},
                        artifacts={},
                        prompt_text=_prompt_text(case.payload),
                        task_meta={
                            "source_prompt": case.payload.get("source_prompt"),
                            "edit_instruction": case.payload.get("edit_instruction"),
                            "raw_case_id": case.payload.get("case_id"),
                        },
                    )
                )
                continue

            pred_case_dir = pred_model.parent
            mv_eval_dir = pred_case_dir / "mv_eval"
            seen_dir = mv_eval_dir / "seen_views"
            novel_dir = mv_eval_dir / "novel_views"
            seen_dir.mkdir(parents=True, exist_ok=True)
            novel_dir.mkdir(parents=True, exist_ok=True)
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
                case_id=case.identity.case_id,
            )

            peer_novel_dirs = []
            for peer_pred_root in peer_pred_roots:
                peer_model = _resolve_pred_model(peer_pred_root, case)
                if peer_model is None:
                    continue
                peer_mv_eval_dir = peer_model.parent / "mv_eval"
                peer_seen_dir = peer_mv_eval_dir / "seen_views"
                peer_novel_dir = peer_mv_eval_dir / "novel_views"
                peer_seen_dir.mkdir(parents=True, exist_ok=True)
                peer_novel_dir.mkdir(parents=True, exist_ok=True)
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
            raw_case_metrics[case.identity.case_id] = metric_payload
            case_results.append(
                CaseResult(
                    identity=case.identity,
                    status="ok",
                    metrics=metric_payload,
                    artifacts={
                        "source_image": case.case_root / "source_image.png",
                        "edit_image": case.case_root / "edit_image.png",
                        "mask_image": case.case_root / "mask_image.png",
                        "source_model_glb": case.case_root / "source_model.glb",
                        "target_model_glb": case.case_root / "target_model.glb",
                        "mask_glb": case.case_root / "edit_region.glb",
                        "input_views": case.case_root / "input_views",
                        "eval_target_views": case.case_root / "eval_gt" / "target_views",
                        "eval_masks": case.case_root / "eval_gt" / "masks",
                        "edit_glb": pred_model,
                        "images": mv_eval_dir,
                        "seen_views": seen_dir,
                        "novel_views": novel_dir,
                    },
                    prompt_text=_prompt_text(case.payload),
                    task_meta={
                        "source_prompt": case.payload.get("source_prompt"),
                        "edit_instruction": case.payload.get("edit_instruction"),
                        "raw_case_id": case.payload.get("case_id"),
                    },
                )
            )

        self._summary_metrics = _summarize_case_metrics(case_results)
        self._report_payload = {
            "dataset_root": str(self.dataset_root),
            "image_size": image_size,
            "device": context.device,
            "metrics": self._summary_metrics,
            "cases": raw_case_metrics,
        }
        return case_results

    def summarize_run(
        self,
        case_results: list[CaseResult],
        context: BenchmarkContext,
    ) -> dict[str, Any]:
        return dict(self._summary_metrics)

    def build_manifest(
        self,
        case_results: list[CaseResult],
        context: BenchmarkContext,
    ) -> dict[str, Any]:
        manifest = super().build_manifest(case_results, context)
        manifest.update(
            {
                "dataset_root": str(self.dataset_root),
                "case_count": len(case_results),
            }
        )
        return manifest

    def collect_run_artifacts(
        self,
        case_results: list[CaseResult],
        context: BenchmarkContext,
    ) -> dict[str, Any]:
        return {"mv_metrics": dict(self._report_payload)}
