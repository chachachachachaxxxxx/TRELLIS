from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from trellis_edit.benchmark_pages_common import TEXT_ALIGNMENT_VIEW_IDS


def load_single_view_metadata(gt_root: Path) -> list[dict[str, Any]]:
    metadata_path = gt_root / "metadata.json"
    if not metadata_path.is_file():
        return []
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def build_single_view_prompt_text_map(
    metadata: list[dict[str, Any]],
) -> dict[tuple[str, str, int], str]:
    prompt_text_map: dict[tuple[str, str, int], str] = {}
    for item in metadata:
        dataset = item.get("dataset")
        object_name = item.get("source_model")
        if not dataset or not object_name:
            continue
        for prompt_id in (1, 2, 3):
            prompt_key = f"prompt_{prompt_id}"
            prompt_text = item.get(prompt_key)
            if isinstance(prompt_text, str) and prompt_text.strip():
                prompt_text_map[(dataset, object_name, prompt_id)] = prompt_text.strip()
    return prompt_text_map


def build_single_view_case_id(dataset: str, object_name: str, prompt_id: int) -> str:
    return f"{dataset}/{object_name}/prompt_{prompt_id}"


def requested_single_view_case_metrics(requested_metrics: list[str]) -> list[str]:
    run_level_metrics = {"fid", "fid_buffered", "fvd"}
    return [metric for metric in requested_metrics if metric not in run_level_metrics]


def load_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def build_single_view_per_case_metrics(
    *,
    gt_root: Path,
    pred_root: Path,
    metadata: list[dict[str, Any]],
    cases: list[tuple[str, str, int]],
    requested_metrics: list[str],
    detailed_results: dict[str, Any],
) -> dict[str, dict[str, float]]:
    per_case: dict[str, dict[str, float]] = defaultdict(dict)
    if not detailed_results:
        return per_case

    eval_index = _build_eval_index(
        gt_root=gt_root,
        pred_root=pred_root,
        metadata=metadata,
        selected_cases=set(cases),
    )
    image_id_map = eval_index["image_id_map"]
    clip_id_map = eval_index["clip_id_map"]
    model_id_map = eval_index["model_id_map"]
    dino_id_map = eval_index["dino_id_map"]
    floater_id_map = eval_index["floater_id_map"]

    for metric in ("psnr", "ssim", "lpips", "ssim_buffered", "lpips_buffered"):
        if metric not in requested_metrics:
            continue
        payload = detailed_results.get(metric)
        explicit_scores = _aggregate_scores_by_eval_ids_from_payload(payload, image_id_map)
        if explicit_scores is None:
            continue
        for case_id, value in explicit_scores.items():
            per_case[case_id][metric] = value

    if "clip_t" in requested_metrics:
        payload = detailed_results.get("clip_t")
        explicit_scores = _aggregate_scores_by_eval_ids_from_payload(payload, clip_id_map)
        if explicit_scores is not None:
            for case_id, value in explicit_scores.items():
                per_case[case_id]["clip_t"] = value

    if "chamfer" in requested_metrics:
        payload = detailed_results.get("chamfer")
        explicit_scores = _aggregate_scores_by_eval_ids_from_payload(payload, model_id_map)
        if explicit_scores is not None:
            for case_id, value in explicit_scores.items():
                per_case[case_id]["chamfer"] = value

    if "floaters" in requested_metrics:
        payload = detailed_results.get("floaters")
        explicit_scores = _aggregate_scores_by_eval_ids_from_payload(payload, floater_id_map)
        if explicit_scores is not None:
            for case_id, value in explicit_scores.items():
                per_case[case_id]["floaters"] = value

    for metric in ("dino_if_max", "dino_if_mean"):
        if metric not in requested_metrics:
            continue
        payload = detailed_results.get(metric)
        explicit_scores = _aggregate_scores_by_eval_ids_from_payload(payload, dino_id_map)
        if explicit_scores is None:
            continue
        for case_id, value in explicit_scores.items():
            per_case[case_id][metric] = value

    return per_case


def _extract_id_by_prefix(filename: str, prefix: str) -> str | None:
    prefix_token = f"{prefix}_"
    if not filename.startswith(prefix_token):
        return None
    suffix = filename[len(prefix_token) :].split(".", 1)[0]
    return suffix if suffix.isdigit() else None


def _build_eval_index(
    *,
    gt_root: Path,
    pred_root: Path,
    metadata: list[dict[str, Any]],
    selected_cases: set[tuple[str, str, int]],
) -> dict[str, Any]:
    image_id_map: dict[str, str] = {}
    clip_id_map: dict[str, str] = {}
    model_id_map: dict[str, str] = {}
    dino_id_map: dict[str, str] = {}
    floater_id_map: dict[str, str] = {}

    for item in metadata:
        dataset = item.get("dataset")
        object_name = item.get("source_model")
        if not dataset or not object_name:
            continue

        for prompt_id in (1, 2, 3):
            if (dataset, object_name, prompt_id) not in selected_cases:
                continue
            prompt_key = f"prompt_{prompt_id}"
            if prompt_key not in item:
                continue

            case_id = build_single_view_case_id(dataset, object_name, prompt_id)
            gt_render_dir = gt_root / dataset / object_name / "source_model" / "render"
            gt_mask_dir = gt_root / dataset / object_name / prompt_key / "render"
            pred_images_dir = pred_root / dataset / object_name / prompt_key / "images"
            if gt_render_dir.exists() and gt_mask_dir.exists() and pred_images_dir.exists():
                gt_images = {
                    image_id: file
                    for file in gt_render_dir.glob("render_*.png")
                    if (image_id := _extract_id_by_prefix(file.name, "render"))
                }
                gt_masks = {
                    mask_id: file
                    for file in gt_mask_dir.glob("mask_*.png")
                    if (mask_id := _extract_id_by_prefix(file.name, "mask"))
                }
                pred_images = {
                    image_id: file
                    for file in pred_images_dir.glob("render_*.png")
                    if (image_id := _extract_id_by_prefix(file.name, "render"))
                }
                common_ids = set(gt_images) & set(gt_masks) & set(pred_images)
                for image_id in sorted(common_ids, key=int):
                    image_id_map[f"{dataset}_{object_name}_{prompt_key}_{image_id}"] = case_id

                for file in pred_images_dir.glob("render_*.png"):
                    image_id = _extract_id_by_prefix(file.name, "render")
                    if image_id and image_id in TEXT_ALIGNMENT_VIEW_IDS:
                        clip_id_map[f"{dataset}_{object_name}_{prompt_key}_{image_id}"] = case_id

                pred_samples = []
                for file in pred_images_dir.glob("render_*.png"):
                    image_id = _extract_id_by_prefix(file.name, "render")
                    if image_id:
                        pred_samples.append((int(image_id), file))
                if pred_samples and (gt_root / dataset / object_name / prompt_key / "2d_edit.png").exists():
                    dino_id_map[f"{dataset}_{object_name}_{prompt_key}"] = case_id

            gt_model_path = gt_root / dataset / object_name / "source_model" / "model.glb"
            pred_model_path = pred_root / dataset / object_name / prompt_key / "edit.glb"
            mask_path = gt_root / dataset / object_name / prompt_key / "3d_edit_region.glb"
            if pred_model_path.exists():
                floater_id_map[f"{dataset}_{object_name}_{prompt_key}"] = case_id
            if gt_model_path.exists() and pred_model_path.exists() and mask_path.exists():
                model_id_map[f"{dataset}_{object_name}_{prompt_key}"] = case_id

    return {
        "image_id_map": image_id_map,
        "clip_id_map": clip_id_map,
        "model_id_map": model_id_map,
        "dino_id_map": dino_id_map,
        "floater_id_map": floater_id_map,
    }


def _aggregate_scores_by_eval_ids(
    eval_ids: list[Any],
    scores: list[Any],
    eval_id_map: dict[str, str],
) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for eval_id, score in zip(eval_ids, scores):
        if score is None:
            continue
        case_id = eval_id_map.get(str(eval_id))
        if case_id is None:
            continue
        buckets[case_id].append(float(score))
    return {
        case_id: float(sum(values) / len(values))
        for case_id, values in buckets.items()
        if values
    }


def _aggregate_scores_by_eval_ids_from_payload(
    payload: Any,
    eval_id_map: dict[str, str],
) -> dict[str, float] | None:
    if not isinstance(payload, dict):
        return None

    eval_ids = payload.get("all_ids")
    scores = payload.get("all_scores")
    if not isinstance(eval_ids, list) or not isinstance(scores, list):
        return None
    if len(eval_ids) != len(scores):
        return None
    return _aggregate_scores_by_eval_ids(eval_ids, scores, eval_id_map)


__all__ = [
    "build_single_view_case_id",
    "build_single_view_per_case_metrics",
    "build_single_view_prompt_text_map",
    "load_optional_json",
    "load_single_view_metadata",
    "requested_single_view_case_metrics",
]
