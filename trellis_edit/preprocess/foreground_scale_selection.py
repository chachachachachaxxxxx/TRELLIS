from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class ActiveBBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    width: int
    height: int
    area: int
    center_x: float
    center_y: float
    base_span: int


def compute_active_bbox(mask: np.ndarray) -> ActiveBBox:
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

    coords = np.argwhere(mask)
    if coords.size == 0:
        raise RuntimeError("Preprocess crop produced an empty active region")

    x_min = int(coords[:, 1].min())
    y_min = int(coords[:, 0].min())
    x_max = int(coords[:, 1].max())
    y_max = int(coords[:, 0].max())
    width = int(x_max - x_min + 1)
    height = int(y_max - y_min + 1)
    return ActiveBBox(
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        width=width,
        height=height,
        area=int(width * height),
        center_x=(x_min + x_max) / 2.0,
        center_y=(y_min + y_max) / 2.0,
        base_span=max(x_max - x_min, y_max - y_min, 1),
    )


def build_square_crop_box(active_bbox: ActiveBBox, crop_scale: float) -> tuple[int, int, int, int]:
    if crop_scale <= 0:
        raise ValueError(f"crop_scale must be positive, got {crop_scale}")

    crop_size = max(int(active_bbox.base_span * crop_scale), 1)
    half = crop_size / 2.0
    return (
        int(round(active_bbox.center_x - half)),
        int(round(active_bbox.center_y - half)),
        int(round(active_bbox.center_x + half)),
        int(round(active_bbox.center_y + half)),
    )


def normalize_sparse_coords(
    coords: torch.Tensor | np.ndarray,
    resolution: int = 64,
) -> np.ndarray:
    if isinstance(coords, torch.Tensor):
        coords_np = coords.detach().cpu().numpy()
    else:
        coords_np = np.asarray(coords)

    if coords_np.ndim != 2 or coords_np.shape[1] not in (3, 4):
        raise ValueError(f"Unexpected coords shape: {coords_np.shape}")
    if coords_np.shape[1] == 4:
        coords_np = coords_np[:, 1:]

    coords_np = coords_np.astype(np.int32, copy=False)
    coords_np = coords_np[np.all((coords_np >= 0) & (coords_np < resolution), axis=1)]
    if len(coords_np) == 0:
        return np.zeros((0, 3), dtype=np.int32)
    return np.unique(coords_np, axis=0)


def compute_sparse_structure_metrics(
    coords_xyz: np.ndarray,
    resolution: int = 64,
) -> dict[str, Any]:
    voxel_count = int(len(coords_xyz))
    grid_volume = int(resolution ** 3)
    metrics: dict[str, Any] = {
        "voxel_count": voxel_count,
        "voxel_count_ratio": float(voxel_count / grid_volume),
        "bbox_x_min": None,
        "bbox_y_min": None,
        "bbox_z_min": None,
        "bbox_x_max": None,
        "bbox_y_max": None,
        "bbox_z_max": None,
        "bbox_dx": 0,
        "bbox_dy": 0,
        "bbox_dz": 0,
        "bbox_volume": 0,
        "bbox_volume_ratio": 0.0,
        "bbox_fill_ratio": 0.0,
    }
    if voxel_count == 0:
        return metrics

    bbox_min = coords_xyz.min(axis=0).astype(int)
    bbox_max = coords_xyz.max(axis=0).astype(int)
    extent = (bbox_max - bbox_min + 1).astype(int)
    bbox_volume = int(np.prod(extent))
    metrics.update(
        {
            "bbox_x_min": int(bbox_min[0]),
            "bbox_y_min": int(bbox_min[1]),
            "bbox_z_min": int(bbox_min[2]),
            "bbox_x_max": int(bbox_max[0]),
            "bbox_y_max": int(bbox_max[1]),
            "bbox_z_max": int(bbox_max[2]),
            "bbox_dx": int(extent[0]),
            "bbox_dy": int(extent[1]),
            "bbox_dz": int(extent[2]),
            "bbox_volume": bbox_volume,
            "bbox_volume_ratio": float(bbox_volume / grid_volume),
            "bbox_fill_ratio": float(voxel_count / bbox_volume) if bbox_volume else 0.0,
        }
    )
    return metrics


def select_crop_scale_from_linearized_size_fit(
    *,
    target_bbox_volume_ratio: float,
    source_bbox: ActiveBBox,
    active_bbox: ActiveBBox,
    slope: float,
    intercept: float,
    fallback_scale: float,
    min_scale: float,
    max_scale: float,
) -> tuple[float, dict[str, Any]]:
    selected_default = float(np.clip(fallback_scale, min_scale, max_scale))
    if abs(float(slope)) < 1e-8:
        return selected_default, {
            "mode": "fixed_default",
            "reason": "global_linearized_size_slope_too_small",
            "selected_scale": selected_default,
        }
    if source_bbox.area <= 0 or active_bbox.base_span <= 0:
        return selected_default, {
            "mode": "fixed_default",
            "reason": "invalid_bbox_geometry_for_linearized_size_fit",
            "selected_scale": selected_default,
        }

    target_bbox_volume_linear_ratio = float(np.cbrt(float(target_bbox_volume_ratio)))
    target_proc_bbox_linear_ratio = (
        target_bbox_volume_linear_ratio - float(intercept)
    ) / float(slope)
    source_proc_bbox_linear_numerator = float(np.sqrt(float(source_bbox.area)) / float(active_bbox.base_span))

    if not np.isfinite(target_proc_bbox_linear_ratio) or target_proc_bbox_linear_ratio <= 0.0:
        selected_scale = float(max_scale)
        predicted_proc_bbox_linear_ratio = float(source_proc_bbox_linear_numerator / selected_scale)
        predicted_bbox_volume_linear_ratio = float(intercept + float(slope) * predicted_proc_bbox_linear_ratio)
        return selected_scale, {
            "mode": "global_linear",
            "reason": "target_below_linearized_size_fit_range",
            "selected_scale": selected_scale,
            "raw_selected_scale": None,
            "target_bbox_volume_ratio": float(target_bbox_volume_ratio),
            "target_bbox_volume_linear_ratio": target_bbox_volume_linear_ratio,
            "target_proc_bbox_linear_ratio": float(target_proc_bbox_linear_ratio),
            "source_bbox_area": int(source_bbox.area),
            "source_proc_bbox_linear_numerator": source_proc_bbox_linear_numerator,
            "active_bbox_base_span": int(active_bbox.base_span),
            "predicted_proc_bbox_linear_ratio": predicted_proc_bbox_linear_ratio,
            "predicted_bbox_volume_linear_ratio": predicted_bbox_volume_linear_ratio,
            "predicted_bbox_volume_ratio": float(predicted_bbox_volume_linear_ratio ** 3),
            "linearized_size_slope": float(slope),
            "linearized_size_intercept": float(intercept),
        }

    raw_scale = float(source_proc_bbox_linear_numerator / float(target_proc_bbox_linear_ratio))
    selected_scale = float(np.clip(raw_scale, min_scale, max_scale))
    reason = "target_outside_linearized_size_fit_range"
    if min_scale <= raw_scale <= max_scale:
        reason = "target_mapped_by_global_linearized_size_fit"

    predicted_proc_bbox_linear_ratio = float(source_proc_bbox_linear_numerator / selected_scale)
    predicted_bbox_volume_linear_ratio = float(intercept + float(slope) * predicted_proc_bbox_linear_ratio)
    return selected_scale, {
        "mode": "global_linear",
        "reason": reason,
        "selected_scale": selected_scale,
        "raw_selected_scale": raw_scale,
        "target_bbox_volume_ratio": float(target_bbox_volume_ratio),
        "target_bbox_volume_linear_ratio": target_bbox_volume_linear_ratio,
        "target_proc_bbox_linear_ratio": float(target_proc_bbox_linear_ratio),
        "source_bbox_area": int(source_bbox.area),
        "source_proc_bbox_linear_numerator": source_proc_bbox_linear_numerator,
        "active_bbox_base_span": int(active_bbox.base_span),
        "predicted_proc_bbox_linear_ratio": predicted_proc_bbox_linear_ratio,
        "predicted_bbox_volume_linear_ratio": predicted_bbox_volume_linear_ratio,
        "predicted_bbox_volume_ratio": float(predicted_bbox_volume_linear_ratio ** 3),
        "linearized_size_slope": float(slope),
        "linearized_size_intercept": float(intercept),
    }


def select_crop_scale_from_probes(
    *,
    target_bbox_volume_ratio: float,
    probe_results: list[dict[str, Any]],
    fallback_scale: float,
    min_scale: float,
    max_scale: float,
    min_bbox_volume_delta: float,
) -> tuple[float, dict[str, Any]]:
    selected_default = float(np.clip(fallback_scale, min_scale, max_scale))
    finite_probe_results = [
        result
        for result in probe_results
        if np.isfinite(float(result.get("crop_scale", np.nan)))
        and np.isfinite(float(result.get("bbox_volume_ratio", np.nan)))
        and np.isfinite(float(result.get("proc_fg_bbox_linear_ratio", np.nan)))
    ]
    finite_probe_results.sort(key=lambda item: float(item["crop_scale"]))

    if len(finite_probe_results) < 2:
        return selected_default, {
            "mode": "default",
            "reason": "insufficient_probe_results",
            "selected_scale": selected_default,
        }

    low = finite_probe_results[0]
    high = finite_probe_results[-1]
    low_scale = float(low["crop_scale"])
    high_scale = float(high["crop_scale"])
    low_bbox = float(low["bbox_volume_ratio"])
    high_bbox = float(high["bbox_volume_ratio"])
    low_bbox_linear = float(np.cbrt(low_bbox))
    high_bbox_linear = float(np.cbrt(high_bbox))
    low_proc_bbox_linear = float(low["proc_fg_bbox_linear_ratio"])
    high_proc_bbox_linear = float(high["proc_fg_bbox_linear_ratio"])
    delta_bbox = high_bbox - low_bbox
    delta_bbox_linear = high_bbox_linear - low_bbox_linear
    delta_proc_bbox_linear = high_proc_bbox_linear - low_proc_bbox_linear
    probe_scale_numerator_candidates = [
        low_proc_bbox_linear * low_scale,
        high_proc_bbox_linear * high_scale,
    ]
    probe_scale_numerator = float(np.mean(probe_scale_numerator_candidates))
    target_bbox_linear = float(np.cbrt(float(target_bbox_volume_ratio)))

    probe_meta = {
        "target_bbox_volume_ratio": float(target_bbox_volume_ratio),
        "target_bbox_volume_linear_ratio": target_bbox_linear,
        "low_probe_scale": low_scale,
        "low_probe_bbox_volume_ratio": low_bbox,
        "low_probe_bbox_volume_linear_ratio": low_bbox_linear,
        "low_probe_proc_bbox_linear_ratio": low_proc_bbox_linear,
        "high_probe_scale": high_scale,
        "high_probe_bbox_volume_ratio": high_bbox,
        "high_probe_bbox_volume_linear_ratio": high_bbox_linear,
        "high_probe_proc_bbox_linear_ratio": high_proc_bbox_linear,
        "probe_bbox_volume_delta": float(delta_bbox),
        "probe_bbox_volume_linear_delta": float(delta_bbox_linear),
        "probe_proc_bbox_linear_delta": float(delta_proc_bbox_linear),
        "probe_scale_numerator": probe_scale_numerator,
    }

    if (
        abs(delta_bbox) < float(min_bbox_volume_delta)
        or abs(delta_bbox_linear) < 1e-8
        or abs(delta_proc_bbox_linear) < 1e-8
        or not np.isfinite(probe_scale_numerator)
        or probe_scale_numerator <= 0.0
    ):
        closest = min(
            finite_probe_results,
            key=lambda item: abs(float(np.cbrt(float(item["bbox_volume_ratio"]))) - target_bbox_linear),
        )
        selected_scale = float(np.clip(float(closest["crop_scale"]), min_scale, max_scale))
        return selected_scale, {
            **probe_meta,
            "mode": "closest_probe",
            "reason": "flat_probe_response",
            "selected_scale": selected_scale,
        }

    if delta_bbox >= 0 or delta_bbox_linear >= 0 or delta_proc_bbox_linear >= 0:
        closest = min(
            finite_probe_results,
            key=lambda item: abs(float(np.cbrt(float(item["bbox_volume_ratio"]))) - target_bbox_linear),
        )
        selected_scale = float(np.clip(float(closest["crop_scale"]), min_scale, max_scale))
        return selected_scale, {
            **probe_meta,
            "mode": "closest_probe",
            "reason": "non_monotonic_probe_response",
            "selected_scale": selected_scale,
        }

    target_proc_bbox_linear_ratio = (
        low_proc_bbox_linear
        + (target_bbox_linear - low_bbox_linear) * delta_proc_bbox_linear / delta_bbox_linear
    )
    if not np.isfinite(target_proc_bbox_linear_ratio) or target_proc_bbox_linear_ratio <= 0.0:
        closest = min(
            finite_probe_results,
            key=lambda item: abs(float(np.cbrt(float(item["bbox_volume_ratio"]))) - target_bbox_linear),
        )
        selected_scale = float(np.clip(float(closest["crop_scale"]), min_scale, max_scale))
        return selected_scale, {
            **probe_meta,
            "mode": "closest_probe",
            "reason": "invalid_target_proc_bbox_linear_ratio",
            "selected_scale": selected_scale,
        }

    raw_scale = float(probe_scale_numerator / target_proc_bbox_linear_ratio)
    selected_scale = float(np.clip(raw_scale, min_scale, max_scale))
    if high_bbox_linear <= target_bbox_linear <= low_bbox_linear:
        mode = "interpolate"
        reason = "target_bracketed_by_probes"
    else:
        mode = "clamped_extrapolate"
        reason = "target_outside_probe_range"

    predicted_proc_bbox_linear_ratio = float(probe_scale_numerator / selected_scale)
    predicted_bbox_volume_linear_ratio = float(
        low_bbox_linear
        + (predicted_proc_bbox_linear_ratio - low_proc_bbox_linear) * delta_bbox_linear / delta_proc_bbox_linear
    )
    return selected_scale, {
        **probe_meta,
        "mode": mode,
        "reason": reason,
        "target_proc_bbox_linear_ratio": float(target_proc_bbox_linear_ratio),
        "predicted_proc_bbox_linear_ratio": predicted_proc_bbox_linear_ratio,
        "predicted_bbox_volume_linear_ratio": predicted_bbox_volume_linear_ratio,
        "predicted_bbox_volume_ratio": float(predicted_bbox_volume_linear_ratio ** 3),
        "raw_selected_scale": raw_scale,
        "selected_scale": selected_scale,
    }
