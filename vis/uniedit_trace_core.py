#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


def _peek_arg(flag: str, default: str = "") -> str:
    if flag not in sys.argv:
        return default
    idx = sys.argv.index(flag)
    if idx + 1 >= len(sys.argv):
        return default
    return sys.argv[idx + 1]


_attn_backend = _peek_arg("--attn-backend", os.environ.get("ATTN_BACKEND", ""))
if _attn_backend:
    os.environ["ATTN_BACKEND"] = _attn_backend

os.environ.setdefault("SPCONV_ALGO", "native")

import numpy as np
import torch
import trimesh
from PIL import Image
from tqdm import tqdm

import example_image_prompt_to_prompt as image_p2p
import example_image_prompt_to_prompt_rf_inversion as rf_utils
import example_image_uniedit_rf_inversion as uniedit
from output_layout import build_output_layout, sanitize_output_name


TRACE_METHOD_NAME = "image_uniedit_rf_inversion_trace"
EventCallback = Callable[[dict], None]

_BOOTSTRAPPED = False
_PIPELINE_CACHE: dict[str, Any] = {}


@dataclass
class TraceConfig:
    model: str = "microsoft/TRELLIS-image-large"
    source_model: str = ""
    input_model: str = ""
    render_dir: str = ""
    image_dir: str = ""
    source_image: str = ""
    edit_image: str = ""
    mask_image: str = ""
    mask_glb: str = ""
    output_path: str = ""
    output_dir: str = ""
    case_name: str = ""
    outputs_root: str = "outputs"
    seed: int = 1
    ss_steps: Optional[int] = None
    slat_steps: Optional[int] = None
    ss_cfg: Optional[float] = None
    slat_cfg: Optional[float] = None
    ss_inverse_cfg: Optional[float] = None
    slat_inverse_cfg: Optional[float] = None
    ss_omega: float = 1.0
    slat_omega: float = 1.0
    cfg_interval_start: float = 0.5
    cfg_interval_end: float = 1.0
    preprocess: bool = True
    mask_threshold: int = 127
    auto_mask_threshold: int = 24
    auto_mask_max_filter: int = 7
    trace_every: int = 5
    render_videos: bool = False
    export_glb: bool = True
    export_ply: bool = True
    quiet: bool = True
    attn_backend: str = field(default_factory=lambda: os.environ.get("ATTN_BACKEND", ""))


@dataclass
class StepContext:
    index: int
    name: str
    title: str
    dir: Path
    counters: Dict[str, int] = field(default_factory=lambda: {"input": 0, "output": 0})


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def bootstrap_trellis_bindings():
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        from trellis.pipelines import TrellisImageTo3DPipeline

        return TrellisImageTo3DPipeline

    from trellis.modules.sparse.basic import SparseTensor as _SparseTensor
    from trellis.pipelines import TrellisImageTo3DPipeline as _TrellisImageTo3DPipeline

    image_p2p.SparseTensor = _SparseTensor
    rf_utils.image_p2p.SparseTensor = _SparseTensor
    uniedit.image_p2p.SparseTensor = _SparseTensor
    _BOOTSTRAPPED = True
    return _TrellisImageTo3DPipeline


def get_pipeline(model_name: str):
    TrellisImageTo3DPipeline = bootstrap_trellis_bindings()
    pipeline = _PIPELINE_CACHE.get(model_name)
    if pipeline is None:
        pipeline = TrellisImageTo3DPipeline.from_pretrained(model_name)
        _PIPELINE_CACHE[model_name] = pipeline
    return pipeline


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def tensor_stats(tensor: torch.Tensor) -> Dict[str, Any]:
    arr = tensor.detach().float().cpu()
    stats = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(arr.min().item()) if arr.numel() else 0.0,
        "max": float(arr.max().item()) if arr.numel() else 0.0,
        "mean": float(arr.mean().item()) if arr.numel() else 0.0,
        "std": float(arr.std().item()) if arr.numel() > 1 else 0.0,
    }
    return stats


def array_stats(array: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(array)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(arr.min()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size > 1 else 0.0,
    }


def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    low = float(arr.min())
    high = float(arr.max())
    if high - low < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = (arr - low) / (high - low)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def image_from_uint8(array: np.ndarray) -> Image.Image:
    arr = np.asarray(array)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    if arr.ndim == 3 and arr.shape[2] == 3:
        return Image.fromarray(arr, mode="RGB")
    raise RuntimeError(f"Unsupported image array shape: {arr.shape}")


def make_projection_strip(coords: np.ndarray, size: int = 256) -> np.ndarray:
    xyz = np.asarray(coords, dtype=np.float32)
    if xyz.size == 0:
        blank = np.zeros((size, size), dtype=np.uint8)
        return np.concatenate([blank, blank, blank], axis=1)

    mins = xyz.min(axis=0, keepdims=True)
    maxs = xyz.max(axis=0, keepdims=True)
    denom = np.maximum(maxs - mins, 1e-6)
    norm = (xyz - mins) / denom

    def project(a: int, b: int) -> np.ndarray:
        canvas = np.zeros((size, size), dtype=np.uint8)
        ij = np.clip((norm[:, [a, b]] * (size - 1)).round().astype(np.int32), 0, size - 1)
        canvas[size - 1 - ij[:, 1], ij[:, 0]] = 255
        return canvas

    xy = project(0, 1)
    xz = project(0, 2)
    yz = project(1, 2)
    return np.concatenate([xy, xz, yz], axis=1)


def colorize_projection_strip(coords: np.ndarray, colors: np.ndarray, size: int = 256) -> np.ndarray:
    xyz = np.asarray(coords, dtype=np.float32)
    rgb = np.asarray(colors, dtype=np.float32)
    if xyz.size == 0:
        blank = np.zeros((size, size, 3), dtype=np.uint8)
        return np.concatenate([blank, blank, blank], axis=1)

    rgb = np.clip(rgb, 0.0, 1.0)
    mins = xyz.min(axis=0, keepdims=True)
    maxs = xyz.max(axis=0, keepdims=True)
    denom = np.maximum(maxs - mins, 1e-6)
    norm = (xyz - mins) / denom

    def project(a: int, b: int) -> np.ndarray:
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        ij = np.clip((norm[:, [a, b]] * (size - 1)).round().astype(np.int32), 0, size - 1)
        for idx, (x, y) in enumerate(ij):
            canvas[size - 1 - y, x] = np.clip(rgb[idx] * 255.0, 0, 255).astype(np.uint8)
        return canvas

    xy = project(0, 1)
    xz = project(0, 2)
    yz = project(1, 2)
    return np.concatenate([xy, xz, yz], axis=1)


def dense_projection_strip(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().float().cpu().numpy()
    if arr.ndim == 5:
        arr = arr[0]
    if arr.ndim == 4:
        arr = arr.mean(axis=0)
    if arr.ndim == 3:
        xy = normalize_to_uint8(arr.max(axis=0))
        xz = normalize_to_uint8(arr.max(axis=1))
        yz = normalize_to_uint8(arr.max(axis=2))
        return np.concatenate([xy, xz, yz], axis=1)
    if arr.ndim == 2:
        heat = normalize_to_uint8(arr)
        return np.concatenate([heat, heat, heat], axis=1)
    flat = normalize_to_uint8(arr.reshape(1, -1))
    return np.concatenate([flat, flat, flat], axis=1)


def compute_pca_colors(features: np.ndarray) -> np.ndarray:
    feats = np.asarray(features, dtype=np.float32)
    if feats.ndim != 2 or feats.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    centered = feats - feats.mean(axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        basis = vh[:3].T
        proj = centered @ basis
    except np.linalg.LinAlgError:
        proj = centered[:, : min(3, centered.shape[1])]
    if proj.shape[1] < 3:
        proj = np.pad(proj, ((0, 0), (0, 3 - proj.shape[1])))
    mins = proj.min(axis=0, keepdims=True)
    maxs = proj.max(axis=0, keepdims=True)
    denom = np.maximum(maxs - mins, 1e-6)
    return (proj - mins) / denom


def value_to_rgb(values: np.ndarray) -> np.ndarray:
    val = np.asarray(values, dtype=np.float32).reshape(-1)
    if val.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    lo = float(val.min())
    hi = float(val.max())
    if hi - lo < 1e-8:
        norm = np.zeros_like(val)
    else:
        norm = (val - lo) / (hi - lo)
    return np.stack([norm, 0.3 + 0.4 * (1.0 - np.abs(norm - 0.5) * 2.0), 1.0 - norm], axis=1)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_mesh_ply(path: Path, mesh) -> None:
    vertices = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(path)


def save_point_cloud_ply(path: Path, points: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
    pts = np.asarray(points, dtype=np.float32)
    if colors is None:
        cloud = trimesh.points.PointCloud(pts)
    else:
        rgb = np.asarray(colors, dtype=np.float32)
        rgb = np.clip(rgb, 0.0, 1.0)
        cloud = trimesh.points.PointCloud(pts, colors=(rgb * 255.0).astype(np.uint8))
    cloud.export(path)


def should_trace_step(step_idx: int, total_steps: int, trace_every: int) -> bool:
    interval = max(int(trace_every), 1)
    if step_idx == 0 or step_idx == total_steps - 1:
        return True
    return ((step_idx + 1) % interval) == 0


class TraceRecorder:
    def __init__(
        self,
        out_dir: Path,
        method_name: str,
        case_name: str,
        run_config: Dict[str, Any],
        on_event: Optional[EventCallback] = None,
    ):
        self.out_dir = ensure_dir(out_dir)
        self.method_name = method_name
        self.case_name = case_name
        self.on_event = on_event
        self.logs: List[str] = []
        self.manifest = {
            "schema_version": 1,
            "method_name": method_name,
            "case_name": case_name,
            "output_dir": str(self.out_dir.resolve()),
            "run_config": run_config,
            "steps": [],
            "artifacts": [],
            "logs": self.logs,
        }
        self.manifest_path = self.out_dir / "trace_manifest.json"
        self.write_manifest()

    def write_manifest(self) -> None:
        save_json(self.manifest_path, self.manifest)

    def snapshot(self) -> dict:
        return copy.deepcopy(self.manifest)

    def emit(self, event_type: str, message: str, selected_artifact_id: Optional[str] = None) -> None:
        self.logs.append(message)
        self.write_manifest()
        if self.on_event is not None:
            self.on_event(
                {
                    "type": event_type,
                    "message": message,
                    "selected_artifact_id": selected_artifact_id,
                    "manifest": self.snapshot(),
                }
            )

    def log(self, message: str) -> None:
        self.emit("log", message)

    def begin_step(self, name: str, title: str, note: str = "") -> StepContext:
        step_index = len(self.manifest["steps"]) + 1
        slug = sanitize_output_name(name, fallback=f"step-{step_index}")
        step_dir = ensure_dir(self.out_dir / f"step_{step_index:03d}_{slug}")
        step_record = {
            "index": step_index,
            "name": name,
            "title": title,
            "note": note,
            "dir": str(step_dir.relative_to(self.out_dir)),
            "artifacts": [],
        }
        self.manifest["steps"].append(step_record)
        self.write_manifest()
        self.emit("step", f"Step {step_index}: {title}")
        return StepContext(index=step_index, name=name, title=title, dir=step_dir)

    def _artifact_stem(self, step: StepContext, role: str, logical_name: str) -> str:
        slot = "input" if role == "input" else "output"
        step.counters[slot] += 1
        return f"第{step.index}步_{slot}{step.counters[slot]}_{logical_name}"

    def _add_artifact(
        self,
        step: StepContext,
        logical_name: str,
        role: str,
        path: Path,
        preview_path: Optional[Path],
        media_type: str,
        viewer_hint: str,
        metadata: Dict[str, Any],
        download_path: Optional[Path] = None,
        video_path: Optional[Path] = None,
    ) -> dict:
        artifact_id = f"{step.index:03d}_{len(self.manifest['artifacts']) + 1:04d}_{logical_name}"
        artifact = {
            "id": artifact_id,
            "label": f"{step.index:02d} | {step.title} | {logical_name}",
            "step_index": step.index,
            "step_name": step.name,
            "step_title": step.title,
            "logical_name": logical_name,
            "role": role,
            "path": str(path.relative_to(self.out_dir)),
            "preview_path": str(preview_path.relative_to(self.out_dir)) if preview_path is not None else None,
            "download_path": str(download_path.relative_to(self.out_dir)) if download_path is not None else None,
            "video_path": str(video_path.relative_to(self.out_dir)) if video_path is not None else None,
            "media_type": media_type,
            "viewer_hint": viewer_hint,
            "metadata": metadata,
        }
        self.manifest["artifacts"].append(artifact)
        self.manifest["steps"][step.index - 1]["artifacts"].append(artifact_id)
        self.emit("artifact", f"Saved {artifact['label']}", selected_artifact_id=artifact_id)
        return artifact

    def record_json(self, step: StepContext, logical_name: str, payload: Dict[str, Any], role: str = "output") -> dict:
        stem = self._artifact_stem(step, role, logical_name)
        path = step.dir / f"{stem}.json"
        save_json(path, payload)
        return self._add_artifact(
            step=step,
            logical_name=logical_name,
            role=role,
            path=path,
            preview_path=None,
            media_type="application/json",
            viewer_hint="json",
            metadata={"keys": sorted(payload.keys())},
            download_path=path,
        )

    def record_image(self, step: StepContext, logical_name: str, image: Image.Image, role: str = "output", metadata: Optional[Dict[str, Any]] = None) -> dict:
        stem = self._artifact_stem(step, role, logical_name)
        path = step.dir / f"{stem}.png"
        image.save(path)
        return self._add_artifact(
            step=step,
            logical_name=logical_name,
            role=role,
            path=path,
            preview_path=path,
            media_type="image/png",
            viewer_hint="image",
            metadata=metadata or {"size": list(image.size), "mode": image.mode},
            download_path=path,
        )

    def record_dense_tensor(self, step: StepContext, logical_name: str, tensor: torch.Tensor, role: str = "output", extra_metadata: Optional[Dict[str, Any]] = None) -> dict:
        stem = self._artifact_stem(step, role, logical_name)
        raw_path = step.dir / f"{stem}.npy"
        preview_path = step.dir / f"{stem}.png"
        array = tensor.detach().float().cpu().numpy()
        np.save(raw_path, array)
        image_from_uint8(dense_projection_strip(tensor)).save(preview_path)
        metadata = tensor_stats(tensor)
        if extra_metadata:
            metadata.update(extra_metadata)
        return self._add_artifact(
            step=step,
            logical_name=logical_name,
            role=role,
            path=raw_path,
            preview_path=preview_path,
            media_type="application/x-npy",
            viewer_hint="image",
            metadata=metadata,
            download_path=raw_path,
        )

    def record_coords(
        self,
        step: StepContext,
        logical_name: str,
        coords: torch.Tensor | np.ndarray,
        role: str = "output",
        colors: Optional[np.ndarray] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> dict:
        stem = self._artifact_stem(step, role, logical_name)
        raw_path = step.dir / f"{stem}.npy"
        ply_path = step.dir / f"{stem}.ply"
        preview_path = step.dir / f"{stem}.png"
        arr = coords.detach().cpu().numpy() if torch.is_tensor(coords) else np.asarray(coords)
        xyz = arr[:, -3:] if arr.ndim == 2 and arr.shape[1] >= 3 else arr.reshape(-1, 3)
        np.save(raw_path, arr)
        save_point_cloud_ply(ply_path, xyz.astype(np.float32), colors=colors)
        if colors is None:
            preview = make_projection_strip(xyz)
            preview_img = image_from_uint8(preview)
        else:
            preview = colorize_projection_strip(xyz, colors)
            preview_img = image_from_uint8(preview)
        preview_img.save(preview_path)
        metadata = {
            "coords": array_stats(arr),
            "xyz_min": xyz.min(axis=0).tolist() if xyz.size else [0.0, 0.0, 0.0],
            "xyz_max": xyz.max(axis=0).tolist() if xyz.size else [0.0, 0.0, 0.0],
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        return self._add_artifact(
            step=step,
            logical_name=logical_name,
            role=role,
            path=ply_path,
            preview_path=preview_path,
            media_type="model/ply",
            viewer_hint="3d",
            metadata=metadata,
            download_path=raw_path,
        )

    def record_sparse_tensor(self, step: StepContext, logical_name: str, tensor, role: str = "output", extra_metadata: Optional[Dict[str, Any]] = None) -> dict:
        stem = self._artifact_stem(step, role, logical_name)
        raw_path = step.dir / f"{stem}.npz"
        ply_path = step.dir / f"{stem}.ply"
        preview_path = step.dir / f"{stem}.png"
        coords = tensor.coords.detach().cpu().numpy()
        feats = tensor.feats.detach().float().cpu().numpy()
        xyz = coords[:, -3:].astype(np.float32)
        colors = compute_pca_colors(feats.reshape(feats.shape[0], -1))
        np.savez_compressed(raw_path, coords=coords, feats=feats)
        save_point_cloud_ply(ply_path, xyz, colors=colors)
        image_from_uint8(colorize_projection_strip(xyz, colors)).save(preview_path)
        metadata = {
            "coords": array_stats(coords),
            "feats": array_stats(feats),
            "num_points": int(coords.shape[0]),
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        return self._add_artifact(
            step=step,
            logical_name=logical_name,
            role=role,
            path=ply_path,
            preview_path=preview_path,
            media_type="model/ply",
            viewer_hint="3d",
            metadata=metadata,
            download_path=raw_path,
        )

    def record_scalar_on_coords(
        self,
        step: StepContext,
        logical_name: str,
        coords: torch.Tensor | np.ndarray,
        values: torch.Tensor | np.ndarray,
        role: str = "output",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> dict:
        stem = self._artifact_stem(step, role, logical_name)
        raw_path = step.dir / f"{stem}.npz"
        ply_path = step.dir / f"{stem}.ply"
        preview_path = step.dir / f"{stem}.png"
        arr_coords = coords.detach().cpu().numpy() if torch.is_tensor(coords) else np.asarray(coords)
        arr_values = values.detach().float().cpu().numpy() if torch.is_tensor(values) else np.asarray(values)
        xyz = arr_coords[:, -3:].astype(np.float32)
        colors = value_to_rgb(arr_values.reshape(-1))
        np.savez_compressed(raw_path, coords=arr_coords, values=arr_values)
        save_point_cloud_ply(ply_path, xyz, colors=colors)
        image_from_uint8(colorize_projection_strip(xyz, colors)).save(preview_path)
        metadata = {
            "coords": array_stats(arr_coords),
            "values": array_stats(arr_values),
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        return self._add_artifact(
            step=step,
            logical_name=logical_name,
            role=role,
            path=ply_path,
            preview_path=preview_path,
            media_type="model/ply",
            viewer_hint="3d",
            metadata=metadata,
            download_path=raw_path,
        )

    def record_existing_file(
        self,
        step: StepContext,
        logical_name: str,
        path: Path,
        role: str = "output",
        preview_path: Optional[Path] = None,
        video_path: Optional[Path] = None,
        media_type: Optional[str] = None,
        viewer_hint: str = "file",
        metadata: Optional[Dict[str, Any]] = None,
        download_path: Optional[Path] = None,
    ) -> dict:
        ext = path.suffix.lower()
        guessed_media = {
            ".glb": "model/gltf-binary",
            ".ply": "model/ply",
            ".png": "image/png",
            ".mp4": "video/mp4",
            ".json": "application/json",
            ".npy": "application/x-npy",
            ".npz": "application/x-npz",
        }.get(ext, "application/octet-stream")
        return self._add_artifact(
            step=step,
            logical_name=logical_name,
            role=role,
            path=path,
            preview_path=preview_path,
            media_type=media_type or guessed_media,
            viewer_hint=viewer_hint,
            metadata=metadata or {},
            download_path=download_path or path,
            video_path=video_path,
        )


class TracedSecondOrderRFSampler(rf_utils.SecondOrderRFSampler):
    def __init__(self, trace_fn: Optional[Callable[[dict], None]] = None, trace_every: int = 5):
        super().__init__()
        self.trace_fn = trace_fn
        self.trace_every = max(int(trace_every), 1)

    def sample(
        self,
        model,
        sample,
        cond_dict: dict,
        steps: int,
        rescale_t: float,
        cfg_strength: float,
        cfg_interval: Tuple[float, float],
        inverse: bool,
        verbose: bool = True,
        trace_stage: str = "",
    ):
        t_seq = np.linspace(1.0, 0.0, int(steps) + 1)
        t_seq = rescale_t * t_seq / (1.0 + (rescale_t - 1.0) * t_seq)
        if inverse:
            t_seq = t_seq[::-1]
            desc = "RF inversion"
        else:
            desc = "RF denoise"
        t_pairs = list((float(t_seq[i]), float(t_seq[i + 1])) for i in range(len(t_seq) - 1))
        total = len(t_pairs)

        for step_idx, (t_curr, t_next) in enumerate(tqdm(t_pairs, desc=desc, disable=not verbose)):
            pred = self._guided_prediction(model, sample, t_curr, cond_dict, cfg_strength, cfg_interval)
            dt = t_next - t_curr
            sample_mid = sample + 0.5 * dt * pred
            t_mid = t_curr + 0.5 * dt
            pred_mid = self._guided_prediction(model, sample_mid, t_mid, cond_dict, cfg_strength, cfg_interval)
            first_order = (pred_mid - pred) / (0.5 * dt)
            sample_next = sample + dt * pred - 0.5 * (dt ** 2) * first_order

            if self.trace_fn is not None and should_trace_step(step_idx, total, self.trace_every):
                self.trace_fn(
                    {
                        "stage": trace_stage,
                        "step_idx": step_idx,
                        "total_steps": total,
                        "t_curr": t_curr,
                        "t_next": t_next,
                        "sample_next": sample_next,
                        "pred": pred,
                    }
                )
            sample = sample_next
        return sample


class TracedUniEditRFSampler(uniedit.UniEditRFSampler):
    def __init__(self, trace_fn: Optional[Callable[[dict], None]] = None, trace_every: int = 5):
        super().__init__()
        self.trace_fn = trace_fn
        self.trace_every = max(int(trace_every), 1)

    def _merged_prediction_detail(
        self,
        model,
        sample,
        t_value: float,
        source_cond: torch.Tensor,
        target_cond: torch.Tensor,
        neg_cond: torch.Tensor,
        cfg_strength: float,
        cfg_interval: Tuple[float, float],
        omega: float,
        selector: Optional[torch.Tensor],
        mode: str,
    ):
        pred_tgt = self._guided_prediction_for_cond(
            model=model,
            sample=sample,
            t_value=t_value,
            cond=target_cond,
            neg_cond=neg_cond,
            cfg_strength=cfg_strength,
            cfg_interval=cfg_interval,
        )
        detail = {"pred_tgt": pred_tgt, "guidance": None, "save_map": None}
        if mode == "target_only":
            detail["pred"] = pred_tgt
            return pred_tgt, detail

        pred_src = self._guided_prediction_for_cond(
            model=model,
            sample=sample,
            t_value=t_value,
            cond=source_cond,
            neg_cond=neg_cond,
            cfg_strength=cfg_strength,
            cfg_interval=cfg_interval,
        )
        guidance = pred_tgt - pred_src
        save_map = uniedit.compute_uniedit_map(guidance, selector=selector if mode == "preserve_overlap" else None)
        fused = pred_tgt * save_map + pred_src * (1.0 - save_map)
        pred = fused + guidance * ((1.0 + save_map) * float(omega))
        if mode == "preserve_overlap" and selector is not None:
            pred = pred * selector + pred_tgt * (1.0 - selector)

        detail.update(
            {
                "pred": pred,
                "guidance": guidance,
                "save_map": save_map,
            }
        )
        return pred, detail

    def sample(
        self,
        model,
        sample,
        source_cond: torch.Tensor,
        target_cond: torch.Tensor,
        neg_cond: torch.Tensor,
        steps: int,
        rescale_t: float,
        cfg_strength: float,
        cfg_interval: Tuple[float, float],
        omega: float,
        selector: Optional[torch.Tensor] = None,
        mode: str = "full_uniedit",
        verbose: bool = True,
        trace_stage: str = "",
    ):
        t_seq = np.linspace(1.0, 0.0, int(steps) + 1)
        t_seq = rescale_t * t_seq / (1.0 + (rescale_t - 1.0) * t_seq)
        t_pairs = list((float(t_seq[i]), float(t_seq[i + 1])) for i in range(len(t_seq) - 1))
        total = len(t_pairs)

        desc_map = {
            "full_uniedit": "UniEdit denoise",
            "preserve_overlap": "UniEdit denoise (preserve overlap)",
            "target_only": "Target-only denoise",
        }
        iterator = tqdm(t_pairs, desc=desc_map.get(mode, "UniEdit denoise"), disable=not verbose)
        for step_idx, (t_curr, t_next) in enumerate(iterator):
            pred, detail = self._merged_prediction_detail(
                model=model,
                sample=sample,
                t_value=t_curr,
                source_cond=source_cond,
                target_cond=target_cond,
                neg_cond=neg_cond,
                cfg_strength=cfg_strength,
                cfg_interval=cfg_interval,
                omega=omega,
                selector=selector,
                mode=mode,
            )
            dt = t_next - t_curr
            sample_mid = sample + 0.5 * dt * pred
            t_mid = t_curr + 0.5 * dt
            pred_mid, _ = self._merged_prediction_detail(
                model=model,
                sample=sample_mid,
                t_value=t_mid,
                source_cond=source_cond,
                target_cond=target_cond,
                neg_cond=neg_cond,
                cfg_strength=cfg_strength,
                cfg_interval=cfg_interval,
                omega=omega,
                selector=selector,
                mode=mode,
            )
            first_order = (pred_mid - pred) / (0.5 * dt)
            sample_next = sample + dt * pred - 0.5 * (dt ** 2) * first_order

            if self.trace_fn is not None and should_trace_step(step_idx, total, self.trace_every):
                self.trace_fn(
                    {
                        "stage": trace_stage,
                        "step_idx": step_idx,
                        "total_steps": total,
                        "t_curr": t_curr,
                        "t_next": t_next,
                        "sample_next": sample_next,
                        "guidance": detail["guidance"],
                        "save_map": detail["save_map"],
                        "mode": mode,
                    }
                )
            sample = sample_next
        return sample


def resolve_trace_paths(config: TraceConfig) -> dict:
    args = argparse.Namespace(
        source_model=config.source_model,
        input_model=config.input_model,
        render_dir=config.render_dir,
        image_dir=config.image_dir,
        source_image=config.source_image,
        edit_image=config.edit_image,
        mask_image=config.mask_image,
        output_path=config.output_path,
    )
    args.source_model = args.source_model or args.input_model
    return rf_utils.resolve_pipeline_paths(args)


def determine_case_name(config: TraceConfig, resolved_paths: dict) -> str:
    explicit = config.case_name.strip()
    if explicit:
        return sanitize_output_name(explicit, fallback="uniedit-trace")
    return sanitize_output_name(
        f"{resolved_paths['asset_dir'].name}_to_{resolved_paths['edit_image_path'].stem}",
        fallback="uniedit-trace",
    )


def determine_output_dir(config: TraceConfig, case_name: str) -> Path:
    if config.output_dir.strip():
        return ensure_dir(Path(config.output_dir).expanduser().resolve())
    layout = build_output_layout(TRACE_METHOD_NAME, case_name, outputs_root=config.outputs_root)
    return ensure_dir(layout.edit_dir.resolve())


def save_condition_tensor_bundle(
    recorder: TraceRecorder,
    step: StepContext,
    logical_prefix: str,
    cond_dict: dict,
) -> None:
    cond = cond_dict["cond"]
    neg = cond_dict["neg_cond"]
    recorder.record_dense_tensor(step, f"{logical_prefix}_cond", cond, extra_metadata={"kind": "cond"})
    recorder.record_dense_tensor(step, f"{logical_prefix}_neg_cond", neg, extra_metadata={"kind": "neg_cond"})


def trace_second_order_events(recorder: TraceRecorder, step: StepContext, payload: dict, sample_name: str) -> None:
    suffix = f"iter_{payload['step_idx'] + 1:03d}_t_{payload['t_curr']:.4f}_to_{payload['t_next']:.4f}"
    recorder.record_dense_tensor(
        step,
        f"{sample_name}_{suffix}",
        payload["sample_next"],
        extra_metadata={
            "trace_stage": payload["stage"],
            "step_idx": int(payload["step_idx"]),
            "total_steps": int(payload["total_steps"]),
            "t_curr": float(payload["t_curr"]),
            "t_next": float(payload["t_next"]),
        },
    ) if torch.is_tensor(payload["sample_next"]) else recorder.record_sparse_tensor(
        step,
        f"{sample_name}_{suffix}",
        payload["sample_next"],
        extra_metadata={
            "trace_stage": payload["stage"],
            "step_idx": int(payload["step_idx"]),
            "total_steps": int(payload["total_steps"]),
            "t_curr": float(payload["t_curr"]),
            "t_next": float(payload["t_next"]),
        },
    )


def trace_uniedit_events(recorder: TraceRecorder, step: StepContext, payload: dict, sample_name: str) -> None:
    suffix = f"iter_{payload['step_idx'] + 1:03d}_t_{payload['t_curr']:.4f}_to_{payload['t_next']:.4f}"
    sample = payload["sample_next"]
    metadata = {
        "trace_stage": payload["stage"],
        "mode": payload["mode"],
        "step_idx": int(payload["step_idx"]),
        "total_steps": int(payload["total_steps"]),
        "t_curr": float(payload["t_curr"]),
        "t_next": float(payload["t_next"]),
    }
    if torch.is_tensor(sample):
        recorder.record_dense_tensor(step, f"{sample_name}_{suffix}", sample, extra_metadata=metadata)
    else:
        recorder.record_sparse_tensor(step, f"{sample_name}_{suffix}", sample, extra_metadata=metadata)

    guidance = payload.get("guidance")
    if guidance is not None:
        if torch.is_tensor(guidance):
            recorder.record_dense_tensor(step, f"guidance_{suffix}", guidance, extra_metadata=metadata)
        else:
            recorder.record_sparse_tensor(step, f"guidance_{suffix}", guidance, extra_metadata=metadata)

    save_map = payload.get("save_map")
    if save_map is not None:
        if torch.is_tensor(sample):
            recorder.record_dense_tensor(step, f"save_map_{suffix}", save_map, extra_metadata=metadata)
        else:
            recorder.record_scalar_on_coords(
                step,
                f"save_map_{suffix}",
                coords=sample.coords,
                values=save_map,
                extra_metadata=metadata,
            )


def record_mesh_artifact(
    recorder: TraceRecorder,
    step: StepContext,
    logical_name: str,
    mesh,
    video_path: Optional[Path] = None,
) -> dict:
    stem = recorder._artifact_stem(step, "output", logical_name)
    mesh_path = step.dir / f"{stem}.ply"
    preview_path = step.dir / f"{stem}.png"
    save_mesh_ply(mesh_path, mesh)
    vertices = mesh.vertices.detach().cpu().numpy()
    image_from_uint8(make_projection_strip(vertices.astype(np.float32))).save(preview_path)
    return recorder._add_artifact(
        step=step,
        logical_name=logical_name,
        role="output",
        path=mesh_path,
        preview_path=preview_path,
        media_type="model/ply",
        viewer_hint="3d",
        metadata={
            "num_vertices": int(mesh.vertices.shape[0]),
            "num_faces": int(mesh.faces.shape[0]),
            "success": bool(getattr(mesh, "success", True)),
        },
        download_path=mesh_path,
        video_path=video_path,
    )


def record_gaussian_artifact(
    recorder: TraceRecorder,
    step: StepContext,
    logical_name: str,
    gaussian,
    video_path: Optional[Path] = None,
) -> dict:
    stem = recorder._artifact_stem(step, "output", logical_name)
    ply_path = step.dir / f"{stem}.ply"
    preview_path = step.dir / f"{stem}.png"
    gaussian.save_ply(str(ply_path))
    xyz = gaussian.get_xyz.detach().cpu().numpy()
    image_from_uint8(make_projection_strip(xyz.astype(np.float32))).save(preview_path)
    return recorder._add_artifact(
        step=step,
        logical_name=logical_name,
        role="output",
        path=ply_path,
        preview_path=preview_path,
        media_type="model/ply",
        viewer_hint="3d",
        metadata={
            "num_points": int(gaussian.get_xyz.shape[0]),
            "xyz": tensor_stats(gaussian.get_xyz),
            "opacity": tensor_stats(gaussian.get_opacity),
        },
        download_path=ply_path,
        video_path=video_path,
    )


def maybe_render_video(path: Path, frames: Sequence[np.ndarray], fps: int = 30) -> None:
    if not frames:
        return
    import imageio.v2 as imageio

    imageio.mimsave(path, list(frames), fps=fps)


def record_variant_outputs(
    recorder: TraceRecorder,
    step: StepContext,
    pipeline,
    variant: str,
    outputs: dict,
    render_videos: bool,
    export_glb: bool,
) -> None:
    video_paths: Dict[str, Path] = {}
    if render_videos:
        from trellis.utils import render_utils

        if "mesh" in outputs:
            video_paths["mesh"] = step.dir / f"{variant}_mesh.mp4"
            maybe_render_video(video_paths["mesh"], render_utils.render_video(outputs["mesh"][0])["normal"])
        if "gaussian" in outputs:
            video_paths["gaussian"] = step.dir / f"{variant}_gaussian.mp4"
            maybe_render_video(video_paths["gaussian"], render_utils.render_video(outputs["gaussian"][0])["color"])

    if "mesh" in outputs:
        record_mesh_artifact(
            recorder=recorder,
            step=step,
            logical_name=f"{variant}_mesh",
            mesh=outputs["mesh"][0],
            video_path=video_paths.get("mesh"),
        )

    if "gaussian" in outputs:
        record_gaussian_artifact(
            recorder=recorder,
            step=step,
            logical_name=f"{variant}_gaussian",
            gaussian=outputs["gaussian"][0],
            video_path=video_paths.get("gaussian"),
        )

    if export_glb and "mesh" in outputs and "gaussian" in outputs:
        from trellis.utils import postprocessing_utils

        stem = recorder._artifact_stem(step, "output", f"{variant}_glb")
        glb_path = step.dir / f"{stem}.glb"
        preview_path = step.dir / f"{stem}.png"
        with torch.inference_mode(False):
            with torch.enable_grad():
                glb = postprocessing_utils.to_glb(
                    outputs["gaussian"][0],
                    outputs["mesh"][0],
                    simplify=0.95,
                    texture_size=1024,
                    verbose=False,
                )
        glb.export(glb_path)
        vertices = outputs["mesh"][0].vertices.detach().cpu().numpy()
        image_from_uint8(make_projection_strip(vertices.astype(np.float32))).save(preview_path)
        recorder._add_artifact(
            step=step,
            logical_name=f"{variant}_glb",
            role="output",
            path=glb_path,
            preview_path=preview_path,
            media_type="model/gltf-binary",
            viewer_hint="3d",
            metadata={"variant": variant, "textured": True},
            download_path=glb_path,
            video_path=video_paths.get("mesh"),
        )


def prepare_source_assets(
    recorder: TraceRecorder,
    step: StepContext,
    pipeline,
    resolved_paths: dict,
    source_cond: dict,
    config: TraceConfig,
    cuda_device: torch.device,
):
    mask_coords, mask_meta = uniedit.load_mask_glb_coords(config.mask_glb, device=cuda_device)
    coords_src = rf_utils.ply_to_coords(resolved_paths["voxels_path"], cuda_device)
    slat_src = uniedit.feats_to_slat_on_device(pipeline, resolved_paths["features_path"], cuda_device)

    recorder.record_coords(step, "coords_source", coords_src)
    recorder.record_sparse_tensor(step, "slat_source", slat_src)
    if mask_coords is not None:
        mask_colors = np.tile(np.array([[1.0, 0.4, 0.1]], dtype=np.float32), (mask_coords.shape[0], 1))
        recorder.record_coords(step, "mask_coords", mask_coords, colors=mask_colors, extra_metadata=mask_meta)
    recorder.record_json(step, "mask_glb_meta", mask_meta)
    recorder.record_json(
        step,
        "resolved_assets",
        {
            "asset_dir": str(resolved_paths["asset_dir"]),
            "voxels_path": str(resolved_paths["voxels_path"]),
            "features_path": str(resolved_paths["features_path"]),
            "source_image_path": str(resolved_paths["source_image_path"]),
            "edit_image_path": str(resolved_paths["edit_image_path"]),
            "mask_image_path": str(resolved_paths["mask_image_path"]) if resolved_paths["mask_image_path"] else None,
            "source_cond_shape": list(source_cond["cond"].shape),
        },
    )
    return coords_src, slat_src, mask_coords, mask_meta


def build_variant_selector_colors(selector: torch.Tensor) -> np.ndarray:
    active = selector.detach().cpu().numpy().reshape(-1) > 0.5
    colors = np.zeros((active.shape[0], 3), dtype=np.float32)
    colors[active] = np.array([0.1, 0.7, 1.0], dtype=np.float32)
    colors[~active] = np.array([1.0, 0.45, 0.05], dtype=np.float32)
    return colors


def run_uniedit_trace(config: TraceConfig, on_event: Optional[EventCallback] = None) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("UniEdit trace visualization currently requires a CUDA-capable environment.")

    if config.attn_backend:
        os.environ["ATTN_BACKEND"] = config.attn_backend

    if config.cfg_interval_start > config.cfg_interval_end:
        raise RuntimeError(
            f"cfg_interval_start must be <= cfg_interval_end, got {config.cfg_interval_start} > {config.cfg_interval_end}"
        )

    resolved_paths = resolve_trace_paths(config)
    case_name = determine_case_name(config, resolved_paths)
    out_dir = determine_output_dir(config, case_name)
    cfg_interval = (float(config.cfg_interval_start), float(config.cfg_interval_end))
    run_config = {
        **vars(config),
        "resolved_paths": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in resolved_paths.items()
        },
        "cfg_interval": list(cfg_interval),
        "output_dir": str(out_dir),
        "trace_method_name": TRACE_METHOD_NAME,
    }

    recorder = TraceRecorder(
        out_dir=out_dir,
        method_name=TRACE_METHOD_NAME,
        case_name=case_name,
        run_config=run_config,
        on_event=on_event,
    )
    recorder.log("Loading pipeline and resolving inputs.")

    pipeline = get_pipeline(config.model)
    cuda_device = torch.device("cuda")

    source_image = Image.open(resolved_paths["source_image_path"])
    edit_image = Image.open(resolved_paths["edit_image_path"])
    mask_image = Image.open(resolved_paths["mask_image_path"]) if resolved_paths["mask_image_path"] is not None else None

    input_step = recorder.begin_step("inputs", "Inputs And Preprocess")
    recorder.record_image(input_step, "source_input", source_image.convert("RGBA"), role="input")
    recorder.record_image(input_step, "edit_input", edit_image.convert("RGBA"), role="input")
    if mask_image is not None:
        recorder.record_image(input_step, "mask_input", mask_image.convert("L"), role="input")

    prepared = rf_utils.prepare_inputs_with_optional_auto_mask(
        pipeline=pipeline,
        source_image=source_image,
        edit_image=edit_image,
        mask_image=mask_image,
        preprocess=bool(config.preprocess),
        mask_threshold=int(config.mask_threshold),
        auto_mask_threshold=int(config.auto_mask_threshold),
        auto_mask_max_filter=int(config.auto_mask_max_filter),
    )
    recorder.record_json(input_step, "input_preprocess_meta", prepared.meta)
    recorder.record_image(input_step, "source_preprocessed", prepared.source)
    recorder.record_image(input_step, "edit_preprocessed", prepared.edit)
    recorder.record_image(input_step, "mask_preprocessed", prepared.mask)

    recorder.log("Encoding source and edit image conditions.")
    cond_step = recorder.begin_step("conditioning", "Image Conditioning")
    source_cond = uniedit.encode_cond_on_device(pipeline, [prepared.source], cuda_device)
    edit_cond = uniedit.encode_cond_on_device(pipeline, [prepared.edit], cuda_device)
    uniedit.move_models(pipeline, ["image_cond_model"], torch.device("cpu"))
    save_condition_tensor_bundle(recorder, cond_step, "source", source_cond)
    save_condition_tensor_bundle(recorder, cond_step, "edit", edit_cond)

    recorder.log("Loading source 3D assets and optional 3D mask.")
    assets_step = recorder.begin_step("source_assets", "Source Assets")
    coords_src, slat_src, mask_coords, mask_meta = prepare_source_assets(
        recorder=recorder,
        step=assets_step,
        pipeline=pipeline,
        resolved_paths=resolved_paths,
        source_cond=source_cond,
        config=config,
        cuda_device=cuda_device,
    )

    ss_inverse_params, ss_forward_params = rf_utils.resolve_stage_sampling_params(
        pipeline.sparse_structure_sampler_params,
        config.ss_steps,
        config.ss_cfg,
        config.ss_inverse_cfg,
    )
    slat_inverse_params, slat_forward_params = rf_utils.resolve_stage_sampling_params(
        pipeline.slat_sampler_params,
        config.slat_steps,
        config.slat_cfg,
        config.slat_inverse_cfg,
    )
    params_step = recorder.begin_step("sampling_params", "Sampling Params")
    recorder.record_json(
        params_step,
        "stage_sampling_params",
        {
            "sparse_structure_inverse_params": ss_inverse_params,
            "sparse_structure_forward_params": ss_forward_params,
            "slat_inverse_params": slat_inverse_params,
            "slat_forward_params": slat_forward_params,
            "cfg_interval": list(cfg_interval),
            "ss_omega": float(config.ss_omega),
            "slat_omega": float(config.slat_omega),
        },
    )

    torch.manual_seed(int(config.seed))
    verbose = not bool(config.quiet)

    with torch.no_grad():
        recorder.log("Running sparse-structure RF inversion.")
        ss_inversion_step = recorder.begin_step("ss_inversion", "Stage 1 Sparse Inversion")
        uniedit.move_models(pipeline, ["sparse_structure_encoder", "sparse_structure_flow_model"], cuda_device)
        voxel_src = rf_utils.coords_to_voxel(coords_src, cuda_device)
        z_src = pipeline.models["sparse_structure_encoder"](voxel_src)
        recorder.record_dense_tensor(ss_inversion_step, "sparse_structure_source_latent", z_src)
        ss_inv_sampler = TracedSecondOrderRFSampler(
            trace_fn=lambda payload: trace_second_order_events(recorder, ss_inversion_step, payload, "ss_terminal_noise_step"),
            trace_every=config.trace_every,
        )
        ss_terminal_noise = ss_inv_sampler.sample(
            model=pipeline.models["sparse_structure_flow_model"],
            sample=z_src,
            cond_dict=source_cond,
            steps=ss_inverse_params["steps"],
            rescale_t=ss_inverse_params["rescale_t"],
            cfg_strength=ss_inverse_params["cfg_strength"],
            cfg_interval=cfg_interval,
            inverse=True,
            verbose=verbose,
            trace_stage="ss_inversion",
        )
        recorder.record_dense_tensor(ss_inversion_step, "ss_terminal_noise_final", ss_terminal_noise)
        uniedit.move_models(pipeline, ["sparse_structure_encoder"], torch.device("cpu"))

        recorder.log("Running SLat RF inversion.")
        slat_inversion_step = recorder.begin_step("slat_inversion", "Stage 2 SLat Inversion")
        recorder.record_sparse_tensor(slat_inversion_step, "slat_source_encoded", slat_src)
        mean, std = rf_utils.get_slat_norm_tensors(pipeline, slat_src.device, slat_src.feats.dtype)
        slat_normalized = (slat_src - mean) / std
        recorder.record_sparse_tensor(slat_inversion_step, "slat_source_normalized", slat_normalized)
        uniedit.move_models(pipeline, ["slat_flow_model"], cuda_device)
        slat_inv_sampler = TracedSecondOrderRFSampler(
            trace_fn=lambda payload: trace_second_order_events(recorder, slat_inversion_step, payload, "slat_terminal_noise_step"),
            trace_every=config.trace_every,
        )
        slat_terminal_noise = slat_inv_sampler.sample(
            model=pipeline.models["slat_flow_model"],
            sample=slat_normalized,
            cond_dict=source_cond,
            steps=slat_inverse_params["steps"],
            rescale_t=slat_inverse_params["rescale_t"],
            cfg_strength=slat_inverse_params["cfg_strength"],
            cfg_interval=cfg_interval,
            inverse=True,
            verbose=verbose,
            trace_stage="slat_inversion",
        )
        recorder.record_sparse_tensor(slat_inversion_step, "slat_terminal_noise_final", slat_terminal_noise)

        recorder.log("Running UniEdit sparse-structure denoising.")
        ss_uniedit_step = recorder.begin_step("ss_uniedit", "Stage 1 UniEdit Denoise")
        uniedit.move_models(pipeline, ["sparse_structure_decoder"], cuda_device)
        ss_uniedit_sampler = TracedUniEditRFSampler(
            trace_fn=lambda payload: trace_uniedit_events(recorder, ss_uniedit_step, payload, "ss_denoise_step"),
            trace_every=config.trace_every,
        )
        z_tgt = ss_uniedit_sampler.sample(
            model=pipeline.models["sparse_structure_flow_model"],
            sample=ss_terminal_noise,
            source_cond=source_cond["cond"],
            target_cond=edit_cond["cond"],
            neg_cond=edit_cond["neg_cond"],
            steps=ss_forward_params["steps"],
            rescale_t=ss_forward_params["rescale_t"],
            cfg_strength=ss_forward_params["cfg_strength"],
            cfg_interval=cfg_interval,
            omega=float(config.ss_omega),
            selector=None,
            mode="full_uniedit",
            verbose=verbose,
            trace_stage="ss_uniedit",
        )
        recorder.record_dense_tensor(ss_uniedit_step, "ss_denoise_final_latent", z_tgt)

        voxel_stage1 = pipeline.models["sparse_structure_decoder"](z_tgt)
        recorder.record_dense_tensor(ss_uniedit_step, "stage1_decoded_voxel_logits", voxel_stage1)
        coords_stage1_raw = torch.argwhere(voxel_stage1 > 0)[:, [0, 2, 3, 4]].int()
        if coords_stage1_raw.shape[0] == 0:
            raise RuntimeError("Stage 1 UniEdit sparse-structure denoising produced an empty target structure.")
        recorder.record_coords(ss_uniedit_step, "coords_stage1_raw", coords_stage1_raw)

        coords_stage1_masked, stage1_meta = uniedit.compose_stage1_coords(
            coords_source=coords_src,
            coords_stage1_raw=coords_stage1_raw,
            mask_coords=mask_coords,
        )
        recorder.record_coords(
            ss_uniedit_step,
            "coords_stage1_masked",
            coords_stage1_masked,
            extra_metadata=stage1_meta,
        )
        recorder.record_json(
            ss_uniedit_step,
            "stage1_structure_stats",
            {
                **uniedit.summarize_coord_transition(
                    coords_source=coords_src,
                    coords_stage1_raw=coords_stage1_raw,
                    coords_stage1_masked=coords_stage1_masked,
                    mask_coords=mask_coords,
                ),
                **stage1_meta,
                "mask_glb_meta": mask_meta,
            },
        )

        recorder.log("Projecting SLat terminal noise onto Stage 1 coordinates.")
        projection_step = recorder.begin_step("slat_projection", "Project Terminal Noise To Stage 1")
        projected_slat_noise = rf_utils.project_sparse_terminal_noise(
            source_noise=slat_terminal_noise,
            target_coords=coords_stage1_masked.to(device=cuda_device),
            device=cuda_device,
        )
        recorder.record_sparse_tensor(projection_step, "projected_slat_noise", projected_slat_noise)
        stage2_selector = uniedit.build_stage2_selector(coords_stage1_masked, coords_src)
        selector_colors = build_variant_selector_colors(stage2_selector)
        recorder.record_coords(
            projection_step,
            "stage2_selector_overlap_vs_new",
            coords_stage1_masked,
            colors=selector_colors,
            extra_metadata={
                "overlap_voxel_count": int(stage2_selector.sum().item()),
                "new_voxel_count": int(stage2_selector.shape[0] - stage2_selector.sum().item()),
            },
        )

        uniedit.move_models(
            pipeline,
            ["sparse_structure_flow_model", "sparse_structure_decoder", "slat_encoder"],
            torch.device("cpu"),
        )

        for variant in uniedit.STAGE2_VARIANTS:
            if variant == "preserve_uniedit":
                selector = stage2_selector
                mode = "preserve_overlap"
            else:
                selector = None
                mode = "target_only"

            recorder.log(f"Running Stage 2 variant: {variant}.")
            variant_step = recorder.begin_step(f"slat_{variant}", f"Stage 2 {variant}")
            slat_sampler = TracedUniEditRFSampler(
                trace_fn=lambda payload, _step=variant_step, _name=variant: trace_uniedit_events(
                    recorder, _step, payload, f"{_name}_slat_step"
                ),
                trace_every=config.trace_every,
            )
            slat_tgt = slat_sampler.sample(
                model=pipeline.models["slat_flow_model"],
                sample=projected_slat_noise,
                source_cond=source_cond["cond"],
                target_cond=edit_cond["cond"],
                neg_cond=edit_cond["neg_cond"],
                steps=slat_forward_params["steps"],
                rescale_t=slat_forward_params["rescale_t"],
                cfg_strength=slat_forward_params["cfg_strength"],
                cfg_interval=cfg_interval,
                omega=float(config.slat_omega),
                selector=selector,
                mode=mode,
                verbose=verbose,
                trace_stage=f"slat_{variant}",
            )
            recorder.record_sparse_tensor(
                variant_step,
                f"{variant}_slat_final",
                slat_tgt,
                extra_metadata={
                    "variant": variant,
                    "mode": mode,
                    "selector_enabled": selector is not None,
                },
            )

            uniedit.move_models(pipeline, ["slat_decoder_mesh", "slat_decoder_gs"], cuda_device)
            outputs = pipeline.decode_slat(slat_tgt, ["mesh", "gaussian"])
            record_variant_outputs(
                recorder=recorder,
                step=variant_step,
                pipeline=pipeline,
                variant=variant,
                outputs=outputs,
                render_videos=bool(config.render_videos),
                export_glb=bool(config.export_glb),
            )
            recorder.record_json(
                variant_step,
                f"{variant}_summary",
                {
                    "variant": variant,
                    "mode": mode,
                    "selector_enabled": selector is not None,
                    "selector_overlap_voxel_count": int(stage2_selector.sum().item()),
                    "selector_new_voxel_count": int(stage2_selector.shape[0] - stage2_selector.sum().item()),
                    "slat_omega": float(config.slat_omega),
                    "slat_forward_params": slat_forward_params,
                },
            )
            uniedit.move_models(pipeline, ["slat_decoder_mesh", "slat_decoder_gs"], torch.device("cpu"))
            del outputs
            del slat_tgt
            release_cuda_memory()

    del voxel_src
    del z_src
    del source_cond
    del edit_cond
    del slat_src
    del ss_terminal_noise
    del slat_terminal_noise
    del projected_slat_noise
    release_cuda_memory()

    recorder.log(f"Trace complete. Artifacts written to {out_dir}.")
    manifest = recorder.snapshot()
    manifest["trace_manifest_path"] = str(recorder.manifest_path)
    return manifest


def load_manifest(manifest_or_dir: str | Path) -> dict:
    path = Path(manifest_or_dir).expanduser().resolve()
    if path.is_dir():
        path = path / "trace_manifest.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def artifact_choices(manifest: dict) -> List[Tuple[str, str]]:
    return [(artifact["label"], artifact["id"]) for artifact in manifest.get("artifacts", [])]


def resolve_artifact(manifest: dict, artifact_id: Optional[str]) -> Optional[dict]:
    artifacts = manifest.get("artifacts", [])
    if not artifacts:
        return None
    if artifact_id is None:
        artifact = artifacts[-1]
    else:
        artifact = next((item for item in artifacts if item["id"] == artifact_id), artifacts[-1])

    out_dir = Path(manifest["output_dir"])

    def absolute(rel_path: Optional[str]) -> Optional[str]:
        if not rel_path:
            return None
        return str((out_dir / rel_path).resolve())

    result = copy.deepcopy(artifact)
    result["absolute_path"] = absolute(artifact.get("path"))
    result["absolute_preview_path"] = absolute(artifact.get("preview_path"))
    result["absolute_video_path"] = absolute(artifact.get("video_path"))
    result["absolute_download_path"] = absolute(artifact.get("download_path") or artifact.get("path"))
    return result


def preview_gallery_items(manifest: dict, limit: int = 18) -> List[Tuple[str, str]]:
    out_dir = Path(manifest["output_dir"])
    items: List[Tuple[str, str]] = []
    for artifact in manifest.get("artifacts", []):
        preview_path = artifact.get("preview_path")
        if not preview_path:
            continue
        abs_preview = out_dir / preview_path
        if abs_preview.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        items.append((str(abs_preview.resolve()), artifact["label"]))
    return items[-limit:]
