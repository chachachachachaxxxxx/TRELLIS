from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import trimesh
from pysdf import SDF
from scipy.spatial import cKDTree
from transformers import CLIPModel, CLIPProcessor

from trellis_edit.metrics.floaters import (
    DEFAULT_FLOATER_AREA_RATIO_THRESHOLD,
    DEFAULT_MERGE_PRECISION,
    DEFAULT_MICRO_FLOATER_AREA_RATIO_THRESHOLD,
    analyze_mesh_floaters,
)


GEOMETRY_EDIT_METRICS = (
    "clipi",
    "clipi_mv",
    "clipi_n",
    "clipi_mv_n",
    "uni3d",
    "cd",
    "floaters",
)

_CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
_UNI3D_METRIC_ROOT = Path("/home/wangxinxing/code/evaluation_0121")


def _resolve_local_snapshot(model_name: str) -> str | None:
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        return None

    snapshot_root = (
        Path(hf_home)
        / "hub"
        / f"models--{model_name.replace('/', '--')}"
        / "snapshots"
    )
    if not snapshot_root.exists():
        return None

    for snapshot in sorted(snapshot_root.iterdir(), reverse=True):
        if snapshot.is_dir() and (snapshot / "config.json").exists():
            return str(snapshot)
    return None


class ClipImageSimilarityEvaluator:
    def __init__(self, *, device: str = "cuda:0"):
        self.device = device
        self.model, self.processor = self._load_components()

    def _load_components(self):
        local_snapshot = _resolve_local_snapshot(_CLIP_MODEL_NAME)
        try:
            if local_snapshot is not None:
                model = CLIPModel.from_pretrained(
                    local_snapshot,
                    local_files_only=True,
                    use_safetensors=False,
                ).to(self.device).eval()
                processor = CLIPProcessor.from_pretrained(
                    local_snapshot,
                    local_files_only=True,
                    use_fast=False,
                )
            else:
                raise OSError("Local snapshot not found")
        except OSError:
            model = CLIPModel.from_pretrained(
                _CLIP_MODEL_NAME,
                use_safetensors=False,
            ).to(self.device).eval()
            processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_NAME, use_fast=False)
        return model, processor

    def _unwrap_embedding(self, features) -> torch.Tensor:
        if hasattr(features, "pooler_output"):
            return features.pooler_output
        if isinstance(features, tuple):
            if len(features) == 0:
                raise ValueError("Empty CLIP feature tuple received.")
            return features[-1]
        return features

    @torch.no_grad()
    def encode_images(self, images: Sequence) -> torch.Tensor:
        inputs = self.processor(images=list(images), return_tensors="pt")
        inputs = {
            key: value.to(self.device, non_blocking=True)
            for key, value in inputs.items()
        }
        features = self.model.get_image_features(**inputs)
        features = self._unwrap_embedding(features)
        return features / features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def similarity(self, image_a, image_b) -> float:
        features = self.encode_images([image_a, image_b])
        return float(torch.sum(features[0] * features[1]).item())


def _ensure_uni3d_import_path() -> None:
    root = str(_UNI3D_METRIC_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


class Uni3DPointCloudEvaluator:
    def __init__(self, *, device: str = "cuda:0", model_name: str = "uni3d-g"):
        _ensure_uni3d_import_path()
        from uni3d_metric.metric import Uni3DMetric

        self.device = device
        self.metric = Uni3DMetric.from_pretrained(
            model_name=model_name,
            device=device,
        )

    @torch.no_grad()
    def similarity(self, point_cloud_a: np.ndarray, point_cloud_b: np.ndarray) -> float:
        batch = torch.stack(
            [
                torch.from_numpy(point_cloud_a).float(),
                torch.from_numpy(point_cloud_b).float(),
            ],
            dim=0,
        ).to(self.device)
        embeddings = self.metric.encode_pc(batch)
        return float(torch.sum(embeddings[0] * embeddings[1]).item())


class FloaterMeshEvaluator:
    def __init__(
        self,
        *,
        merge_precision: float = DEFAULT_MERGE_PRECISION,
        floater_area_ratio_threshold: float = DEFAULT_FLOATER_AREA_RATIO_THRESHOLD,
        micro_floater_area_ratio_threshold: float = DEFAULT_MICRO_FLOATER_AREA_RATIO_THRESHOLD,
    ):
        self.merge_precision = float(merge_precision)
        self.floater_area_ratio_threshold = float(floater_area_ratio_threshold)
        self.micro_floater_area_ratio_threshold = float(micro_floater_area_ratio_threshold)

    def analyze(self, mesh_path: str | Path) -> dict[str, int | float]:
        return analyze_mesh_floaters(
            Path(mesh_path),
            merge_precision=self.merge_precision,
            floater_area_ratio_threshold=self.floater_area_ratio_threshold,
            micro_floater_area_ratio_threshold=self.micro_floater_area_ratio_threshold,
        )

    def score(self, mesh_path: str | Path) -> float:
        stats = self.analyze(mesh_path)
        return float(stats["floater_count_lt_0p1pct"])


def load_trimesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes: list[trimesh.Trimesh] = []
        for node_name in loaded.graph.nodes_geometry:
            transform, geometry_name = loaded.graph[node_name]
            geometry = loaded.geometry.get(geometry_name)
            if not isinstance(geometry, trimesh.Trimesh):
                continue
            mesh = geometry.copy()
            mesh.apply_transform(transform)
            meshes.append(mesh)
        if not meshes:
            raise RuntimeError(f"No mesh geometry found in {path}")
        if len(meshes) == 1:
            return meshes[0].copy()
        return trimesh.util.concatenate(tuple(mesh.copy() for mesh in meshes))
    if isinstance(loaded, trimesh.Trimesh):
        return loaded.copy()
    raise RuntimeError(f"Unsupported mesh payload in {path}: {type(loaded)!r}")


def source_normalization(mesh: trimesh.Trimesh) -> tuple[np.ndarray, float]:
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    center = bounds.mean(axis=0)
    extent = float(np.max(bounds[1] - bounds[0]))
    if not np.isfinite(extent) or extent <= 0:
        extent = 1.0
    return center.astype(np.float64), extent


def build_mask_sdf(mask_mesh: trimesh.Trimesh) -> SDF | None:
    if mask_mesh.vertices.size == 0 or mask_mesh.faces.size == 0:
        return None
    vertices = np.asarray(mask_mesh.vertices, dtype=np.float64)
    faces = np.asarray(mask_mesh.faces, dtype=np.int32)
    if vertices.size == 0 or faces.size == 0:
        return None
    return SDF(vertices, faces)


def _sample_surface_points(
    mesh: trimesh.Trimesh,
    *,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        points, _ = trimesh.sample.sample_surface(mesh, sample_count)
    finally:
        np.random.set_state(state)
    return np.asarray(points, dtype=np.float64)


def _normalize_xyz(points_xyz: np.ndarray, *, center: np.ndarray, extent: float) -> np.ndarray:
    return ((points_xyz - center[None, :]) / float(extent)).astype(np.float32)


def _constant_rgb(points_xyz: np.ndarray, value: float = 0.4) -> np.ndarray:
    return np.full(points_xyz.shape, fill_value=value, dtype=np.float32)


def _resample_points(points_xyz: np.ndarray, *, num_points: int, seed: int) -> np.ndarray:
    if points_xyz.shape[0] == 0:
        raise ValueError("Cannot resample an empty point cloud.")
    rng = np.random.default_rng(seed)
    replace = points_xyz.shape[0] < num_points
    indices = rng.choice(points_xyz.shape[0], size=num_points, replace=replace)
    return points_xyz[indices]


def sample_outside_mask_point_cloud(
    mesh: trimesh.Trimesh,
    *,
    mask_sdf: SDF | None,
    center: np.ndarray,
    extent: float,
    num_points: int = 10000,
    initial_sample_count: int = 40000,
    max_rounds: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray | None, dict[str, float | int | None]]:
    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        return None, {"sampled": 0, "outside": 0, "mask_mode": "sdf"}

    if mask_sdf is None:
        points_xyz = _sample_surface_points(mesh, sample_count=num_points, seed=seed)
        normalized_xyz = _normalize_xyz(points_xyz, center=center, extent=extent)
        return np.concatenate((normalized_xyz, _constant_rgb(normalized_xyz)), axis=1), {
            "sampled": int(points_xyz.shape[0]),
            "outside": int(points_xyz.shape[0]),
            "mask_mode": "none",
        }

    outside_chunks: list[np.ndarray] = []
    sampled_total = 0
    outside_total = 0
    sample_count = max(initial_sample_count, num_points)

    for round_idx in range(max_rounds):
        points_xyz = _sample_surface_points(
            mesh,
            sample_count=sample_count,
            seed=seed + round_idx,
        )
        sampled_total += int(points_xyz.shape[0])
        if points_xyz.shape[0] == 0:
            sample_count *= 2
            continue
        distances = mask_sdf(points_xyz)
        inside_mask = np.asarray(distances, dtype=np.float64) > 0
        outside_xyz = points_xyz[~inside_mask]
        outside_total += int(outside_xyz.shape[0])
        if outside_xyz.shape[0] > 0:
            outside_chunks.append(outside_xyz)
        if sum(chunk.shape[0] for chunk in outside_chunks) >= num_points:
            break
        sample_count *= 2

    if not outside_chunks:
        return None, {
            "sampled": int(sampled_total),
            "outside": int(outside_total),
            "mask_mode": "sdf",
        }

    all_outside_xyz = np.concatenate(outside_chunks, axis=0)
    sampled_xyz = _resample_points(all_outside_xyz, num_points=num_points, seed=seed + 97)
    normalized_xyz = _normalize_xyz(sampled_xyz, center=center, extent=extent)
    return np.concatenate((normalized_xyz, _constant_rgb(normalized_xyz)), axis=1), {
        "sampled": int(sampled_total),
        "outside": int(outside_total),
        "mask_mode": "sdf",
    }


def symmetric_chamfer_distance(point_cloud_a: np.ndarray, point_cloud_b: np.ndarray) -> float:
    if point_cloud_a.shape[0] == 0 or point_cloud_b.shape[0] == 0:
        raise ValueError("Chamfer distance requires non-empty point clouds.")
    tree_b = cKDTree(point_cloud_b)
    dist_a, _ = tree_b.query(point_cloud_a, k=1)
    tree_a = cKDTree(point_cloud_a)
    dist_b, _ = tree_a.query(point_cloud_b, k=1)
    return float(0.5 * (float(dist_a.mean()) + float(dist_b.mean())))
