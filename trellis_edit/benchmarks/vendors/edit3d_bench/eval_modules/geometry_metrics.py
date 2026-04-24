"""
Geometry Evaluation Module
Contains 3D geometry metrics such as Chamfer Distance
"""

import logging
from typing import List

import numpy as np
import open3d as o3d
import torch
import trimesh
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from ..utils import load_trimesh, sample_out_points

logger = logging.getLogger(__name__)


class ModelTripletDataset(Dataset):
    """3D model triplet dataset."""

    def __init__(
        self,
        model_triplets,
        num_points=20480,
        voxel_size=0.01,
        ignore_mask=False,
    ):
        self.model_triplets = model_triplets
        self.num_points = num_points
        self.voxel_size = voxel_size
        self.ignore_mask = ignore_mask

    def __len__(self):
        return len(self.model_triplets)

    def __getitem__(self, idx):
        triplet = self.model_triplets[idx]
        if len(triplet) == 4:
            gt_path, pred_path, mask_path, sample_id = triplet
        else:
            gt_path, pred_path, mask_path = triplet
            sample_id = f"sample_{idx}"

        try:
            gt_outside_pts = self._sample_and_preprocess_out_points(
                gt_path, mask_path, self.num_points
            )
            pred_outside_pts = self._sample_and_preprocess_out_points(
                pred_path, mask_path, self.num_points
            )

            if gt_outside_pts is None or pred_outside_pts is None:
                return None

            gt_tensor = torch.from_numpy(gt_outside_pts).float()
            pred_tensor = torch.from_numpy(pred_outside_pts).float()

            return {
                "id": sample_id,
                "gt_path": gt_path,
                "pred_path": pred_path,
                "mask_path": mask_path,
                "gt_points": gt_tensor,
                "pred_points": pred_tensor,
            }

        except Exception as exc:
            logger.warning("Failed to process model pair %s: %s", gt_path, exc)
            return None

    def _preprocess_point_cloud(
        self, points: np.ndarray, voxel_size: float = 0.01
    ) -> np.ndarray:
        """Preprocess point cloud: voxel downsampling."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        return np.asarray(downpcd.points, dtype=np.float32)

    def _sample_and_preprocess_out_points(
        self,
        model_path: str,
        mask_path: str,
        num_points: int = 20480,
        voxel_size: float = 0.01,
    ) -> np.ndarray:
        """Sample and preprocess outside-mask point cloud."""
        try:
            if not self.ignore_mask:
                pts = sample_out_points(model_path, mask_path, num_points)
            else:
                trimesh_model = load_trimesh(model_path)
                pts = trimesh.sample.sample_surface(trimesh_model, num_points)[0]
            return self._preprocess_point_cloud(pts, voxel_size)
        except Exception as exc:
            logger.warning("Point cloud sampling failed %s: %s", model_path, exc)
            return None


class GeometryMetricsEvaluator:
    """Geometry evaluator."""

    _supported_metrics = ["chamfer"]

    def __init__(
        self,
        device: str = "cuda:0",
        batch_size: int = 8,
        num_workers: int = 4,
        ignore_mask: bool = False,
        num_points: int = 20480,
        voxel_size: float = 0.01,
    ):
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ignore_mask = ignore_mask
        self.num_points = max(int(num_points), 1)
        self.voxel_size = float(voxel_size)

    def _nearest_dist_torch(
        self, pts0: torch.Tensor, pts1: torch.Tensor, batch_size: int = 4096
    ) -> torch.Tensor:
        """Calculate nearest distance from each point in pts0 to pts1 using PyTorch."""
        pn0 = pts0.shape[0]
        dists = []

        for i in range(0, pn0, batch_size):
            end_idx = min(i + batch_size, pn0)
            dist = torch.norm(pts0[i:end_idx, None, :] - pts1[None, :, :], dim=-1)
            dists.append(torch.min(dist, 1)[0])

        return torch.cat(dists, 0)

    def _chamfer_distance_torch(
        self, pts0: torch.Tensor, pts1: torch.Tensor, batch_size: int = 4096
    ) -> tuple:
        """Calculate Chamfer Distance using PyTorch."""
        if pts0.shape[0] == 0 or pts1.shape[0] == 0:
            raise ValueError("Point cloud is empty.")

        dist0 = self._nearest_dist_torch(pts0, pts1, batch_size)
        dist1 = self._nearest_dist_torch(pts1, pts0, batch_size)
        chamfer = (torch.mean(dist0) + torch.mean(dist1)) / 2
        return chamfer, dist0, dist1

    def _collate_fn(self, batch):
        """Custom collate function to handle None values."""
        valid_batch = [item for item in batch if item is not None]
        if not valid_batch:
            return None

        collated = {}
        for key in valid_batch[0].keys():
            if key in ["id", "gt_path", "pred_path", "mask_path"]:
                collated[key] = [item[key] for item in valid_batch]
            elif key in ["gt_points", "pred_points"]:
                collated[key] = [item[key] for item in valid_batch]
        return collated

    def _compute_chamfer_with_dataloader(self, model_triplets: list) -> dict:
        """Calculate Chamfer Distance using DataLoader."""
        logger.info(
            "Computing Chamfer Distance (batch_size=%s, num_workers=%s)...",
            self.batch_size,
            self.num_workers,
        )

        dataset = ModelTripletDataset(
            model_triplets,
            num_points=self.num_points,
            voxel_size=self.voxel_size,
            ignore_mask=self.ignore_mask,
        )
        dataloader = TorchDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=False,
        )

        scores = []
        score_ids = []
        processed = 0
        skipped = 0

        for batch in tqdm(dataloader, desc="Computing Chamfer Distance"):
            if batch is None:
                skipped += self.batch_size
                continue

            batch_size = len(batch["gt_path"])
            for i in range(batch_size):
                try:
                    chamfer, dist0, dist1 = self._chamfer_distance_torch(
                        batch["gt_points"][i].to(self.device),
                        batch["pred_points"][i].to(self.device),
                        batch_size=4096,
                    )
                    scores.append(chamfer.cpu().item())
                    score_ids.append(batch["id"][i])
                    processed += 1
                except Exception as exc:
                    logger.warning(
                        "Failed to process model pair %s: %s",
                        batch["gt_path"][i],
                        exc,
                    )
                    skipped += 1
                    continue

        return scores, score_ids, processed, skipped

    def compute_chamfer_distance(
        self,
        gt_models: list,
        pred_models: list,
        masks: list,
        model_ids: list | None = None,
    ) -> dict:
        """Calculate Chamfer Distance metric."""
        if model_ids:
            model_triplets = list(zip(gt_models, pred_models, masks, model_ids))
        else:
            model_triplets = list(zip(gt_models, pred_models, masks))

        scores, score_ids, processed, skipped = self._compute_chamfer_with_dataloader(
            model_triplets
        )

        logger.info(
            "Successfully processed %s model pairs, skipped %s",
            processed,
            skipped,
        )

        if len(scores) == 0:
            return {"mean": None, "std": None, "count": 0}

        scores = np.array(scores)
        score_ids = np.array(score_ids, dtype=object)
        valid_mask = np.isfinite(scores)
        valid_scores = scores[valid_mask]
        valid_score_ids = score_ids[valid_mask]

        if len(valid_scores) == 0:
            return {"mean": None, "std": None, "count": 0}

        return {
            "mean": float(np.mean(valid_scores)),
            "std": float(np.std(valid_scores)),
            "count": len(valid_scores),
            "all_scores": valid_scores.tolist(),
            "all_ids": [str(score_id) for score_id in valid_score_ids.tolist()],
        }

    def compute(
        self,
        gt_models: list,
        pred_models: list,
        masks: list,
        metrics_to_compute: List[str] = _supported_metrics,
        model_ids: list | None = None,
    ) -> dict:
        """Calculate Chamfer Distance metric."""
        result = {}
        metrics_to_compute = list(
            set(metrics_to_compute) & set(self._supported_metrics)
        )
        logger.info(
            "Computing %s metrics (batch_size=%s, num_workers=%s)...",
            metrics_to_compute,
            self.batch_size,
            self.num_workers,
        )
        for metric in metrics_to_compute:
            if metric == "chamfer":
                result[metric] = self.compute_chamfer_distance(
                    gt_models, pred_models, masks, model_ids=model_ids
                )
        return result
