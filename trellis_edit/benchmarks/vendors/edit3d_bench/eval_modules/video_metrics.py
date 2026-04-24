"""
Video Evaluation Module
Contains video metrics such as FVD
"""

import logging
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
import scipy.linalg
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from .models import InceptionI3d

logger = logging.getLogger(__name__)


class VideoDataset(Dataset):
    """Video dataset"""

    def __init__(
        self, video_paths, target_size=(224, 224), max_frames=None, frame_interval=1
    ):
        self.video_paths = video_paths
        self.target_size = target_size
        self.max_frames = max_frames
        self.frame_interval = frame_interval

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]

        try:
            # Load video
            cap = cv2.VideoCapture(video_path)
            frames = []
            frame_count = 0
            frame_index = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if self.max_frames and frame_index >= self.max_frames:
                    break

                # Take one frame every frame_interval frames
                if frame_count % self.frame_interval == 0:
                    # BGR -> RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(
                        frame, (self.target_size[1], self.target_size[0])
                    )
                    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
                    frames.append(frame_tensor)
                    frame_index += 1

                frame_count += 1

            cap.release()

            if frames:
                video_tensor = torch.stack(frames, dim=1)  # [C, T, H, W]
                return {"path": video_path, "video_tensor": video_tensor}
            else:
                return None

        except Exception as e:
            logger.warning(f"Failed to load video {video_path}: {e}")
            return None


class VideoMetricsEvaluator:
    """Video evaluator"""

    MODEL_URL = "https://raw.githubusercontent.com/piergiaj/pytorch-i3d/refs/heads/master/models/rgb_charades.pt"
    MODEL_PATH = Path(
        os.environ.get(
            "TRELLIS_EDIT_I3D_PATH",
            Path(__file__).resolve().parents[1] / "pytorch_i3d.pth",
        )
    )

    _supported_metrics = ["fvd"]

    def __init__(
        self,
        device: str = "cuda:0",
        batch_size: int = 8,
        num_workers: int = 4,
        **kwargs,
    ):
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Initialize I3D model
        self.i3d_model = InceptionI3d(157, in_channels=3)
        if not self.MODEL_PATH.exists():
            torch.hub.download_url_to_file(self.MODEL_URL, str(self.MODEL_PATH))
        self.i3d_model.load_state_dict(torch.load(str(self.MODEL_PATH)))
        self.i3d_model.to(device).eval()

    @torch.no_grad()
    def _extract_i3d_features(self, video_tensor: torch.Tensor) -> np.ndarray:
        """Extract I3D features"""
        features = self.i3d_model.extract_features(video_tensor.to(self.device))
        features = features.mean(
            dim=[2, 3, 4]
        )  # Spatiotemporal average pooling -> [B, C]
        return features.cpu().numpy()

    def _compute_frechet_distance(
        self, mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray
    ) -> float:
        """Calculate Fréchet distance"""
        try:
            # Calculate mean difference
            mu_diff = mu1 - mu2

            # Calculate geometric mean of covariance matrices
            covmean = scipy.linalg.sqrtm(sigma1 @ sigma2)

            # Check numerical stability
            if np.iscomplexobj(covmean):
                covmean = covmean.real

            # Calculate FVD
            fvd = mu_diff @ mu_diff + np.trace(sigma1 + sigma2 - 2 * covmean)
            return fvd

        except Exception as e:
            logger.warning(f"FVD calculation failed: {e}")
            # Use simplified distance calculation
            return np.sum((mu1 - mu2) ** 2) + np.trace(sigma1 + sigma2)

    def _collate_fn(self, batch):
        """Custom collate function to handle None values"""
        # Filter out None values
        valid_batch = [item for item in batch if item is not None]

        if not valid_batch:
            return None

        # Reorganize data
        collated = {}
        for key in valid_batch[0].keys():
            if key == "path":
                collated[key] = [item[key] for item in valid_batch]
            elif key == "video_tensor":
                collated[key] = torch.stack([item[key] for item in valid_batch], dim=0)

        return collated

    def _extract_video_features_with_dataloader(self, video_paths: list) -> list:
        """Extract video features using DataLoader"""
        dataset = VideoDataset(video_paths)
        dataloader = TorchDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=False,  # Video data is usually large, don't use pin_memory
        )

        features = []
        processed = 0

        for batch in tqdm(dataloader, desc="Extracting video features"):
            if batch is not None:
                try:
                    features.append(self._extract_i3d_features(batch["video_tensor"]))
                    processed += len(batch["video_tensor"])
                except Exception as e:
                    logger.warning(f"Failed to process video {batch['path']}: {e}")
                    continue

        return features, processed

    @torch.no_grad()
    def compute_fvd(self, gt_videos: list, pred_videos: list) -> float:
        """Calculate FVD metric"""

        # Batch process to extract real video features
        logger.info("Batch processing to extract real video features...")
        gt_features, processed_gt = self._extract_video_features_with_dataloader(
            gt_videos
        )

        # Batch process to extract generated video features
        logger.info("Batch processing to extract generated video features...")
        pred_features, processed_pred = self._extract_video_features_with_dataloader(
            pred_videos
        )

        logger.info(
            f"Successfully processed {processed_gt} real videos and {processed_pred} generated videos"
        )

        if len(gt_features) == 0 or len(pred_features) == 0:
            logger.warning("No valid features available, cannot calculate FVD")
            return None

        # Calculate statistics
        gt_features = np.concatenate(gt_features, axis=0)  # [N, feature_dim]
        pred_features = np.concatenate(pred_features, axis=0)  # [M, feature_dim]

        # Calculate mean and covariance
        mu1 = np.mean(gt_features, axis=0)
        sigma1 = np.cov(gt_features, rowvar=False)

        mu2 = np.mean(pred_features, axis=0)
        sigma2 = np.cov(pred_features, rowvar=False)

        # Calculate FVD
        fvd_score = self._compute_frechet_distance(mu1, sigma1, mu2, sigma2)

        return fvd_score

    def compute(
        self,
        gt_videos: list,
        pred_videos: list,
        metrics_to_compute: List[str] = _supported_metrics,
    ) -> float:
        """Calculate FVD metric"""
        results = {}
        metrics_to_compute = list(
            set(metrics_to_compute) & set(self._supported_metrics)
        )
        logger.info(
            f"Computing {metrics_to_compute} metrics (batch_size={self.batch_size}, num_workers={self.num_workers})..."
        )
        for metric in metrics_to_compute:
            if metric == "fvd":
                results[metric] = self.compute_fvd(gt_videos, pred_videos)
        return results
