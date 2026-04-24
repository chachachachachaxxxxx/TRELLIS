"""
Distribution Evaluation Module
Contains distribution metrics such as FID, DINO-I
"""

import logging
from typing import List

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from .mask_utils import (
    DEFAULT_BACKGROUND_RGB,
    apply_include_mask_to_image,
    build_include_mask,
    load_prepared_rgb_image,
)

logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    """Image dataset for whole-image distribution metrics."""

    def __init__(self, image_paths, image_size=(512, 512)):
        self.image_paths = image_paths
        self.image_size = image_size
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        try:
            img = Image.open(img_path)
            if img.mode != "RGB":
                background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                background.paste(img, (0, 0), img)
                img = background.convert("RGB")
            img = img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)

            img_tensor = self.to_tensor(img)
            img_uint8 = (img_tensor * 255).type(torch.uint8)
            return {"path": img_path, "img_uint8": img_uint8}

        except Exception as exc:
            logger.warning("Failed to load image %s: %s", img_path, exc)
            return None


class MaskedImagePairDataset(Dataset):
    """Image-pair dataset for buffered preserved-region FID."""

    def __init__(
        self,
        image_pairs,
        image_size=(512, 512),
        ignore_mask: bool = False,
        mask_dilation_px: int = 0,
        fill_rgb=DEFAULT_BACKGROUND_RGB,
    ):
        self.image_pairs = image_pairs
        self.image_size = image_size
        self.ignore_mask = ignore_mask
        self.mask_dilation_px = max(int(mask_dilation_px), 0)
        self.fill_rgb = fill_rgb
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        pair = self.image_pairs[idx]

        try:
            gt_image = load_prepared_rgb_image(pair.gt_path, self.image_size)
            pred_image = load_prepared_rgb_image(pair.pred_path, self.image_size)
            include_mask = build_include_mask(
                pair.mask_path,
                self.image_size,
                dilation_px=self.mask_dilation_px,
                ignore_mask=self.ignore_mask,
            )

            gt_image = apply_include_mask_to_image(
                gt_image, include_mask, fill_rgb=self.fill_rgb
            )
            pred_image = apply_include_mask_to_image(
                pred_image, include_mask, fill_rgb=self.fill_rgb
            )

            gt_tensor = self.to_tensor(gt_image)
            pred_tensor = self.to_tensor(pred_image)
            return {
                "id": pair.id,
                "gt_uint8": (gt_tensor * 255).type(torch.uint8),
                "pred_uint8": (pred_tensor * 255).type(torch.uint8),
            }
        except Exception as exc:
            logger.warning("Failed to load image pair %s: %s", pair.id, exc)
            return None


class DistributionMetricsEvaluator:
    """Distribution evaluator."""

    _supported_metrics = ["fid", "fid_buffered"]

    def __init__(
        self,
        device: str = "cuda:0",
        image_size: tuple = (512, 512),
        batch_size: int = 32,
        num_workers: int = 4,
        ignore_mask: bool = False,
        mask_dilation_px: int = 0,
    ):
        self.device = device
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ignore_mask = ignore_mask
        self.mask_dilation_px = max(int(mask_dilation_px), 0)

        self.fid = FrechetInceptionDistance(feature=2048).to(device)
        self.to_tensor = transforms.ToTensor()

    def _collate_fn(self, batch):
        """Custom collate function to handle None values."""
        valid_batch = [item for item in batch if item is not None]
        if not valid_batch:
            return None

        collated = {}
        for key in valid_batch[0].keys():
            if key in {"path", "id"}:
                collated[key] = [item[key] for item in valid_batch]
            else:
                collated[key] = torch.stack([item[key] for item in valid_batch])
        return collated

    def _compute_fid_with_dataloader(self, gt_images: list, pred_images: list) -> float:
        """Calculate legacy whole-image FID using DataLoader."""
        self.fid.reset()

        logger.info("Processing real images...")
        gt_dataset = ImageDataset(gt_images, self.image_size)
        gt_dataloader = TorchDataLoader(
            gt_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True if self.device.startswith("cuda") else False,
        )

        processed_gt = 0
        for batch in tqdm(gt_dataloader, desc="Real images"):
            if batch is None:
                continue
            self.fid.update(batch["img_uint8"].to(self.device), real=True)
            processed_gt += len(batch["path"])

        logger.info("Processing generated images...")
        pred_dataset = ImageDataset(pred_images, self.image_size)
        pred_dataloader = TorchDataLoader(
            pred_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True if self.device.startswith("cuda") else False,
        )

        processed_pred = 0
        for batch in tqdm(pred_dataloader, desc="Generated images"):
            if batch is None:
                continue
            self.fid.update(batch["img_uint8"].to(self.device), real=False)
            processed_pred += len(batch["path"])

        logger.info(
            "Successfully processed %s real images and %s generated images",
            processed_gt,
            processed_pred,
        )
        return self.fid.compute().item()

    def _compute_fid_buffered_with_dataloader(self, image_pairs: list) -> float:
        """Calculate buffered preserved-region FID."""
        self.fid.reset()

        dataset = MaskedImagePairDataset(
            image_pairs,
            image_size=self.image_size,
            ignore_mask=self.ignore_mask,
            mask_dilation_px=self.mask_dilation_px,
        )
        dataloader = TorchDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True if self.device.startswith("cuda") else False,
        )

        processed = 0
        for batch in tqdm(dataloader, desc="Buffered FID"):
            if batch is None:
                continue
            self.fid.update(batch["gt_uint8"].to(self.device), real=True)
            self.fid.update(batch["pred_uint8"].to(self.device), real=False)
            processed += len(batch["id"])

        logger.info("Successfully processed %s image pairs for buffered FID", processed)
        return self.fid.compute().item()

    def compute(
        self,
        gt_images: list,
        pred_images: list,
        metrics_to_compute: List[str] = _supported_metrics,
        image_pairs: list | None = None,
    ) -> dict:
        """Calculate distribution metrics."""
        results = {}
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
            if metric == "fid":
                results[metric] = self._compute_fid_with_dataloader(gt_images, pred_images)
            elif metric == "fid_buffered":
                if image_pairs is None:
                    raise ValueError("image_pairs are required when computing fid_buffered")
                results[metric] = self._compute_fid_buffered_with_dataloader(image_pairs)

        return results
