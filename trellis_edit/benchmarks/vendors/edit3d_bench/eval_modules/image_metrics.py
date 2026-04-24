"""
Image Quality Evaluation Module
Contains image quality metrics such as PSNR, SSIM, LPIPS, etc.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import lpips
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoImageProcessor, Dinov2Model

from .data_loader import EditInstructionSample
from .mask_utils import build_include_mask, load_prepared_rgb_image

logger = logging.getLogger(__name__)


class ImagePairDataset(Dataset):
    """Image pair dataset"""

    def __init__(
        self,
        image_pairs,
        image_size=(512, 512),
        ignore_mask=False,
        include_pil_images=False,
        return_mask=True,
        load_mask=True,
        mask_dilation_px=0,
    ):
        self.image_pairs = image_pairs
        self.image_size = image_size
        self.to_tensor = transforms.ToTensor()
        self.ignore_mask = ignore_mask
        self.include_pil_images = include_pil_images
        self.return_mask = return_mask
        self.load_mask = load_mask
        self.mask_dilation_px = max(int(mask_dilation_px), 0)

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        pair = self.image_pairs[idx]

        try:
            # Load images
            img1 = load_prepared_rgb_image(pair.gt_path, self.image_size)
            img2 = load_prepared_rgb_image(pair.pred_path, self.image_size)

            # Convert to tensor
            img1_tensor = self.to_tensor(img1)  # [0,1]
            img2_tensor = self.to_tensor(img2)  # [0,1]

            sample = {
                "id": pair.id,
                "img1": img1_tensor,  # [0,1] for PSNR and SSIM
                "img2": img2_tensor,  # [0,1] for PSNR and SSIM
            }

            if self.include_pil_images:
                sample["img1_pil"] = img1
                sample["img2_pil"] = img2

            if self.return_mask:
                include_mask = build_include_mask(
                    pair.mask_path if self.load_mask else None,
                    self.image_size,
                    dilation_px=self.mask_dilation_px,
                    ignore_mask=self.ignore_mask or not self.load_mask,
                )
                sample["include_mask"] = torch.from_numpy(
                    include_mask.astype(np.float32)
                )

            return sample

        except Exception as e:
            logger.warning(f"Failed to load image pair {pair.id}: {e}")
            return None


class ImageMetricsEvaluator:
    """Image quality evaluator"""

    _supported_metrics = ["psnr", "ssim", "lpips", "ssim_buffered", "lpips_buffered", "dino_if_max", "dino_if_mean"]
    _canonical_default_metrics = [
        "psnr",
        "ssim",
        "lpips",
        "dino_if_max",
        "dino_if_mean",
    ]
    _metric_aliases: Dict[str, str] = {}
    _dino_model_name = "facebook/dinov2-base"

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

        # Initialize heavy models lazily to avoid unnecessary startup overhead.
        self.lpips_fn = None
        self.dino_model = None
        self.dino_processor = None

        # Image preprocessing
        self.to_tensor = transforms.ToTensor()

        # SSIM window cache
        self._ssim_window = None

    def _ensure_lpips_loaded(self):
        """Load LPIPS only when it is actually needed."""
        if self.lpips_fn is None:
            self.lpips_fn = (
                lpips.LPIPS(net="alex", spatial=True, verbose=False)
                .to(self.device)
                .eval()
            )

    def _ensure_dino_loaded(self):
        """Load DINO only when it is actually needed."""
        if self.dino_model is None or self.dino_processor is None:
            self.dino_model, self.dino_processor = self._load_dino_components()

    def _load_dino_components(self):
        """Load DINO components, preferring local cache if online resolution fails."""
        local_snapshot = self._resolve_local_snapshot(self._dino_model_name)
        try:
            if local_snapshot is not None:
                model = Dinov2Model.from_pretrained(
                    local_snapshot, local_files_only=True
                ).to(self.device).eval()
                processor = AutoImageProcessor.from_pretrained(
                    local_snapshot, local_files_only=True, use_fast=False
                )
            else:
                raise OSError("Local snapshot not found")
        except OSError:
            model = Dinov2Model.from_pretrained(self._dino_model_name).to(
                self.device
            ).eval()
            processor = AutoImageProcessor.from_pretrained(
                self._dino_model_name, use_fast=False
            )
        return model, processor

    def _resolve_local_snapshot(self, model_name: str):
        """Return the newest local Hugging Face snapshot path when available."""
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

    def _normalize_metrics(
        self, metrics_to_compute: Optional[List[str]]
    ) -> Tuple[List[str], Dict[str, str]]:
        """Normalize deprecated aliases to canonical metric names."""
        if metrics_to_compute is None:
            metrics_to_compute = list(self._canonical_default_metrics)

        requested_metrics = []
        output_to_canonical = {}

        for metric in metrics_to_compute:
            canonical_metric = self._metric_aliases.get(metric, metric)
            if canonical_metric not in self._supported_metrics:
                continue

            output_to_canonical[metric] = canonical_metric
            if canonical_metric not in requested_metrics:
                requested_metrics.append(canonical_metric)

        return requested_metrics, output_to_canonical

    def _prepare_rgb_image(self, image: Image.Image) -> Image.Image:
        """Convert image to RGB on a neutral background and resize consistently."""
        if image.mode != "RGB":
            background = Image.new("RGBA", image.size, (127, 127, 127, 255))
            background.paste(image, (0, 0), image)
            image = background.convert("RGB")
        return image.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)

    def _load_prepared_pil(self, image_path: str) -> Image.Image:
        """Load and normalize a PIL image for DINO evaluation."""
        return self._prepare_rgb_image(Image.open(image_path))

    @torch.no_grad()
    def _encode_dino_embeddings(self, images: Sequence[Image.Image]) -> torch.Tensor:
        """Encode images with DINOv2 and return normalized mean pooled embeddings."""
        self._ensure_dino_loaded()
        inputs = self.dino_processor(images=list(images), return_tensors="pt").to(
            self.device
        )
        features = self.dino_model(**inputs).last_hidden_state.mean(dim=1)
        return torch.nn.functional.normalize(features, dim=-1)

    def _extract_view_id(self, image_path: str) -> str:
        """Extract render view id from `render_XXXX.png` paths."""
        stem = Path(image_path).stem
        if "_" not in stem:
            return stem
        return stem.rsplit("_", 1)[-1]

    def _build_metric_payload(
        self, scores: Sequence[float], score_ids: Optional[Sequence[str]] = None
    ) -> dict:
        """Build a metric payload with explicit sample ids when available."""
        if len(scores) == 0:
            return {"mean": None, "std": None, "count": 0}

        scores_array = np.array(scores)
        payload = {
            "mean": float(np.mean(scores_array)),
            "std": float(np.std(scores_array)),
            "count": len(scores_array),
            "all_scores": scores_array.tolist(),
        }
        if score_ids is not None:
            payload["all_ids"] = [str(score_id) for score_id in score_ids]
        return payload

    def _build_dino_metric_payload(
        self,
        *,
        scores: Sequence[float],
        details: Sequence[dict],
        view_aggregation: str,
    ) -> dict:
        payload = self._build_metric_payload(
            scores,
            [str(item.get("id")) for item in details],
        )
        payload["details"] = list(details)
        payload["view_aggregation"] = view_aggregation
        payload["reference_image"] = "2d_edit.png"
        return payload

    def _get_ssim_window(self, window_size: int = 11, channel: int = 3):
        """Get SSIM window with caching"""
        if self._ssim_window is None or self._ssim_window.size(2) != window_size:

            def gaussian(window_size, sigma):
                gauss = torch.Tensor(
                    [
                        torch.exp(
                            torch.tensor(
                                -((x - window_size // 2) ** 2) / float(2 * sigma**2)
                            )
                        )
                        for x in range(window_size)
                    ]
                )
                return gauss / gauss.sum()

            _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
            _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
            window = _2D_window.expand(
                channel, 1, window_size, window_size
            ).contiguous()
            self._ssim_window = window.to(self.device)

        return self._ssim_window

    @torch.no_grad()
    def _calculate_psnr_torch(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        mask: torch.Tensor,
        max_val: float = 255.0,
    ) -> float:
        """Calculate PSNR with mask using PyTorch"""
        # Ensure inputs are on the correct device
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        mask = mask.to(self.device)

        # Convert [0,1] range to [0,255] range to match skimage
        img1 = img1 * 255.0
        img2 = img2 * 255.0

        mse = torch.mean((img1 - img2) ** 2, dim=0)  # [C,H,W]

        # Calculate average MSE in valid regions
        valid_pixels = mask.bool()  # Ensure mask is [H,W] shape
        if valid_pixels.sum() == 0:
            return None

        mse_valid = mse[valid_pixels].mean()

        if mse_valid == 0:
            return None

        # Calculate PSNR using standard formula
        psnr = 20 * torch.log10(max_val / torch.sqrt(mse_valid))

        return psnr.item()

    @torch.no_grad()
    def _calculate_psnr_batch_torch(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        mask: torch.Tensor,
        max_val: float = 255.0,
    ) -> List[float]:
        """Calculate masked PSNR for a whole batch."""
        img1 = img1.to(self.device, non_blocking=True) * 255.0
        img2 = img2.to(self.device, non_blocking=True) * 255.0
        valid_mask = mask.to(self.device, non_blocking=True).bool()

        mse = torch.mean((img1 - img2) ** 2, dim=1)
        valid_counts = valid_mask.flatten(1).sum(dim=1)
        mse_sum = (mse * valid_mask.float()).flatten(1).sum(dim=1)

        scores = [None] * img1.size(0)
        valid_indices = valid_counts > 0
        if valid_indices.any():
            mse_valid = torch.zeros_like(mse_sum)
            mse_valid[valid_indices] = (
                mse_sum[valid_indices] / valid_counts[valid_indices].float()
            )
            non_zero_indices = valid_indices & (mse_valid > 0)
            if non_zero_indices.any():
                psnr_values = 20 * torch.log10(
                    torch.tensor(max_val, device=self.device)
                    / torch.sqrt(mse_valid[non_zero_indices])
                )
                for idx, value in zip(
                    non_zero_indices.nonzero(as_tuple=False).flatten().tolist(),
                    psnr_values.detach().cpu().tolist(),
                ):
                    scores[idx] = value

        return scores

    @torch.no_grad()
    def _calculate_ssim_torch(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        mask: torch.Tensor,
        window_size: int = 11,
    ) -> float:
        """Calculate SSIM with mask using PyTorch"""
        # Ensure inputs are on the correct device
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        mask = mask.to(self.device)

        # Get cached SSIM window
        window = self._get_ssim_window(window_size, img1.size(1))

        # Calculate SSIM
        mu1 = torch.nn.functional.conv2d(
            img1, window, padding=window_size // 2, groups=img1.size(1)
        )
        mu2 = torch.nn.functional.conv2d(
            img2, window, padding=window_size // 2, groups=img2.size(1)
        )

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            torch.nn.functional.conv2d(
                img1 * img1, window, padding=window_size // 2, groups=img1.size(1)
            )
            - mu1_sq
        )
        sigma2_sq = (
            torch.nn.functional.conv2d(
                img2 * img2, window, padding=window_size // 2, groups=img2.size(1)
            )
            - mu2_sq
        )
        sigma12 = (
            torch.nn.functional.conv2d(
                img1 * img2, window, padding=window_size // 2, groups=img1.size(1)
            )
            - mu1_mu2
        )

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        # Calculate average SSIM in valid regions
        valid_pixels = mask.squeeze().bool()  # Ensure mask is [H,W] shape
        if valid_pixels.sum() == 0:
            return None

        # Ensure ssim_map is [H,W] shape
        ssim_map = ssim_map.squeeze()  # Remove batch dimension
        ssim_valid = ssim_map.mean(dim=0)[
            valid_pixels
        ].mean()  # First average over channels, then average over valid pixels
        return ssim_valid.item()

    @torch.no_grad()
    def _calculate_ssim_batch_torch(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        mask: torch.Tensor,
        window_size: int = 11,
    ) -> List[float]:
        """Calculate masked SSIM for a whole batch."""
        img1 = img1.to(self.device, non_blocking=True)
        img2 = img2.to(self.device, non_blocking=True)
        valid_mask = mask.to(self.device, non_blocking=True).bool()

        window = self._get_ssim_window(window_size, img1.size(1))

        mu1 = torch.nn.functional.conv2d(
            img1, window, padding=window_size // 2, groups=img1.size(1)
        )
        mu2 = torch.nn.functional.conv2d(
            img2, window, padding=window_size // 2, groups=img2.size(1)
        )

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            torch.nn.functional.conv2d(
                img1 * img1, window, padding=window_size // 2, groups=img1.size(1)
            )
            - mu1_sq
        )
        sigma2_sq = (
            torch.nn.functional.conv2d(
                img2 * img2, window, padding=window_size // 2, groups=img2.size(1)
            )
            - mu2_sq
        )
        sigma12 = (
            torch.nn.functional.conv2d(
                img1 * img2, window, padding=window_size // 2, groups=img1.size(1)
            )
            - mu1_mu2
        )

        C1 = 0.01**2
        C2 = 0.03**2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        ssim_map = ssim_map.mean(dim=1)
        valid_counts = valid_mask.flatten(1).sum(dim=1)
        ssim_sum = (ssim_map * valid_mask.float()).flatten(1).sum(dim=1)

        scores = [None] * img1.size(0)
        valid_indices = valid_counts > 0
        if valid_indices.any():
            ssim_values = ssim_sum[valid_indices] / valid_counts[valid_indices].float()
            for idx, value in zip(
                valid_indices.nonzero(as_tuple=False).flatten().tolist(),
                ssim_values.detach().cpu().tolist(),
            ):
                scores[idx] = value

        return scores

    @torch.no_grad()
    def _calculate_lpips_masked(
        self, img1_t: torch.Tensor, img2_t: torch.Tensor, include_mask_t: torch.Tensor
    ) -> float:
        """Calculate LPIPS with mask"""
        self._ensure_lpips_loaded()
        # Calculate LPIPS spatial map
        lpips_map = self.lpips_fn(img1_t, img2_t)  # [B,1,H,W]
        if lpips_map.shape[-2:] != include_mask_t.shape[-2:]:
            include_mask_t = torch.nn.functional.interpolate(
                include_mask_t.float(),
                size=lpips_map.shape[-2:],
                mode="nearest",
            )

        # Calculate average LPIPS in valid regions
        valid_lpips = lpips_map[include_mask_t.bool()]
        if len(valid_lpips) == 0:
            return None

        return valid_lpips.mean().cpu().numpy().tolist()

    @torch.no_grad()
    def _calculate_lpips_batch_masked(
        self, img1_t: torch.Tensor, img2_t: torch.Tensor, include_mask_t: torch.Tensor
    ) -> List[float]:
        """Calculate masked LPIPS for a whole batch."""
        self._ensure_lpips_loaded()
        img1_t = img1_t.to(self.device, non_blocking=True)
        img2_t = img2_t.to(self.device, non_blocking=True)
        include_mask_t = include_mask_t.to(self.device, non_blocking=True)

        lpips_map = self.lpips_fn(img1_t, img2_t)
        if lpips_map.shape[-2:] != include_mask_t.shape[-2:]:
            include_mask_t = torch.nn.functional.interpolate(
                include_mask_t.float(),
                size=lpips_map.shape[-2:],
                mode="nearest",
            )
        valid_mask = include_mask_t.bool()
        valid_counts = valid_mask.flatten(1).sum(dim=1)
        lpips_sum = (lpips_map * valid_mask.float()).flatten(1).sum(dim=1)

        scores = [None] * img1_t.size(0)
        valid_indices = valid_counts > 0
        if valid_indices.any():
            lpips_values = lpips_sum[valid_indices] / valid_counts[valid_indices].float()
            for idx, value in zip(
                valid_indices.nonzero(as_tuple=False).flatten().tolist(),
                lpips_values.detach().cpu().tolist(),
            ):
                scores[idx] = value

        return scores

    @torch.no_grad()
    def _calculate_dino_if_variants(
        self, edit_instruction_samples: List[EditInstructionSample]
    ) -> Dict[str, List]:
        """Compare `2d_edit.png` with all predicted render views and return max/mean variants."""
        max_scores = []
        mean_scores = []
        details = []

        for sample in tqdm(edit_instruction_samples, desc="Computing dino_if variants"):
            try:
                reference_image = self._load_prepared_pil(sample.reference_path)
                pred_images = [
                    self._load_prepared_pil(image_path) for image_path in sample.pred_paths
                ]

                if not pred_images:
                    continue

                embeddings = self._encode_dino_embeddings([reference_image] + pred_images)
                reference_embedding = embeddings[0]
                pred_embeddings = embeddings[1:]

                similarities = pred_embeddings @ reference_embedding
                best_score, best_index = similarities.max(dim=0)
                best_view_index = int(best_index.item())

                best_value = float(best_score.item())
                mean_value = float(similarities.mean().item())
                best_view_path = sample.pred_paths[best_view_index]

                max_scores.append(best_value)
                mean_scores.append(mean_value)
                details.append(
                    {
                        "id": sample.id,
                        "score_max": best_value,
                        "score_mean": mean_value,
                        "best_view_score": best_value,
                        "best_view_id": self._extract_view_id(best_view_path),
                        "best_view_path": best_view_path,
                        "num_views": len(sample.pred_paths),
                        "reference_path": sample.reference_path,
                    }
                )
            except Exception as exc:
                logger.warning(
                    "Failed to compute dino_if variants for %s: %s", sample.id, exc
                )

        return {
            "max_scores": max_scores,
            "mean_scores": mean_scores,
            "details": details,
        }

    def _compute_metrics_with_dataloader(
        self,
        image_pairs,
        metrics_to_compute,
        *,
        mask_dilation_px: int = 0,
    ):
        """Calculate metrics using DataLoader."""
        return_mask = bool(
            {"psnr", "ssim", "lpips", "ssim_buffered", "lpips_buffered"}
            & set(metrics_to_compute)
        )

        # Create dataset
        dataset = ImagePairDataset(
            image_pairs,
            self.image_size,
            self.ignore_mask,
            return_mask=return_mask,
            load_mask=return_mask and not self.ignore_mask,
            mask_dilation_px=mask_dilation_px,
        )

        # Create data loader
        dataloader = TorchDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True if self.device.startswith("cuda") else False,
        )

        scores = {metric: [] for metric in metrics_to_compute}
        score_ids = {metric: [] for metric in metrics_to_compute}
        processed = 0
        skipped = 0

        for batch in tqdm(
            dataloader, desc=f"Computing {', '.join(metrics_to_compute)}"
        ):
            if batch is None:
                skipped += self.batch_size
                continue

            if "psnr" in metrics_to_compute:
                for sample_id, psnr in zip(
                    batch["id"],
                    self._calculate_psnr_batch_torch(
                        batch["img1"], batch["img2"], batch["include_mask"]
                    ),
                ):
                    if psnr is not None:
                        scores["psnr"].append(psnr)
                        score_ids["psnr"].append(sample_id)
                        processed += 1
                    else:
                        skipped += 1

            ssim_metric_names = [
                metric_name
                for metric_name in ("ssim", "ssim_buffered")
                if metric_name in metrics_to_compute
            ]
            if ssim_metric_names:
                for sample_id, ssim in zip(
                    batch["id"],
                    self._calculate_ssim_batch_torch(
                        batch["img1"], batch["img2"], batch["include_mask"]
                    ),
                ):
                    if ssim is None:
                        continue
                    for metric_name in ssim_metric_names:
                        scores[metric_name].append(ssim)
                        score_ids[metric_name].append(sample_id)

            lpips_metric_names = [
                metric_name
                for metric_name in ("lpips", "lpips_buffered")
                if metric_name in metrics_to_compute
            ]
            if lpips_metric_names:
                for sample_id, lpips_score in zip(
                    batch["id"],
                    self._calculate_lpips_batch_masked(
                        batch["img1"],
                        batch["img2"],
                        batch["include_mask"].unsqueeze(1),
                    ),
                ):
                    if lpips_score is None:
                        continue
                    for metric_name in lpips_metric_names:
                        scores[metric_name].append(lpips_score)
                        score_ids[metric_name].append(sample_id)

        return scores, score_ids, processed, skipped

    def _collate_fn(self, batch):
        """Custom collate function to handle None values"""
        # Filter out None values
        valid_batch = [item for item in batch if item is not None]

        if not valid_batch:
            return None

        # Reorganize data
        collated = {}
        for key in valid_batch[0].keys():
            if key == "id":
                collated[key] = [item[key] for item in valid_batch]
            elif key == "img1_pil" or key == "img2_pil":
                collated[key] = [item[key] for item in valid_batch]
            else:
                collated[key] = torch.stack([item[key] for item in valid_batch])

        return collated

    def compute(
        self,
        image_pairs,
        metrics_to_compute: Optional[List[str]] = None,
        edit_instruction_samples: Optional[List[EditInstructionSample]] = None,
    ) -> dict:
        requested_metrics, output_to_canonical = self._normalize_metrics(
            metrics_to_compute
        )
        pair_metrics = [
            metric
            for metric in requested_metrics
            if metric in {"psnr", "ssim", "lpips", "ssim_buffered", "lpips_buffered"}
        ]
        base_pair_metrics = [
            metric for metric in pair_metrics if metric in {"psnr", "ssim", "lpips"}
        ]
        buffered_pair_metrics = [
            metric
            for metric in pair_metrics
            if metric in {"ssim_buffered", "lpips_buffered"}
        ]

        if {"lpips", "lpips_buffered"} & set(pair_metrics):
            self._ensure_lpips_loaded()
        if {"dino_if_max", "dino_if_mean"} & set(requested_metrics):
            self._ensure_dino_loaded()

        logger.info(
            "Computing %s metrics (batch_size=%s, num_workers=%s)...",
            requested_metrics,
            self.batch_size,
            self.num_workers,
        )

        canonical_results = {}

        if base_pair_metrics:
            scores, score_ids, processed, skipped = self._compute_metrics_with_dataloader(
                image_pairs,
                base_pair_metrics,
                mask_dilation_px=0,
            )
            for metric in base_pair_metrics:
                canonical_results[metric] = self._build_metric_payload(
                    scores[metric], score_ids[metric]
                )
        else:
            processed = 0
            skipped = 0

        if buffered_pair_metrics:
            buffered_scores, buffered_score_ids, _, _ = self._compute_metrics_with_dataloader(
                image_pairs,
                buffered_pair_metrics,
                mask_dilation_px=self.mask_dilation_px,
            )
            for metric in buffered_pair_metrics:
                canonical_results[metric] = self._build_metric_payload(
                    buffered_scores[metric], buffered_score_ids[metric]
                )

        if {"dino_if_max", "dino_if_mean"} & set(requested_metrics):
            dino_if_payload = self._calculate_dino_if_variants(
                edit_instruction_samples or []
            )
            dino_if_details = dino_if_payload["details"]

            if "dino_if_max" in requested_metrics:
                canonical_results["dino_if_max"] = self._build_dino_metric_payload(
                    scores=dino_if_payload["max_scores"],
                    details=dino_if_details,
                    view_aggregation="max",
                )
            if "dino_if_mean" in requested_metrics:
                canonical_results["dino_if_mean"] = self._build_dino_metric_payload(
                    scores=dino_if_payload["mean_scores"],
                    details=dino_if_details,
                    view_aggregation="mean",
                )

        results = {}
        for metric, canonical_metric in output_to_canonical.items():
            if canonical_metric in canonical_results:
                results[metric] = dict(canonical_results[canonical_metric])

        return results
