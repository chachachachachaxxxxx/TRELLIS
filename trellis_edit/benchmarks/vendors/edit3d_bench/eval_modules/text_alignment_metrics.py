"""
Text Alignment Evaluation Module
Contains text alignment metrics such as CLIP-T
"""

import logging
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)


class ImageTextPairDataset(Dataset):
    """Image-text pair dataset"""

    def __init__(self, image_text_pairs, image_size=(512, 512)):
        self.image_text_pairs = image_text_pairs
        self.image_size = image_size
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_text_pairs)

    def __getitem__(self, idx):
        pair = self.image_text_pairs[idx]

        try:
            # Load image
            img = Image.open(pair.image_path)
            if img.mode != "RGB":
                background = Image.new("RGBA", img.size, (127, 127, 127, 255))
                background.paste(img, (0, 0), img)
                img = background.convert("RGB")
            img = img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)

            # Get text
            text = pair.text

            return {"id": pair.id, "image": img, "text": text}

        except Exception as e:
            logger.warning(f"Failed to load image-text pair {pair.id}: {e}")
            return None


class TextAlignmentMetricsEvaluator:
    """Text alignment evaluator"""

    _supported_metrics = ["clip_t"]
    _clip_model_name = "openai/clip-vit-base-patch32"

    def __init__(
        self,
        device: str = "cuda:0",
        image_size: tuple = (512, 512),
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        self.device = device
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Initialize CLIP model
        self.clip_model, self.clip_processor = self._load_clip_components()
        self._text_feature_cache = {}

        # Image preprocessing
        self.to_tensor = transforms.ToTensor()

    def _load_clip_components(self):
        """Load CLIP model and processor, with a local-cache fallback."""
        local_snapshot = self._resolve_local_snapshot(self._clip_model_name)
        try:
            if local_snapshot is not None:
                model = CLIPModel.from_pretrained(
                    local_snapshot, local_files_only=True, use_safetensors=False
                ).to(self.device).eval()
                processor = CLIPProcessor.from_pretrained(
                    local_snapshot, local_files_only=True
                )
            else:
                raise OSError("Local snapshot not found")
        except OSError:
            model = CLIPModel.from_pretrained(
                self._clip_model_name, use_safetensors=False
            ).to(
                self.device
            ).eval()
            processor = CLIPProcessor.from_pretrained(self._clip_model_name)
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

    def _unwrap_clip_embedding(self, features: torch.Tensor) -> torch.Tensor:
        """Extract the projected embedding across different transformers versions."""
        if hasattr(features, "pooler_output"):
            return features.pooler_output
        if isinstance(features, tuple):
            if len(features) == 0:
                raise ValueError("Empty CLIP feature tuple received")
            return features[-1]
        return features

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
            elif key == "text":
                collated[key] = [item[key] for item in valid_batch]
            elif key == "image":
                collated[key] = [item[key] for item in valid_batch]

        return collated

    @torch.no_grad()
    def _get_text_features(self, texts: List[str]) -> torch.Tensor:
        """Encode texts once and reuse cached CLIP text features."""
        missing_texts = []
        for text in texts:
            if text not in self._text_feature_cache:
                missing_texts.append(text)

        if missing_texts:
            unique_missing_texts = list(dict.fromkeys(missing_texts))
            text_inputs = self.clip_processor(
                text=unique_missing_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            text_inputs = {
                key: value.to(self.device, non_blocking=True)
                for key, value in text_inputs.items()
            }
            text_features = self.clip_model.get_text_features(**text_inputs)
            text_features = self._unwrap_clip_embedding(text_features)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            for text, feature in zip(unique_missing_texts, text_features):
                self._text_feature_cache[text] = feature.detach().cpu()

        return torch.stack([self._text_feature_cache[text] for text in texts], dim=0).to(
            self.device, non_blocking=True
        )

    @torch.no_grad()
    def _compute_clip_similarity_batch(
        self, images: List[Image.Image], texts: List[str]
    ) -> List[float]:
        """Calculate CLIP image-text similarity for a whole batch."""
        image_inputs = self.clip_processor(images=images, return_tensors="pt")
        image_inputs = {
            key: value.to(self.device, non_blocking=True)
            for key, value in image_inputs.items()
        }

        image_features = self.clip_model.get_image_features(**image_inputs)
        image_features = self._unwrap_clip_embedding(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_features = self._get_text_features(texts)
        similarity = torch.sum(image_features * text_features, dim=-1)
        return similarity.detach().cpu().tolist()

    def _compute_clip_t_with_dataloader(self, image_text_pairs) -> dict:
        """Calculate CLIP-T using DataLoader"""
        logger.info(
            f"Computing CLIP-T (batch_size={self.batch_size}, num_workers={self.num_workers})..."
        )

        # Create dataset
        dataset = ImageTextPairDataset(image_text_pairs, self.image_size)

        # Create data loader
        dataloader = TorchDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True if self.device.startswith("cuda") else False,
        )

        scores = []
        score_ids = []
        processed = 0
        skipped = 0

        for batch in tqdm(dataloader, desc="Computing CLIP-T"):
            if batch is None:
                skipped += self.batch_size
                continue

            try:
                similarities = self._compute_clip_similarity_batch(
                    batch["image"], batch["text"]
                )
                for sample_id, similarity in zip(batch["id"], similarities):
                    scores.append(similarity)
                    score_ids.append(sample_id)
                processed += len(similarities)
            except Exception as e:
                logger.warning(
                    "Failed to process CLIP-T batch starting at %s: %s",
                    batch["id"][0],
                    e,
                )
                skipped += len(batch["id"])
                continue

        return scores, score_ids, processed, skipped

    def compute_clip_t(self, image_text_pairs) -> dict:
        """Calculate CLIP-T metric"""
        scores, score_ids, processed, skipped = self._compute_clip_t_with_dataloader(
            image_text_pairs
        )

        logger.info(
            f"Successfully processed {processed} image-text pairs, skipped {skipped}"
        )

        if len(scores) == 0:
            return {"mean": None, "std": None, "count": 0}

        scores = np.array(scores)
        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "count": len(scores),
            "all_scores": scores.tolist(),
            "all_ids": [str(score_id) for score_id in score_ids],
        }

    def compute(
        self,
        image_text_pairs: list,
        metrics_to_compute: List[str] = _supported_metrics,
    ) -> dict:
        """Calculate CLIP-T metric"""
        result = {}
        metrics_to_compute = list(
            set(metrics_to_compute) & set(self._supported_metrics)
        )
        for metric in metrics_to_compute:
            if metric == "clip_t":
                result[metric] = self.compute_clip_t(image_text_pairs)
        return result
