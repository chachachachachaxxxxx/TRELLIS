from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from trellis_edit.preprocess import save_preprocessed_inputs
from trellis_edit.preprocess.image_alignment import (
    PROC_IMAGE_SIZE,
    CropContext,
    PreparedInputs,
    composite_rgb_from_rgba,
    extract_foreground_rgba,
    has_useful_alpha,
    scale_image,
)
from trellis_edit.preprocess.mask_utils import build_auto_mask, build_blank_mask, binarize_mask_image, extract_mask_channel

from .artifacts import PreprocessArtifact
from .base import ExperimentContext, PreprocessPlugin
from .config import ForegroundPolicy, MaskPolicy, PreprocessConfig


@dataclass(frozen=True)
class _PreparedForeground:
    image: Image.Image
    mask: np.ndarray


class SharedImagePreprocessPlugin(PreprocessPlugin):
    name = "shared_image"

    def _resolve_mask(self, context: ExperimentContext, config: PreprocessConfig) -> Image.Image:
        source_image = context.loaded_inputs.source_image
        edit_image = context.loaded_inputs.edit_image
        mask_image = context.loaded_inputs.mask_image

        policy: MaskPolicy = config.mask_policy
        if policy == "provided":
            if mask_image is None:
                raise RuntimeError("mask_policy='provided' requires inputs.mask_image")
            return mask_image
        if policy == "auto_diff":
            return build_auto_mask(
                source_image=source_image,
                edit_image=edit_image,
                threshold=config.auto_mask_diff_threshold,
                max_filter=config.auto_mask_max_filter,
            )
        if policy == "blank":
            return build_blank_mask(source_image.size)
        raise RuntimeError(f"Unknown mask policy: {policy}")

    def _prepare_foreground(
        self,
        image: Image.Image,
        scale: float,
        pipeline,
        policy: ForegroundPolicy,
    ) -> _PreparedForeground:
        if policy == "alpha_or_rembg":
            rgba = extract_foreground_rgba(image, pipeline, scale)
            alpha_mask = np.asarray(rgba)[:, :, 3] > int(0.8 * 255)
            return _PreparedForeground(image=rgba, mask=alpha_mask)

        scaled = scale_image(image, scale, Image.Resampling.LANCZOS)
        if policy == "alpha_only":
            if not has_useful_alpha(scaled):
                raise RuntimeError("foreground_policy='alpha_only' requires RGBA inputs with useful alpha")
            rgba = scaled.convert("RGBA")
            alpha_mask = np.asarray(rgba)[:, :, 3] > int(0.8 * 255)
            return _PreparedForeground(image=rgba, mask=alpha_mask)

        if policy == "raw_rgb":
            rgb = scaled.convert("RGB")
            mask = np.ones((rgb.height, rgb.width), dtype=bool)
            return _PreparedForeground(image=rgb, mask=mask)

        raise RuntimeError(f"Unknown foreground policy: {policy}")

    @staticmethod
    def _compute_bbox(active: np.ndarray) -> tuple[int, int, int, int]:
        coords = np.argwhere(active)
        if coords.size == 0:
            raise RuntimeError("Preprocess crop produced an empty active region")

        x_min = int(coords[:, 1].min())
        y_min = int(coords[:, 0].min())
        x_max = int(coords[:, 1].max())
        y_max = int(coords[:, 0].max())
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        size = max(int(max(x_max - x_min, y_max - y_min) * 1.2), 1)
        return (
            int(round(center_x - size / 2.0)),
            int(round(center_y - size / 2.0)),
            int(round(center_x + size / 2.0)),
            int(round(center_y + size / 2.0)),
        )

    def _build_crop_context(
        self,
        source_fg: _PreparedForeground,
        edit_fg: _PreparedForeground,
        mask_image: Image.Image,
        original_size: tuple[int, int],
        config: PreprocessConfig,
        scale: float,
    ) -> CropContext:
        if config.crop_policy == "disabled":
            return CropContext(scale=1.0, bbox=(0, 0, original_size[0], original_size[1]))

        scaled_mask = scale_image(
            extract_mask_channel(mask_image),
            scale,
            Image.Resampling.NEAREST,
        )
        mask_binary = np.asarray(scaled_mask) > int(config.mask_threshold)

        if config.crop_policy == "union_crop":
            active = source_fg.mask | edit_fg.mask | mask_binary
        elif config.crop_policy == "mask_crop":
            active = mask_binary
        else:
            raise RuntimeError(f"Unknown crop policy: {config.crop_policy}")

        return CropContext(scale=scale, bbox=self._compute_bbox(active))

    @staticmethod
    def _render_output_image(image: Image.Image) -> Image.Image:
        if image.mode == "RGBA":
            return composite_rgb_from_rgba(image)
        return image.convert("RGB")

    def run(
        self,
        context: ExperimentContext,
        config: PreprocessConfig,
    ) -> PreprocessArtifact:
        source_image = context.loaded_inputs.source_image
        edit_image = context.loaded_inputs.edit_image

        if source_image.size != edit_image.size:
            raise RuntimeError(
                f"source_image and edit_image must have the same size, got {source_image.size} and {edit_image.size}"
            )

        mask_image = self._resolve_mask(context, config)

        if config.crop_policy == "disabled":
            source = self._render_output_image(
                source_image.resize((PROC_IMAGE_SIZE, PROC_IMAGE_SIZE), Image.Resampling.LANCZOS)
            )
            edit = self._render_output_image(
                edit_image.resize((PROC_IMAGE_SIZE, PROC_IMAGE_SIZE), Image.Resampling.LANCZOS)
            )
            mask = binarize_mask_image(
                extract_mask_channel(mask_image).resize((PROC_IMAGE_SIZE, PROC_IMAGE_SIZE), Image.Resampling.NEAREST),
                threshold=config.mask_threshold,
            )
            prepared = PreparedInputs(
                source=source,
                edit=edit,
                mask=mask,
                meta={
                    "crop_policy": config.crop_policy,
                    "foreground_policy": config.foreground_policy,
                    "mask_policy": config.mask_policy,
                    "mask_threshold": config.mask_threshold,
                    "proc_size": [PROC_IMAGE_SIZE, PROC_IMAGE_SIZE],
                    "scale": 1.0,
                    "crop_bbox_xyxy": [0, 0, int(source_image.width), int(source_image.height)],
                },
            )
            return PreprocessArtifact(
                prepared_inputs=prepared,
                artifact_paths=save_preprocessed_inputs(context.preprocess_dir, prepared, include_mask=True),
            )

        scale = min(1.0, 1024.0 / float(max(source_image.size)))
        source_fg = self._prepare_foreground(source_image, scale, context.pipeline, config.foreground_policy)
        edit_fg = self._prepare_foreground(edit_image, scale, context.pipeline, config.foreground_policy)
        crop_context = self._build_crop_context(
            source_fg=source_fg,
            edit_fg=edit_fg,
            mask_image=mask_image,
            original_size=source_image.size,
            config=config,
            scale=scale,
        )

        scaled_mask = scale_image(
            extract_mask_channel(mask_image),
            crop_context.scale,
            Image.Resampling.NEAREST,
        )

        source = self._render_output_image(
            source_fg.image.crop(crop_context.bbox).resize((PROC_IMAGE_SIZE, PROC_IMAGE_SIZE), Image.Resampling.LANCZOS)
        )
        edit = self._render_output_image(
            edit_fg.image.crop(crop_context.bbox).resize((PROC_IMAGE_SIZE, PROC_IMAGE_SIZE), Image.Resampling.LANCZOS)
        )
        mask = binarize_mask_image(
            scaled_mask.crop(crop_context.bbox).resize((PROC_IMAGE_SIZE, PROC_IMAGE_SIZE), Image.Resampling.NEAREST),
            threshold=config.mask_threshold,
        )

        prepared = PreparedInputs(
            source=source,
            edit=edit,
            mask=mask,
            meta={
                "crop_policy": config.crop_policy,
                "foreground_policy": config.foreground_policy,
                "mask_policy": config.mask_policy,
                "mask_threshold": config.mask_threshold,
                "proc_size": [PROC_IMAGE_SIZE, PROC_IMAGE_SIZE],
                "scale": float(crop_context.scale),
                "crop_bbox_xyxy": [int(value) for value in crop_context.bbox],
            },
        )
        return PreprocessArtifact(
            prepared_inputs=prepared,
            artifact_paths=save_preprocessed_inputs(context.preprocess_dir, prepared, include_mask=True),
        )
