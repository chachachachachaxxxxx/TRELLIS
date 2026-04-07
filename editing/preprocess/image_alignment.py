from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import Tuple

import numpy as np
from PIL import Image

from .mask_utils import binarize_mask_image, extract_mask_channel


PROC_IMAGE_SIZE = 518


@dataclass(frozen=True)
class CropContext:
    scale: float
    bbox: Tuple[int, int, int, int]


@dataclass(frozen=True)
class PreparedInputs:
    source: Image.Image
    edit: Image.Image
    mask: Image.Image
    meta: dict


@dataclass(frozen=True)
class PreparedEditCondition:
    edit: Image.Image
    meta: dict


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def has_useful_alpha(image: Image.Image) -> bool:
    if image.mode != "RGBA":
        return False
    alpha = np.asarray(image)[:, :, 3]
    return not np.all(alpha == 255)


def scale_image(image: Image.Image, scale: float, resample) -> Image.Image:
    if abs(scale - 1.0) < 1e-8:
        return image
    new_w = max(1, int(round(image.width * scale)))
    new_h = max(1, int(round(image.height * scale)))
    return image.resize((new_w, new_h), resample)


def composite_rgb_from_rgba(image: Image.Image) -> Image.Image:
    rgba = np.asarray(image.convert("RGBA")).astype(np.float32) / 255.0
    rgb = rgba[:, :, :3] * rgba[:, :, 3:4]
    return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")


def extract_foreground_rgba(
    image: Image.Image,
    pipeline,
    scale: float,
) -> Image.Image:
    image = scale_image(image, scale, Image.Resampling.LANCZOS)
    if has_useful_alpha(image):
        return image.convert("RGBA")

    if not _module_available("rembg"):
        raise RuntimeError(
            "rembg is not installed in the current environment. "
            "Please install rembg or provide RGBA source/edit images with alpha."
        )

    import rembg

    if getattr(pipeline, "rembg_session", None) is None:
        pipeline.rembg_session = rembg.new_session("u2net")
    output = rembg.remove(image.convert("RGB"), session=pipeline.rembg_session)
    if not isinstance(output, Image.Image):
        raise RuntimeError("rembg did not return a PIL image as expected.")
    return output.convert("RGBA")


def build_union_crop_context(
    source_image: Image.Image,
    edit_image: Image.Image,
    mask_image: Image.Image,
    pipeline,
    mask_threshold: int,
) -> CropContext:
    if source_image.size != edit_image.size:
        raise RuntimeError(
            f"source-image and edit-image must have the same spatial size for aligned token editing, "
            f"got {source_image.size} vs {edit_image.size}."
        )

    max_size = max(source_image.size)
    scale = min(1.0, 1024.0 / float(max_size))

    source_rgba = extract_foreground_rgba(source_image, pipeline, scale)
    edit_rgba = extract_foreground_rgba(edit_image, pipeline, scale)
    aligned_mask = extract_mask_channel(mask_image)
    if aligned_mask.size != source_image.size:
        aligned_mask = aligned_mask.resize(source_image.size, Image.Resampling.NEAREST)
    scaled_mask = scale_image(aligned_mask, scale, Image.Resampling.NEAREST)

    source_alpha = np.asarray(source_rgba)[:, :, 3] > int(0.8 * 255)
    edit_alpha = np.asarray(edit_rgba)[:, :, 3] > int(0.8 * 255)
    mask_binary = np.asarray(scaled_mask) > int(mask_threshold)
    union = source_alpha | edit_alpha | mask_binary

    bbox_pixels = np.argwhere(union)
    if bbox_pixels.size == 0:
        return CropContext(scale=scale, bbox=(0, 0, source_rgba.width, source_rgba.height))

    x_min = int(np.min(bbox_pixels[:, 1]))
    y_min = int(np.min(bbox_pixels[:, 0]))
    x_max = int(np.max(bbox_pixels[:, 1]))
    y_max = int(np.max(bbox_pixels[:, 0]))
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    size = int(max(x_max - x_min, y_max - y_min) * 1.2)
    size = max(size, 1)
    bbox = (
        int(round(center_x - size / 2.0)),
        int(round(center_y - size / 2.0)),
        int(round(center_x + size / 2.0)),
        int(round(center_y + size / 2.0)),
    )
    return CropContext(scale=scale, bbox=bbox)


def prepare_aligned_inputs(
    source_image: Image.Image,
    edit_image: Image.Image,
    mask_image: Image.Image,
    pipeline,
    preprocess: bool,
    mask_threshold: int,
) -> PreparedInputs:
    if source_image.size != edit_image.size:
        raise RuntimeError(
            f"source-image and edit-image must have the same spatial size, "
            f"got {source_image.size} vs {edit_image.size}."
        )

    if preprocess:
        crop_context = build_union_crop_context(
            source_image=source_image,
            edit_image=edit_image,
            mask_image=mask_image,
            pipeline=pipeline,
            mask_threshold=mask_threshold,
        )

        source_rgba = extract_foreground_rgba(source_image, pipeline, crop_context.scale)
        edit_rgba = extract_foreground_rgba(edit_image, pipeline, crop_context.scale)
        aligned_mask = extract_mask_channel(mask_image)
        if aligned_mask.size != source_image.size:
            aligned_mask = aligned_mask.resize(source_image.size, Image.Resampling.NEAREST)
        scaled_mask = scale_image(aligned_mask, crop_context.scale, Image.Resampling.NEAREST)

        source = composite_rgb_from_rgba(
            source_rgba.crop(crop_context.bbox).resize((PROC_IMAGE_SIZE, PROC_IMAGE_SIZE), Image.Resampling.LANCZOS)
        )
        edit = composite_rgb_from_rgba(
            edit_rgba.crop(crop_context.bbox).resize((PROC_IMAGE_SIZE, PROC_IMAGE_SIZE), Image.Resampling.LANCZOS)
        )
        mask = binarize_mask_image(
            scaled_mask.crop(crop_context.bbox).resize((PROC_IMAGE_SIZE, PROC_IMAGE_SIZE), Image.Resampling.NEAREST),
            threshold=mask_threshold,
        )
        meta = {
            "preprocess": True,
            "proc_size": [PROC_IMAGE_SIZE, PROC_IMAGE_SIZE],
            "source_original_size": list(source_image.size),
            "edit_original_size": list(edit_image.size),
            "mask_original_size": list(mask_image.size),
            "scale": float(crop_context.scale),
            "crop_bbox_xyxy": [int(v) for v in crop_context.bbox],
            "mask_threshold": int(mask_threshold),
            "crop_rule": "shared_union_of_source_foreground_edit_foreground_and_mask",
        }
        return PreparedInputs(source=source, edit=edit, mask=mask, meta=meta)

    source = source_image.resize((PROC_IMAGE_SIZE, PROC_IMAGE_SIZE), Image.Resampling.LANCZOS)
    edit = edit_image.resize((PROC_IMAGE_SIZE, PROC_IMAGE_SIZE), Image.Resampling.LANCZOS)
    if source.mode == "RGBA":
        source = composite_rgb_from_rgba(source)
    else:
        source = source.convert("RGB")
    if edit.mode == "RGBA":
        edit = composite_rgb_from_rgba(edit)
    else:
        edit = edit.convert("RGB")
    mask = binarize_mask_image(
        extract_mask_channel(mask_image).resize((PROC_IMAGE_SIZE, PROC_IMAGE_SIZE), Image.Resampling.NEAREST),
        threshold=mask_threshold,
    )
    meta = {
        "preprocess": False,
        "proc_size": [PROC_IMAGE_SIZE, PROC_IMAGE_SIZE],
        "source_original_size": list(source_image.size),
        "edit_original_size": list(edit_image.size),
        "mask_original_size": list(mask_image.size),
        "scale": 1.0,
        "crop_bbox_xyxy": [0, 0, int(source_image.width), int(source_image.height)],
        "mask_threshold": int(mask_threshold),
        "crop_rule": "disabled_resize_only",
    }
    return PreparedInputs(source=source, edit=edit, mask=mask, meta=meta)


def prepare_edit_condition_image(pipeline, image: Image.Image, preprocess: bool) -> PreparedEditCondition:
    if preprocess:
        edit = pipeline.preprocess_image(image)
        meta = {
            "preprocess": True,
            "edit_original_size": list(image.size),
            "edit_preprocessed_size": list(edit.size),
            "crop_rule": "trellis_pipeline_preprocess_image",
        }
        return PreparedEditCondition(edit=edit, meta=meta)

    meta = {
        "preprocess": False,
        "edit_original_size": list(image.size),
        "edit_preprocessed_size": list(image.size),
        "crop_rule": "disabled_raw_image",
    }
    return PreparedEditCondition(edit=image, meta=meta)
