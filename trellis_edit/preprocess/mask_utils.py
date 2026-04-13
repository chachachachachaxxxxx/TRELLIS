from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter


def extract_mask_channel(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        alpha = image.getchannel("A")
        alpha_np = np.asarray(alpha)
        if np.any(alpha_np > 0):
            rgb_np = np.asarray(image.convert("RGB"))
            if not np.any(rgb_np):
                return alpha
    return image.convert("L")


def binarize_mask_image(mask: Image.Image, threshold: int) -> Image.Image:
    mask_np = np.asarray(extract_mask_channel(mask))
    mask_np = (mask_np > int(threshold)).astype(np.uint8) * 255
    return Image.fromarray(mask_np, mode="L")


def build_blank_mask(size: Tuple[int, int]) -> Image.Image:
    return Image.new("L", size, color=0)


def build_auto_mask(
    source_image: Image.Image,
    edit_image: Image.Image,
    threshold: int,
    max_filter: int,
) -> Image.Image:
    src = np.asarray(source_image.convert("RGB"), dtype=np.int16)
    tgt = np.asarray(edit_image.convert("RGB"), dtype=np.int16)
    diff = np.max(np.abs(src - tgt), axis=-1)
    mask = (diff >= int(threshold)).astype(np.uint8) * 255
    mask_image = Image.fromarray(mask, mode="L")
    filter_size = max(1, int(max_filter))
    if filter_size % 2 == 0:
        filter_size += 1
    if filter_size > 1:
        mask_image = mask_image.filter(ImageFilter.MaxFilter(size=filter_size))
    return mask_image
