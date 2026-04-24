from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter


DEFAULT_BACKGROUND_RGB: Tuple[int, int, int] = (127, 127, 127)
DEFAULT_BACKGROUND_RGBA: Tuple[int, int, int, int] = (127, 127, 127, 255)


def load_prepared_rgb_image(
    image_path: str,
    image_size: tuple[int, int],
    *,
    background_rgba: tuple[int, int, int, int] = DEFAULT_BACKGROUND_RGBA,
) -> Image.Image:
    image = Image.open(image_path)
    if image.mode != "RGB":
        background = Image.new("RGBA", image.size, background_rgba)
        background.paste(image, (0, 0), image)
        image = background.convert("RGB")
    return image.resize((image_size[1], image_size[0]), Image.BILINEAR)


def _dilate_binary_mask(edit_region: np.ndarray, dilation_px: int) -> np.ndarray:
    if dilation_px <= 0:
        return edit_region
    kernel_size = dilation_px * 2 + 1
    binary_mask = Image.fromarray(edit_region.astype(np.uint8) * 255, mode="L")
    dilated = binary_mask.filter(ImageFilter.MaxFilter(size=kernel_size))
    return np.asarray(dilated, dtype=np.uint8) > 0


def build_include_mask(
    mask_path: str | None,
    image_size: tuple[int, int],
    *,
    dilation_px: int = 0,
    ignore_mask: bool = False,
) -> np.ndarray:
    if ignore_mask or not mask_path:
        return np.ones(image_size, dtype=np.float32)

    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((image_size[1], image_size[0]), Image.NEAREST)
    edit_region = np.asarray(mask, dtype=np.uint8) > 0
    dilated_region = _dilate_binary_mask(edit_region, max(int(dilation_px), 0))
    return (~dilated_region).astype(np.float32)


def apply_include_mask_to_image(
    image: Image.Image,
    include_mask: np.ndarray,
    *,
    fill_rgb: tuple[int, int, int] = DEFAULT_BACKGROUND_RGB,
) -> Image.Image:
    image_np = np.asarray(image, dtype=np.uint8).copy()
    masked_pixels = ~include_mask.astype(bool)
    image_np[masked_pixels] = np.asarray(fill_rgb, dtype=np.uint8)
    return Image.fromarray(image_np, mode="RGB")
