from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def save_patch_grid_preview(token_meta: dict, path: Path, upscale: int = 14) -> None:
    """Save patch grid preview image.

    Args:
        token_meta: Token metadata dictionary
        path: Output path
        upscale: Upscale factor for visualization
    """
    edited_grid = np.asarray(token_meta["edited_patch_grid"], dtype=np.uint8) * 255
    preview = Image.fromarray(edited_grid, mode="L").resize(
        (edited_grid.shape[1] * upscale, edited_grid.shape[0] * upscale),
        Image.Resampling.NEAREST,
    )
    preview.save(path)


def save_mask_overlay_preview(
    edit_image: Image.Image,
    mask_image: Image.Image,
    token_meta: dict,
    path: Path,
) -> None:
    """Save mask overlay preview with patch boundaries.

    Args:
        edit_image: Edit image
        mask_image: Mask image
        token_meta: Token metadata dictionary
        path: Output path
    """
    image_np = np.asarray(edit_image.convert("RGB")).astype(np.float32) / 255.0
    mask_np = np.asarray(mask_image.convert("L")).astype(np.float32) / 255.0
    overlay = image_np.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] + mask_np * 0.75, 0.0, 1.0)
    overlay[..., 1] *= 1.0 - mask_np * 0.4
    overlay[..., 2] *= 1.0 - mask_np * 0.4
    preview = Image.fromarray((overlay * 255).astype(np.uint8), mode="RGB").convert("RGBA")

    draw = ImageDraw.Draw(preview)
    edited_grid = np.asarray(token_meta["edited_patch_grid"], dtype=bool)
    patch_size = int(token_meta["patch_size"])
    for row in range(edited_grid.shape[0]):
        for col in range(edited_grid.shape[1]):
            if not edited_grid[row, col]:
                continue
            x0 = col * patch_size
            y0 = row * patch_size
            x1 = x0 + patch_size - 1
            y1 = y0 + patch_size - 1
            draw.rectangle((x0, y0, x1, y1), outline=(46, 196, 182, 255), width=2)

    preview.save(path)
