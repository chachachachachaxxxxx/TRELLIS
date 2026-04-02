#!/usr/bin/env python3
from __future__ import annotations
"""
TRELLIS Prompt-to-Prompt image editing example.

This script keeps TRELLIS core code untouched and injects Prompt-to-Prompt
cross-attention edits at runtime via monkey patching.

Mask semantics:
- DINOv2 patch tokens that overlap the mask are treated as edited tokens.
- Only unmasked patch tokens receive source attention injection.
- Prefix tokens ([CLS] / register tokens) are kept on the edit side.

The source image, edit image, and mask should describe the same aligned view.
By default the script applies a shared crop/resize using the union of source
foreground, edit foreground, and mask so the visual token grid stays aligned.

Usage examples:
  python example_image_prompt_to_prompt.py \
    --source-image assets/example_edit/2d_render.png \
    --edit-image assets/example_edit/2d_edit.png \
    --mask-image assets/example_edit/2d_mask.png \
    --case-name "cat_to_tiger"

  python example_image_prompt_to_prompt.py \
    --source-image assets/example_edit/2d_render.png \
    --edit-image assets/example_edit/2d_edit.png \
    --mask-image assets/example_edit/2d_mask.png \
    --inject-stages st \
    --ss-t-start 1.0 --ss-t-end 0.3

  python example_image_prompt_to_prompt.py \
    --source-image assets/example_edit/2d_render.png \
    --edit-image assets/example_edit/2d_edit.png \
    --mask-image assets/example_edit/2d_mask.png \
    --inject-stages slat \
    --slat-t-start 0.8 --slat-t-end 0.0
"""

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from output_layout import build_output_layout


def _peek_arg(flag: str, default: str = "") -> str:
    if flag not in sys.argv:
        return default
    idx = sys.argv.index(flag)
    if idx + 1 >= len(sys.argv):
        return default
    return sys.argv[idx + 1]


_attn_backend = _peek_arg("--attn-backend", "")
if _attn_backend:
    os.environ["ATTN_BACKEND"] = _attn_backend

os.environ.setdefault("SPCONV_ALGO", "native")

import numpy as np
import torch
from PIL import Image, ImageDraw


MultiHeadAttention = None
SparseMultiHeadAttention = None
SparseTensor = None
TrellisImageTo3DPipeline = None


PROC_IMAGE_SIZE = 518
EDIT_METHOD_NAME = "image_prompt_to_prompt"


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def configure_attention_backend(requested_backend: str) -> str:
    backend = requested_backend.strip().lower()
    if backend:
        if backend not in {"flash_attn", "xformers"}:
            raise RuntimeError(
                "TRELLIS image-to-3D requires sparse attention, so "
                "--attn-backend must be flash_attn or xformers."
            )
        if not module_available(backend):
            raise RuntimeError(
                f"Requested attention backend '{backend}' is not installed. "
                "Please install it or switch to the other supported backend."
            )
    else:
        if module_available("flash_attn"):
            backend = "flash_attn"
        elif module_available("xformers"):
            backend = "xformers"
        else:
            raise RuntimeError(
                "Neither flash_attn nor xformers is installed, but TRELLIS image-to-3D "
                "needs one of them for sparse attention."
            )

    os.environ["ATTN_BACKEND"] = backend
    os.environ.setdefault("SPARSE_ATTN_BACKEND", backend)
    return backend


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def slugify(text: str, max_len: int = 96) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^0-9a-zA-Z_\u4e00-\u9fff]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    if not text:
        text = "image_prompt_to_prompt_edit"
    return text[:max_len]


def tensor_signature(tensor: torch.Tensor, rows: int = 3, cols: int = 8) -> torch.Tensor:
    return tensor[:1, :rows, :cols].detach().float().cpu()


def signature_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.max(torch.abs(a - b)).item())


def resolve_patch_size(value) -> int:
    if isinstance(value, tuple):
        if len(value) != 2 or value[0] != value[1]:
            raise RuntimeError(f"Unsupported patch size: {value}")
        return int(value[0])
    return int(value)


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


def extract_foreground_rgba(
    image: Image.Image,
    pipeline,
    scale: float,
) -> Image.Image:
    image = scale_image(image, scale, Image.Resampling.LANCZOS)
    if has_useful_alpha(image):
        return image.convert("RGBA")

    if not module_available("rembg"):
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


def mask_to_patch_selection(
    mask: Image.Image,
    patch_size: int,
    coverage_threshold: float,
) -> Tuple[np.ndarray, List[int], np.ndarray]:
    mask_np = np.asarray(mask.convert("L")) > 0
    if mask_np.shape[0] % patch_size != 0 or mask_np.shape[1] % patch_size != 0:
        raise RuntimeError(
            f"Mask size {mask_np.shape[::-1]} is not divisible by patch size {patch_size}."
        )

    grid_h = mask_np.shape[0] // patch_size
    grid_w = mask_np.shape[1] // patch_size
    edited_grid = np.zeros((grid_h, grid_w), dtype=bool)
    coverage_grid = np.zeros((grid_h, grid_w), dtype=np.float32)
    edited_linear_indices: List[int] = []

    for row in range(grid_h):
        for col in range(grid_w):
            y0 = row * patch_size
            y1 = (row + 1) * patch_size
            x0 = col * patch_size
            x1 = (col + 1) * patch_size
            patch = mask_np[y0:y1, x0:x1]
            coverage = float(patch.mean())
            coverage_grid[row, col] = coverage
            if coverage_threshold <= 0.0:
                is_edited = bool(patch.any())
            else:
                is_edited = bool(coverage >= coverage_threshold)
            if is_edited:
                edited_grid[row, col] = True
                edited_linear_indices.append(row * grid_w + col)

    return edited_grid, edited_linear_indices, coverage_grid


def build_image_token_metadata(
    cond: torch.Tensor,
    mask: Image.Image,
    patch_size: int,
    patch_coverage_threshold: float,
) -> dict:
    total_tokens = int(cond.shape[1])
    edited_patch_grid, edited_patch_linear_indices, coverage_grid = mask_to_patch_selection(
        mask=mask,
        patch_size=patch_size,
        coverage_threshold=patch_coverage_threshold,
    )
    grid_h, grid_w = edited_patch_grid.shape
    patch_token_count = grid_h * grid_w
    prefix_token_count = total_tokens - patch_token_count
    if prefix_token_count < 0:
        raise RuntimeError(
            f"Unexpected token layout: total_tokens={total_tokens}, patch_token_count={patch_token_count}"
        )

    patch_labels = [f"patch_r{row:02d}_c{col:02d}" for row in range(grid_h) for col in range(grid_w)]
    token_labels = ["[CLS]"] + [f"[REG{idx}]" for idx in range(max(prefix_token_count - 1, 0))] + patch_labels
    if len(token_labels) != total_tokens:
        token_labels = [f"token_{idx:04d}" for idx in range(total_tokens)]

    edited_patch_set = set(edited_patch_linear_indices)
    keep_patch_linear_indices = [idx for idx in range(patch_token_count) if idx not in edited_patch_set]
    edited_token_indices = [prefix_token_count + idx for idx in edited_patch_linear_indices]
    keep_token_indices = [prefix_token_count + idx for idx in keep_patch_linear_indices]

    selection_rule = (
        "Any patch with any mask overlap is counted as an edited visual token."
        if patch_coverage_threshold <= 0.0
        else f"Any patch with mask coverage >= {patch_coverage_threshold:.4f} is counted as an edited visual token."
    )

    return {
        "proc_size": [int(mask.width), int(mask.height)],
        "patch_grid_size": [int(grid_h), int(grid_w)],
        "patch_size": int(patch_size),
        "total_tokens": total_tokens,
        "prefix_token_count": int(prefix_token_count),
        "patch_token_count": int(patch_token_count),
        "special_token_indices": list(range(prefix_token_count)),
        "source_keep_indices": keep_token_indices,
        "edit_keep_indices": keep_token_indices,
        "keep_patch_linear_indices": keep_patch_linear_indices,
        "edited_patch_linear_indices": edited_patch_linear_indices,
        "edited_patch_grid": edited_patch_grid.astype(np.uint8).tolist(),
        "edited_patch_coverage_grid": np.round(coverage_grid, 6).tolist(),
        "edited_token_indices": edited_token_indices,
        "keep_token_labels": [token_labels[idx] for idx in keep_token_indices],
        "edited_token_labels": [token_labels[idx] for idx in edited_token_indices],
        "token_labels": token_labels,
        "selection_rule": selection_rule,
        "patch_coverage_threshold": float(patch_coverage_threshold),
    }


def save_patch_grid_preview(token_meta: dict, path: Path, upscale: int = 14) -> None:
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


def normalize_stage_name(name: str) -> str:
    key = name.strip().lower()
    if key in {"ss", "st", "sparse_structure", "sparse-structure"}:
        return "sparse_structure"
    if key in {"slat", "structured_latent", "structured-latent"}:
        return "slat"
    raise ValueError(f"Unknown stage name: {name}")


def parse_stage_list(text: str) -> List[str]:
    if not text.strip():
        return []
    raw = [item.strip() for item in text.split(",") if item.strip()]
    if not raw:
        return []
    if len(raw) == 1 and raw[0].lower() == "none":
        return []
    normalized = []
    for item in raw:
        stage = normalize_stage_name(item)
        if stage not in normalized:
            normalized.append(stage)
    return normalized


@dataclass(frozen=True)
class StageConfig:
    name: str
    enabled: bool
    t_start: float
    t_end: float
    strength: float

    def contains(self, t_norm: float) -> bool:
        lo = min(self.t_start, self.t_end)
        hi = max(self.t_start, self.t_end)
        return self.enabled and lo <= t_norm <= hi and self.strength > 0.0


class ImagePromptToPromptEditor:
    def __init__(
        self,
        source_cond: torch.Tensor,
        edit_cond: torch.Tensor,
        neg_cond: torch.Tensor,
        token_meta: dict,
        stage_configs: Dict[str, StageConfig],
        query_chunk: int = 1024,
    ):
        self.source_cond = source_cond
        self.edit_cond_signature = tensor_signature(edit_cond)
        self.neg_cond_signature = tensor_signature(neg_cond)
        self.token_meta = token_meta
        self.stage_configs = stage_configs
        self.query_chunk = max(1, int(query_chunk))

        self.current_forward_context: Optional[dict] = None
        self._restore_stack: List[Tuple[object, str, object]] = []
        self._index_cache: Dict[Tuple[str, str], Tuple[torch.Tensor, torch.Tensor]] = {}

    def prepare_stage(self, stage_name: str, model: torch.nn.Module) -> None:
        config = self.stage_configs[stage_name]
        if not config.enabled:
            return
        self._patch_model_forward(stage_name, model)
        self._patch_cross_modules(stage_name, model)

    def restore(self) -> None:
        while self._restore_stack:
            obj, attr, original = self._restore_stack.pop()
            setattr(obj, attr, original)

    def _patch_model_forward(self, stage_name: str, model: torch.nn.Module) -> None:
        original_forward = model.forward
        self._restore_stack.append((model, "forward", original_forward))
        editor = self

        def wrapped_forward(model_self, x, t, cond):
            pass_kind = editor._infer_pass_kind(cond)
            t_value = float(t[0].detach().float().cpu().item()) if torch.is_tensor(t) else float(t)
            editor.current_forward_context = {
                "stage": stage_name,
                "pass_kind": pass_kind,
                "t_raw": t_value,
                "t_norm": t_value / 1000.0,
            }
            try:
                return original_forward(x, t, cond)
            finally:
                editor.current_forward_context = None

        model.forward = types.MethodType(wrapped_forward, model)

    def _infer_pass_kind(self, cond: torch.Tensor) -> str:
        sig = tensor_signature(cond)
        if signature_distance(sig, self.edit_cond_signature) <= signature_distance(sig, self.neg_cond_signature):
            return "cond"
        return "neg"

    def _patch_cross_modules(self, stage_name: str, model: torch.nn.Module) -> None:
        for _, module in model.named_modules():
            if isinstance(module, MultiHeadAttention) and getattr(module, "_type", None) == "cross":
                self._patch_dense_cross(stage_name, module)
            elif isinstance(module, SparseMultiHeadAttention) and getattr(module, "_type", None) == "cross":
                self._patch_sparse_cross(stage_name, module)

    def _patch_dense_cross(self, stage_name: str, module: MultiHeadAttention) -> None:
        original_forward = module.forward
        self._restore_stack.append((module, "forward", original_forward))
        editor = self

        def wrapped_forward(attn_self, x, context=None, indices=None):
            strength = editor._active_strength(stage_name)
            if strength <= 0.0 or context is None:
                return original_forward(x, context, indices)

            batch_size, num_queries, _ = x.shape
            q = attn_self.to_q(x).reshape(batch_size, num_queries, attn_self.num_heads, -1)
            kv_edit = attn_self.to_kv(context).reshape(batch_size, context.shape[1], 2, attn_self.num_heads, -1)
            k_edit, v_edit = kv_edit.unbind(dim=2)

            source_context = editor._source_context_for_batch(
                batch_size=batch_size,
                device=context.device,
                dtype=context.dtype,
            )
            kv_src = attn_self.to_kv(source_context).reshape(
                batch_size, source_context.shape[1], 2, attn_self.num_heads, -1
            )
            k_src, _ = kv_src.unbind(dim=2)

            if attn_self.qk_rms_norm:
                q = attn_self.q_rms_norm(q)
                k_edit = attn_self.k_rms_norm(k_edit)
                k_src = attn_self.k_rms_norm(k_src)

            h = editor._chunked_prompt_to_prompt_attention(q, k_src, k_edit, v_edit, strength)
            h = h.reshape(batch_size, num_queries, -1)
            h = attn_self.to_out(h)
            return h

        module.forward = types.MethodType(wrapped_forward, module)

    def _patch_sparse_cross(self, stage_name: str, module: SparseMultiHeadAttention) -> None:
        original_forward = module.forward
        self._restore_stack.append((module, "forward", original_forward))
        editor = self

        def wrapped_forward(attn_self, x, context=None):
            strength = editor._active_strength(stage_name)
            if strength <= 0.0 or context is None:
                return original_forward(x, context)

            q = attn_self._linear(attn_self.to_q, x)
            q = attn_self._reshape_chs(q, (attn_self.num_heads, -1))

            kv_edit = attn_self._linear(attn_self.to_kv, context)
            kv_edit = attn_self._fused_pre(kv_edit, num_fused=2)
            k_edit, v_edit = kv_edit.unbind(dim=2)

            source_context = editor._source_context_for_batch(
                batch_size=context.shape[0],
                device=context.device,
                dtype=context.dtype,
            )
            kv_src = attn_self._linear(attn_self.to_kv, source_context)
            kv_src = attn_self._fused_pre(kv_src, num_fused=2)
            k_src, _ = kv_src.unbind(dim=2)

            if attn_self.qk_rms_norm:
                q = attn_self.q_rms_norm(q)
                k_edit = attn_self.k_rms_norm(k_edit)
                k_src = attn_self.k_rms_norm(k_src)

            h = editor._chunked_sparse_prompt_to_prompt_attention(q, k_src, k_edit, v_edit, strength)
            h = attn_self._reshape_chs(h, (-1,))
            h = attn_self._linear(attn_self.to_out, h)
            return h

        module.forward = types.MethodType(wrapped_forward, module)

    def _active_strength(self, stage_name: str) -> float:
        ctx = self.current_forward_context
        if ctx is None:
            return 0.0
        if ctx["stage"] != stage_name or ctx["pass_kind"] != "cond":
            return 0.0
        config = self.stage_configs[stage_name]
        if not config.contains(ctx["t_norm"]):
            return 0.0
        return float(config.strength)

    def _source_context_for_batch(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        source_context = self.source_cond.to(device=device, dtype=dtype)
        if source_context.shape[0] == 1 and batch_size > 1:
            source_context = source_context.repeat(batch_size, 1, 1)
        return source_context

    def _token_indices(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (str(device), str(device.index))
        if key not in self._index_cache:
            src_idx = torch.tensor(
                self.token_meta["source_keep_indices"],
                device=device,
                dtype=torch.long,
            )
            edit_idx = torch.tensor(
                self.token_meta["edit_keep_indices"],
                device=device,
                dtype=torch.long,
            )
            self._index_cache[key] = (src_idx, edit_idx)
        return self._index_cache[key]

    def _mix_attention_maps(
        self,
        attn_src: torch.Tensor,
        attn_edit: torch.Tensor,
        strength: float,
    ) -> torch.Tensor:
        src_idx, edit_idx = self._token_indices(attn_edit.device)
        if edit_idx.numel() == 0:
            return attn_edit

        mixed = attn_edit.clone()
        if strength >= 1.0:
            replacement = attn_src.index_select(dim=-1, index=src_idx)
        else:
            src_selected = attn_src.index_select(dim=-1, index=src_idx)
            edit_selected = attn_edit.index_select(dim=-1, index=edit_idx)
            replacement = strength * src_selected + (1.0 - strength) * edit_selected
        mixed.index_copy_(dim=-1, index=edit_idx, source=replacement)
        return mixed

    def _chunked_prompt_to_prompt_attention(
        self,
        q: torch.Tensor,
        k_src: torch.Tensor,
        k_edit: torch.Tensor,
        v_edit: torch.Tensor,
        strength: float,
    ) -> torch.Tensor:
        batch_size, num_queries, _, head_dim = q.shape
        scale = 1.0 / math.sqrt(head_dim)
        k_src_t = k_src.permute(0, 2, 3, 1).float()
        k_edit_t = k_edit.permute(0, 2, 3, 1).float()
        v_edit_h = v_edit.permute(0, 2, 1, 3).float()
        out_chunks = []

        for start in range(0, num_queries, self.query_chunk):
            end = min(start + self.query_chunk, num_queries)
            q_chunk = q[:, start:end].permute(0, 2, 1, 3).float()

            scores_src = torch.matmul(q_chunk, k_src_t) * scale
            scores_edit = torch.matmul(q_chunk, k_edit_t) * scale
            scores_src = scores_src - scores_src.amax(dim=-1, keepdim=True)
            scores_edit = scores_edit - scores_edit.amax(dim=-1, keepdim=True)

            attn_src = torch.softmax(scores_src, dim=-1)
            attn_edit = torch.softmax(scores_edit, dim=-1)
            attn_mix = self._mix_attention_maps(attn_src, attn_edit, strength)
            out_chunk = torch.matmul(attn_mix, v_edit_h)
            out_chunks.append(out_chunk.permute(0, 2, 1, 3).to(q.dtype))

        return torch.cat(out_chunks, dim=1)

    def _chunked_sparse_prompt_to_prompt_attention(
        self,
        q: SparseTensor,
        k_src: torch.Tensor,
        k_edit: torch.Tensor,
        v_edit: torch.Tensor,
        strength: float,
    ) -> SparseTensor:
        new_feats = torch.empty_like(q.feats)
        for batch_idx in range(q.shape[0]):
            q_slice = q.feats[q.layout[batch_idx]].unsqueeze(0)
            out_slice = self._chunked_prompt_to_prompt_attention(
                q=q_slice,
                k_src=k_src[batch_idx : batch_idx + 1],
                k_edit=k_edit[batch_idx : batch_idx + 1],
                v_edit=v_edit[batch_idx : batch_idx + 1],
                strength=strength,
            )
            new_feats[q.layout[batch_idx]] = out_slice[0]
        return q.replace(new_feats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prompt-to-Prompt editing for TRELLIS image-to-3D.")
    parser.add_argument("--model", default="microsoft/TRELLIS-image-large", help="Pipeline checkpoint or HF repo.")
    parser.add_argument("--source-image", required=True, help="Original source image path.")
    parser.add_argument("--edit-image", required=True, help="Edited target image path.")
    parser.add_argument("--mask-image", required=True, help="Binary or grayscale edit mask path.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to generate.")
    parser.add_argument(
        "--inject-stages",
        default="st,slat",
        help="Comma-separated stages to inject: st/ss/sparse_structure, slat, or none.",
    )
    parser.add_argument("--ss-t-start", type=float, default=1.0, help="Sparse-structure injection start t in [0, 1].")
    parser.add_argument("--ss-t-end", type=float, default=0.5, help="Sparse-structure injection end t in [0, 1].")
    parser.add_argument("--slat-t-start", type=float, default=1.0, help="SLat injection start t in [0, 1].")
    parser.add_argument("--slat-t-end", type=float, default=0.5, help="SLat injection end t in [0, 1].")
    parser.add_argument("--ss-strength", type=float, default=1.0, help="Sparse-structure replacement strength.")
    parser.add_argument("--slat-strength", type=float, default=1.0, help="SLat replacement strength.")
    parser.add_argument("--ss-steps", type=int, default=None, help="Override sparse-structure sampling steps.")
    parser.add_argument("--slat-steps", type=int, default=None, help="Override SLat sampling steps.")
    parser.add_argument("--ss-cfg", type=float, default=None, help="Override sparse-structure CFG strength.")
    parser.add_argument("--slat-cfg", type=float, default=None, help="Override SLat CFG strength.")
    parser.add_argument("--query-chunk", type=int, default=1024, help="Query chunk size for patched attention.")
    parser.add_argument("--mask-threshold", type=int, default=127, help="Mask pixel threshold in [0, 255].")
    parser.add_argument(
        "--patch-coverage-threshold",
        type=float,
        default=0.0,
        help="Minimum mask coverage ratio for a patch token to count as edited. 0 means any overlap.",
    )
    parser.set_defaults(preprocess=True)
    parser.add_argument(
        "--preprocess",
        dest="preprocess",
        action="store_true",
        help="Apply a shared crop/resize based on source foreground, edit foreground, and mask.",
    )
    parser.add_argument(
        "--no-preprocess",
        dest="preprocess",
        action="store_false",
        help="Assume the three inputs are already aligned and only resize them to the DINO input size.",
    )
    parser.add_argument("--skip-source", action="store_true", help="Skip generating the original source model.")
    parser.add_argument("--case-name", default="", help="Optional output case name.")
    parser.add_argument("--skip-render", action="store_true", help="Skip rendering mp4 previews.")
    parser.add_argument("--skip-glb", action="store_true", help="Skip exporting GLB.")
    parser.add_argument("--skip-ply", action="store_true", help="Skip exporting gaussian PLY.")
    parser.add_argument(
        "--attn-backend",
        default=_attn_backend or "",
        help="Attention backend override: flash_attn or xformers.",
    )
    return parser.parse_args()


def save_outputs(outputs: dict, out_dir: Path, skip_render: bool, skip_glb: bool, skip_ply: bool) -> None:
    imageio = None
    render_utils = None
    postprocessing_utils = None
    if not skip_render:
        import imageio as _imageio
        from trellis.utils import render_utils as _render_utils

        imageio = _imageio
        render_utils = _render_utils
    if not skip_glb:
        from trellis.utils import postprocessing_utils as _postprocessing_utils

        postprocessing_utils = _postprocessing_utils

    num_samples = 0
    for key in ("mesh", "gaussian", "radiance_field"):
        if key in outputs:
            num_samples = max(num_samples, len(outputs[key]))
    num_samples = max(num_samples, 1)

    for sample_idx in range(num_samples):
        prefix = f"sample_{sample_idx:02d}"

        if not skip_render:
            if "gaussian" in outputs:
                video = render_utils.render_video(outputs["gaussian"][sample_idx])["color"]
                imageio.mimsave(str(out_dir / f"{prefix}_gs.mp4"), video, fps=30)
            if "radiance_field" in outputs:
                video = render_utils.render_video(outputs["radiance_field"][sample_idx])["color"]
                imageio.mimsave(str(out_dir / f"{prefix}_rf.mp4"), video, fps=30)
            if "mesh" in outputs:
                video = render_utils.render_video(outputs["mesh"][sample_idx])["normal"]
                imageio.mimsave(str(out_dir / f"{prefix}_mesh.mp4"), video, fps=30)

        if not skip_glb and "gaussian" in outputs and "mesh" in outputs:
            glb = postprocessing_utils.to_glb(
                outputs["gaussian"][sample_idx],
                outputs["mesh"][sample_idx],
                simplify=0.95,
                texture_size=1024,
            )
            glb.export(str(out_dir / f"{prefix}.glb"))

        if not skip_ply and "gaussian" in outputs:
            outputs["gaussian"][sample_idx].save_ply(str(out_dir / f"{prefix}.ply"))


def main() -> int:
    args = parse_args()
    backend = configure_attention_backend(args.attn_backend)

    global MultiHeadAttention
    global SparseMultiHeadAttention
    global SparseTensor
    global TrellisImageTo3DPipeline

    from trellis.modules.attention.modules import MultiHeadAttention as _MultiHeadAttention
    from trellis.modules.sparse.attention.modules import SparseMultiHeadAttention as _SparseMultiHeadAttention
    from trellis.modules.sparse.basic import SparseTensor as _SparseTensor
    from trellis.pipelines import TrellisImageTo3DPipeline as _TrellisImageTo3DPipeline

    MultiHeadAttention = _MultiHeadAttention
    SparseMultiHeadAttention = _SparseMultiHeadAttention
    SparseTensor = _SparseTensor
    TrellisImageTo3DPipeline = _TrellisImageTo3DPipeline

    inject_stages = parse_stage_list(args.inject_stages)
    stage_configs = {
        "sparse_structure": StageConfig(
            name="sparse_structure",
            enabled="sparse_structure" in inject_stages,
            t_start=float(args.ss_t_start),
            t_end=float(args.ss_t_end),
            strength=float(args.ss_strength),
        ),
        "slat": StageConfig(
            name="slat",
            enabled="slat" in inject_stages,
            t_start=float(args.slat_t_start),
            t_end=float(args.slat_t_end),
            strength=float(args.slat_strength),
        ),
    }

    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()

    source_path = Path(args.source_image)
    edit_path = Path(args.edit_image)
    mask_path = Path(args.mask_image)
    if not source_path.is_file():
        raise RuntimeError(f"source-image does not exist: {source_path}")
    if not edit_path.is_file():
        raise RuntimeError(f"edit-image does not exist: {edit_path}")
    if not mask_path.is_file():
        raise RuntimeError(f"mask-image does not exist: {mask_path}")

    source_image = Image.open(source_path)
    edit_image = Image.open(edit_path)
    mask_image = Image.open(mask_path)

    prepared = prepare_aligned_inputs(
        source_image=source_image,
        edit_image=edit_image,
        mask_image=mask_image,
        pipeline=pipeline,
        preprocess=bool(args.preprocess),
        mask_threshold=int(args.mask_threshold),
    )

    source_cond_dict = pipeline.get_cond([prepared.source])
    edit_cond = pipeline.get_cond([prepared.edit])

    patch_size = resolve_patch_size(pipeline.models["image_cond_model"].patch_size)
    token_meta = build_image_token_metadata(
        cond=edit_cond["cond"],
        mask=prepared.mask,
        patch_size=patch_size,
        patch_coverage_threshold=float(args.patch_coverage_threshold),
    )
    if not token_meta["edited_token_indices"]:
        print("Warning: the mask selected no visual patch tokens; all patch tokens will be treated as unchanged.")
    if not token_meta["source_keep_indices"]:
        print("Warning: the mask covers all visual patch tokens; injection will behave like a normal edit run.")

    editor = ImagePromptToPromptEditor(
        source_cond=source_cond_dict["cond"],
        edit_cond=edit_cond["cond"],
        neg_cond=edit_cond["neg_cond"],
        token_meta=token_meta,
        stage_configs=stage_configs,
        query_chunk=args.query_chunk,
    )

    ss_params = {}
    if args.ss_steps is not None:
        ss_params["steps"] = args.ss_steps
    if args.ss_cfg is not None:
        ss_params["cfg_strength"] = args.ss_cfg

    slat_params = {}
    if args.slat_steps is not None:
        slat_params["steps"] = args.slat_steps
    if args.slat_cfg is not None:
        slat_params["cfg_strength"] = args.slat_cfg

    case_name = args.case_name.strip()
    if not case_name:
        case_name = slugify(f"{source_path.stem}_to_{edit_path.stem}")
    output_layout = build_output_layout(EDIT_METHOD_NAME, case_name)
    out_dir = ensure_dir(output_layout.edit_dir)
    source_out_dir = output_layout.source_original_dir

    save_json(
        out_dir / "config.json",
        {
            "method_name": EDIT_METHOD_NAME,
            "case_name": case_name,
            "model": args.model,
            "attn_backend": backend,
            "output_root_dir": str(output_layout.root_dir),
            "output_case_dir": str(output_layout.case_dir),
            "output_edit_dir": str(out_dir),
            "output_source_dir": str(source_out_dir),
            "source_image": str(source_path),
            "edit_image": str(edit_path),
            "mask_image": str(mask_path),
            "seed": args.seed,
            "num_samples": args.num_samples,
            "inject_stages": inject_stages,
            "query_chunk": args.query_chunk,
            "mask_threshold": args.mask_threshold,
            "patch_coverage_threshold": args.patch_coverage_threshold,
            "preprocess": args.preprocess,
            "stage_configs": {
                name: {
                    "enabled": config.enabled,
                    "t_start": config.t_start,
                    "t_end": config.t_end,
                    "strength": config.strength,
                }
                for name, config in stage_configs.items()
            },
            "sparse_structure_sampler_params": ss_params,
            "slat_sampler_params": slat_params,
        },
    )
    save_json(out_dir / "input_preprocess.json", prepared.meta)
    save_json(out_dir / "token_metadata.json", token_meta)

    prepared.source.save(out_dir / "source_preprocessed.png")
    prepared.edit.save(out_dir / "edit_preprocessed.png")
    prepared.mask.save(out_dir / "mask_preprocessed.png")
    save_mask_overlay_preview(
        edit_image=prepared.edit,
        mask_image=prepared.mask,
        token_meta=token_meta,
        path=out_dir / "mask_token_overlay.png",
    )
    save_patch_grid_preview(token_meta, out_dir / "edited_patch_grid.png")

    source_outputs = None
    if not args.skip_source:
        torch.manual_seed(args.seed)
        source_coords = pipeline.sample_sparse_structure(
            source_cond_dict,
            num_samples=args.num_samples,
            sampler_params=ss_params,
        )
        source_slat = pipeline.sample_slat(
            source_cond_dict,
            source_coords,
            sampler_params=slat_params,
        )
        source_outputs = pipeline.decode_slat(source_slat, ["mesh", "gaussian", "radiance_field"])
        del source_coords, source_slat

    try:
        editor.prepare_stage("sparse_structure", pipeline.models["sparse_structure_flow_model"])
        editor.prepare_stage("slat", pipeline.models["slat_flow_model"])

        torch.manual_seed(args.seed)
        coords = pipeline.sample_sparse_structure(
            edit_cond,
            num_samples=args.num_samples,
            sampler_params=ss_params,
        )
        slat = pipeline.sample_slat(
            edit_cond,
            coords,
            sampler_params=slat_params,
        )
        outputs = pipeline.decode_slat(slat, ["mesh", "gaussian", "radiance_field"])
    finally:
        editor.restore()

    if source_outputs is not None:
        ensure_dir(source_out_dir)
        save_outputs(
            outputs=source_outputs,
            out_dir=source_out_dir,
            skip_render=args.skip_render,
            skip_glb=args.skip_glb,
            skip_ply=args.skip_ply,
        )
        print(f"Saved source original results to: {source_out_dir}")

    save_outputs(
        outputs=outputs,
        out_dir=out_dir,
        skip_render=args.skip_render,
        skip_glb=args.skip_glb,
        skip_ply=args.skip_ply,
    )

    print(f"Saved image Prompt-to-Prompt edit results to: {out_dir}")
    print(f"Edited patch tokens: {len(token_meta['edited_token_indices'])}")
    print(f"Kept patch tokens: {len(token_meta['source_keep_indices'])}")
    print(f"Injected stages: {inject_stages if inject_stages else ['none']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
