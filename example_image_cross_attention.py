#!/usr/bin/env python3
"""
在 TRELLIS 图像生成 3D 的流程中，追踪并可视化图像 patch token 与空间 token 的 cross-attention。

默认会读取 `assets/example_edit/` 目录，自动把 `*mask*` 识别为掩码图，其余图片视为条件图像，
并对每张条件图像分别运行一次 image-to-3D 推理与 cross-attention tracing。

重点规则：
- 掩码到 visual token 的映射使用“任意重叠即选中”。
- 只要某个 patch 内有任意一个掩码像素，该 patch 对应的整个 visual token 都算作关注 token。
- 导出的 3D 激活图展示的是这些被掩码命中的 visual token 在 cross-attention 中对应的 3D 空间区域。

运行示例：
  python example_image_cross_attention.py
  python example_image_cross_attention.py --input-dir assets/example_edit
  python example_image_cross_attention.py --images assets/example_edit/2d_render.png,assets/example_edit/2d_edit.png --mask assets/example_edit/2d_mask.png

输出目录：
  outputs/image_cross_attention/<case_name>/
"""

import os
import sys


def _peek_arg(flag: str, default: str = "") -> str:
    if flag in sys.argv:
        idx = sys.argv.index(flag)
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return default


os.environ.setdefault("SPCONV_ALGO", "native")
_attn_backend = _peek_arg("--attn-backend", "")
if _attn_backend:
    os.environ["ATTN_BACKEND"] = _attn_backend

import argparse
import importlib.util
import json
import math
import re
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from PIL import Image

from output_layout import build_output_layout
if importlib.util.find_spec("rembg") is None:
    rembg_stub = types.ModuleType("rembg")

    def _stub_new_session(*args, **kwargs):
        return None

    def _stub_remove(*args, **kwargs):
        raise RuntimeError(
            "rembg is not installed in the current environment. "
            "Please provide RGBA images or use a reference image with alpha so the script can reuse that alpha mask."
        )

    rembg_stub.new_session = _stub_new_session
    rembg_stub.remove = _stub_remove
    sys.modules["rembg"] = rembg_stub

from trellis.modules.attention.full_attn import scaled_dot_product_attention
from trellis.modules.attention.modules import MultiHeadAttention
from trellis.modules.sparse.attention.full_attn import sparse_scaled_dot_product_attention
from trellis.modules.sparse.attention.modules import SparseMultiHeadAttention
from trellis.modules.sparse.basic import SparseTensor
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.pipelines import samplers as trellis_samplers
from trellis import models as trellis_models


PROC_IMAGE_SIZE = 518
DINO_PATCH_SIZE = 14


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def sanitize_name(text: str, max_len: int = 64) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^0-9a-zA-Z_\u4e00-\u9fff]+", "_", text)
    text = text.strip("_")
    if not text:
        text = "case"
    return text[:max_len]


def tensor_signature(tensor: torch.Tensor, rows: int = 3, cols: int = 8) -> np.ndarray:
    return tensor[:1, :rows, :cols].detach().float().cpu().numpy()


def select_key_indices(total: int, max_items: int = 3) -> List[int]:
    if total <= 0:
        return []
    if total <= max_items:
        return list(range(total))
    if max_items <= 1:
        return [total - 1]
    indices = sorted(
        {
            int(round(i * (total - 1) / (max_items - 1)))
            for i in range(max_items)
        }
    )
    return indices


def dense_query_coords(resolution: int, patch_size: int, batch_size: int) -> List[np.ndarray]:
    side = resolution // patch_size
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(side),
            torch.arange(side),
            torch.arange(side),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3)
    coords_np = coords.cpu().numpy().astype(np.int16)
    return [coords_np.copy() for _ in range(batch_size)]


def compute_headmean_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    query_chunk: int,
    keep_map: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    q = q.detach()
    k = k.detach()
    num_queries = q.shape[0]
    num_tokens = k.shape[0]
    head_dim = q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)

    k_float = k.float()
    token_sum = np.zeros((num_tokens,), dtype=np.float64)
    map_chunks: List[np.ndarray] = []

    for start in range(0, num_queries, query_chunk):
        end = min(start + query_chunk, num_queries)
        q_chunk = q[start:end].float()
        scores = torch.einsum("qhd,khd->qhk", q_chunk, k_float) * scale
        attn = torch.softmax(scores, dim=-1)
        attn_mean = attn.mean(dim=1)
        token_sum += attn_mean.sum(dim=0).cpu().numpy()
        if keep_map:
            map_chunks.append(attn_mean.cpu().to(torch.float16).numpy())

    attn_map = np.concatenate(map_chunks, axis=0) if keep_map else None
    return token_sum, attn_map


def compute_attention_vmax(values_list: List[np.ndarray], percentile: float = 99.5) -> float:
    flattened = []
    for values in values_list:
        array = np.asarray(values).reshape(-1)
        if array.size > 0:
            flattened.append(array)
    if not flattened:
        return 1.0
    merged = np.concatenate(flattened, axis=0)
    return max(float(np.percentile(merged, percentile)), 1e-6)


def serialize_coords(coords: np.ndarray) -> dict:
    coords = np.asarray(coords, dtype=np.int16).reshape(-1, 3)
    return {
        "x": coords[:, 0].astype(int).tolist(),
        "y": coords[:, 1].astype(int).tolist(),
        "z": coords[:, 2].astype(int).tolist(),
    }


def serialize_values(values: np.ndarray, decimals: int = 7) -> List[float]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    return np.round(values, decimals=decimals).tolist()


def summarize_values(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def build_interactive_stage_payload(
    stage_name: str,
    state: dict,
    selected_token_indices: Sequence[int],
    avg_selected_values: Optional[np.ndarray],
) -> Optional[dict]:
    raw_views = []
    if avg_selected_values is not None and state["overview_coords"] is not None:
        raw_views.append(
            {
                "id": "average_over_steps",
                "label": f"Average over steps · block {state['overview_block_index']:02d}",
                "group": "overview",
                "step_index": None,
                "block_index": int(state["overview_block_index"]),
                "timestep": None,
                "coords": np.asarray(state["overview_coords"], dtype=np.int16).reshape(-1, 3),
                "values": np.asarray(avg_selected_values, dtype=np.float32).reshape(-1),
            }
        )

    for (step_index, block_index), item in sorted(state["selected_maps"].items()):
        values = item["attn_map"][:, selected_token_indices].sum(axis=1).astype(np.float32)
        raw_views.append(
            {
                "id": f"step_{step_index:02d}_block_{block_index:02d}",
                "label": f"Step {step_index:02d} · Block {block_index:02d} · t={item['timestep']:.3g}",
                "group": "captured_maps",
                "step_index": int(step_index),
                "block_index": int(block_index),
                "timestep": float(item["timestep"]),
                "coords": np.asarray(item["coords"], dtype=np.int16).reshape(-1, 3),
                "values": values,
            }
        )

    if not raw_views:
        return None

    shared_coords = raw_views[0]["coords"]
    if not all(
        view["coords"].shape == shared_coords.shape and np.array_equal(view["coords"], shared_coords)
        for view in raw_views[1:]
    ):
        shared_coords = None

    global_max = max(float(np.max(view["values"])) for view in raw_views if view["values"].size > 0)
    global_vmax = compute_attention_vmax([view["values"] for view in raw_views])

    default_view = raw_views[0]
    default_threshold = summarize_values(default_view["values"])["p95"]
    payload = {
        "stage": stage_name,
        "query_type": state["query_type"],
        "selected_patch_count": len(selected_token_indices),
        "point_count": int(raw_views[0]["coords"].shape[0]),
        "global_max": float(max(global_max, 1e-6)),
        "global_vmax": float(max(global_vmax, 1e-6)),
        "default_view_id": str(default_view["id"]),
        "default_threshold": float(max(default_threshold, 0.0)),
        "shared_coords": serialize_coords(shared_coords) if shared_coords is not None else None,
        "views": [],
    }

    for view in raw_views:
        view_payload = {
            "id": view["id"],
            "label": view["label"],
            "group": view["group"],
            "step_index": view["step_index"],
            "block_index": view["block_index"],
            "timestep": view["timestep"],
            "stats": summarize_values(view["values"]),
            "values": serialize_values(view["values"]),
        }
        if shared_coords is None:
            view_payload["coords"] = serialize_coords(view["coords"])
        payload["views"].append(view_payload)

    return payload


def plot_spatial_attention_subplot(
    ax,
    coords: np.ndarray,
    values: np.ndarray,
    title: str,
    vmax: float,
    point_size: float = 5.0,
) -> None:
    coords = np.asarray(coords)
    values = np.asarray(values).reshape(-1)
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=values,
        s=point_size,
        alpha=0.92,
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
    )
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    spans = np.ptp(coords, axis=0).astype(np.float32) if coords.size > 0 else np.ones((3,), dtype=np.float32)
    ax.set_box_aspect(tuple(np.maximum(spans, 1.0)))
    ax.view_init(elev=28, azim=42)
    ax.grid(False)
    try:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
    except AttributeError:
        pass


def save_spatial_panels(
    panels: List[Tuple[str, np.ndarray, np.ndarray]],
    path: Path,
    title: str,
) -> None:
    if not panels:
        return
    fig = plt.figure(figsize=(6 * len(panels), 5.5))
    vmax = compute_attention_vmax([values for _, _, values in panels])
    for idx, (panel_title, coords, values) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, len(panels), idx, projection="3d")
        plot_spatial_attention_subplot(
            ax=ax,
            coords=coords,
            values=values,
            title=panel_title,
            vmax=vmax,
            point_size=4.5,
        )
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_curve(
    values: np.ndarray,
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    x_tick_labels: Optional[List[str]] = None,
) -> None:
    values = np.asarray(values).reshape(-1)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(values.shape[0])
    ax.plot(x, values, marker="o", color="#0D7A72", linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if x_tick_labels is not None and len(x_tick_labels) == len(x):
        ax.set_xticks(x)
        ax.set_xticklabels(x_tick_labels, rotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_heatmap(
    matrix: np.ndarray,
    x_labels: List[str],
    y_labels: List[str],
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    if matrix.size == 0:
        return
    fig_w = max(10, 0.28 * len(x_labels))
    fig_h = max(4, 0.35 * len(y_labels))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=70, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_patch_grid_heatmap(
    values: np.ndarray,
    selected_patch_grid: np.ndarray,
    path: Path,
    title: str,
) -> None:
    values = np.asarray(values, dtype=np.float32)
    selected_patch_grid = np.asarray(selected_patch_grid, dtype=bool)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im = axes[0].imshow(values, cmap="magma")
    axes[0].set_title("Patch Attention")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    for row in range(selected_patch_grid.shape[0]):
        for col in range(selected_patch_grid.shape[1]):
            if selected_patch_grid[row, col]:
                axes[0].add_patch(
                    Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="#2EC4B6", linewidth=1.6)
                )
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].imshow(selected_patch_grid.astype(np.float32), cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("Mask-Selected Patch Tokens")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_mask_patch_overview(
    image: Image.Image,
    mask: Image.Image,
    selected_patch_grid: np.ndarray,
    path: Path,
    title: str,
) -> None:
    image_np = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    mask_np = np.asarray(mask.convert("L")).astype(np.float32) / 255.0
    overlay = image_np.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] + mask_np * 0.75, 0.0, 1.0)
    overlay[..., 1] *= 1.0 - mask_np * 0.4
    overlay[..., 2] *= 1.0 - mask_np * 0.4

    patch_size = image_np.shape[0] // selected_patch_grid.shape[0]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(image_np)
    axes[0].set_title("Preprocessed Image")
    axes[1].imshow(mask_np, cmap="gray")
    axes[1].set_title("Preprocessed Mask")
    axes[2].imshow(overlay)
    axes[2].set_title("Mask + Selected Patch Tokens")

    for row in range(selected_patch_grid.shape[0]):
        for col in range(selected_patch_grid.shape[1]):
            if not selected_patch_grid[row, col]:
                continue
            axes[2].add_patch(
                Rectangle(
                    (col * patch_size - 0.5, row * patch_size - 0.5),
                    patch_size,
                    patch_size,
                    fill=False,
                    edgecolor="#2EC4B6",
                    linewidth=1.1,
                )
            )
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_top_query_highlight(
    coords: np.ndarray,
    values: np.ndarray,
    path: Path,
    title: str,
    topk: int,
) -> List[dict]:
    coords = np.asarray(coords, dtype=np.int16)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    top_indices = np.argsort(values)[::-1][:topk]
    top_items = [
        {
            "rank": int(rank),
            "query_index": int(query_idx),
            "coord": coords[query_idx].tolist(),
            "attention": float(values[query_idx]),
        }
        for rank, query_idx in enumerate(top_indices, start=1)
    ]

    vmax = compute_attention_vmax([values])
    fig = plt.figure(figsize=(12, 5.5))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    plot_spatial_attention_subplot(ax1, coords, values, "Attention Heatmap", vmax=vmax, point_size=4.2)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c="#D8E2E7", s=3, alpha=0.18)
    if len(top_indices) > 0:
        top_coords = coords[top_indices]
        ax2.scatter(
            top_coords[:, 0],
            top_coords[:, 1],
            top_coords[:, 2],
            c=np.linspace(1.0, 0.35, len(top_indices)),
            cmap="autumn",
            s=18,
            alpha=0.95,
        )
        for item in top_items:
            coord = item["coord"]
            ax2.text(coord[0], coord[1], coord[2], f"#{item['rank']}", fontsize=8)
    ax2.set_title(f"Top-{topk} Highlight")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])
    spans = np.ptp(coords, axis=0).astype(np.float32) if coords.size > 0 else np.ones((3,), dtype=np.float32)
    ax2.set_box_aspect(tuple(np.maximum(spans, 1.0)))
    ax2.view_init(elev=28, azim=42)
    ax2.grid(False)
    try:
        ax2.xaxis.pane.fill = False
        ax2.yaxis.pane.fill = False
        ax2.zaxis.pane.fill = False
    except AttributeError:
        pass

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return top_items


def render_snapshot_image(sample, path: Path) -> None:
    from trellis.utils import render_utils

    result = render_utils.render_snapshot(sample, resolution=512)
    frames = result["color"] if "color" in result else result["normal"]
    n = len(frames)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()
    for i, frame in enumerate(frames):
        axes[i].imshow(frame)
        axes[i].axis("off")
    for i in range(n, len(axes)):
        axes[i].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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


@dataclass
class CropContext:
    scale: float
    bbox: Tuple[int, int, int, int]


def extract_foreground_rgba(
    image: Image.Image,
    pipeline: TrellisImageTo3DPipeline,
    scale: float,
) -> Image.Image:
    image = scale_image(image, scale, Image.Resampling.LANCZOS)
    if has_useful_alpha(image):
        return image.convert("RGBA")
    import rembg

    fallback_alpha = getattr(pipeline, "_fallback_alpha_mask", None)
    if fallback_alpha is not None and fallback_alpha.size == image.size:
        rgb_np = np.asarray(image.convert("RGB"), dtype=np.uint8)
        alpha_np = np.asarray(fallback_alpha.convert("L"), dtype=np.uint8)
        rgba_np = np.dstack([rgb_np, alpha_np])
        return Image.fromarray(rgba_np, mode="RGBA")

    rgb = image.convert("RGB")
    if getattr(pipeline, "rembg_session", None) is None:
        pipeline.rembg_session = rembg.new_session("u2net")
    return rembg.remove(rgb, session=pipeline.rembg_session)


def build_crop_context(
    reference_image: Image.Image,
    pipeline: TrellisImageTo3DPipeline,
) -> CropContext:
    max_size = max(reference_image.size)
    scale = min(1.0, 1024.0 / float(max_size))
    fg = extract_foreground_rgba(reference_image, pipeline, scale)
    alpha = np.asarray(fg)[:, :, 3]
    bbox_pixels = np.argwhere(alpha > int(0.8 * 255))
    if bbox_pixels.size == 0:
        bbox = (0, 0, fg.width, fg.height)
        return CropContext(scale=scale, bbox=bbox)

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


def preprocess_condition_image(
    image: Image.Image,
    pipeline: TrellisImageTo3DPipeline,
    crop_context: CropContext,
) -> Image.Image:
    fg = extract_foreground_rgba(image, pipeline, crop_context.scale)
    cropped = fg.crop(crop_context.bbox)
    resized = cropped.resize((PROC_IMAGE_SIZE, PROC_IMAGE_SIZE), Image.Resampling.LANCZOS)
    resized_np = np.asarray(resized).astype(np.float32) / 255.0
    rgb = resized_np[:, :, :3] * resized_np[:, :, 3:4]
    return Image.fromarray((rgb * 255).astype(np.uint8))


def preprocess_mask_image(mask: Image.Image, crop_context: CropContext) -> Image.Image:
    mask = mask.convert("L")
    scaled = scale_image(mask, crop_context.scale, Image.Resampling.NEAREST)
    cropped = scaled.crop(crop_context.bbox)
    resized = cropped.resize((PROC_IMAGE_SIZE, PROC_IMAGE_SIZE), Image.Resampling.NEAREST)
    mask_np = np.asarray(resized)
    mask_np = (mask_np > 0).astype(np.uint8) * 255
    return Image.fromarray(mask_np, mode="L")


def mask_to_patch_selection(mask: Image.Image) -> Tuple[np.ndarray, List[int]]:
    mask_np = np.asarray(mask.convert("L")) > 0
    grid_size = PROC_IMAGE_SIZE // DINO_PATCH_SIZE
    selected = np.zeros((grid_size, grid_size), dtype=bool)
    selected_linear_indices: List[int] = []
    for row in range(grid_size):
        for col in range(grid_size):
            y0 = row * DINO_PATCH_SIZE
            y1 = min((row + 1) * DINO_PATCH_SIZE, PROC_IMAGE_SIZE)
            x0 = col * DINO_PATCH_SIZE
            x1 = min((col + 1) * DINO_PATCH_SIZE, PROC_IMAGE_SIZE)
            if bool(mask_np[y0:y1, x0:x1].any()):
                selected[row, col] = True
                selected_linear_indices.append(row * grid_size + col)
    return selected, selected_linear_indices


def build_image_token_metadata(cond: torch.Tensor, mask: Image.Image) -> dict:
    total_tokens = int(cond.shape[1])
    grid_size = PROC_IMAGE_SIZE // DINO_PATCH_SIZE
    patch_token_count = grid_size * grid_size
    prefix_token_count = total_tokens - patch_token_count
    if prefix_token_count < 0:
        raise ValueError(
            f"Unexpected token layout: total_tokens={total_tokens}, patch_token_count={patch_token_count}"
        )

    selected_patch_grid, selected_patch_linear_indices = mask_to_patch_selection(mask)

    token_labels: List[str] = []
    for idx in range(prefix_token_count):
        if idx == 0:
            token_labels.append("[CLS]")
        else:
            token_labels.append(f"[REG{idx}]")
    for row in range(grid_size):
        for col in range(grid_size):
            token_labels.append(f"patch_r{row:02d}_c{col:02d}")

    selected_token_indices = [prefix_token_count + idx for idx in selected_patch_linear_indices]
    selected_token_labels = [token_labels[idx] for idx in selected_token_indices]
    special_token_indices = list(range(prefix_token_count))

    return {
        "grid_size": [grid_size, grid_size],
        "patch_size": DINO_PATCH_SIZE,
        "proc_size": PROC_IMAGE_SIZE,
        "total_tokens": total_tokens,
        "prefix_token_count": prefix_token_count,
        "patch_token_count": patch_token_count,
        "token_labels": token_labels,
        "special_token_indices": special_token_indices,
        "selected_patch_linear_indices": selected_patch_linear_indices,
        "selected_patch_grid": selected_patch_grid.astype(np.uint8).tolist(),
        "selected_token_indices": selected_token_indices,
        "selected_token_labels": selected_token_labels,
        "selection_rule": "Any patch that overlaps the mask by at least one pixel is counted as a selected visual token.",
    }


class ImageCrossAttentionTracer:
    def __init__(
        self,
        token_meta: dict,
        query_chunk: int,
        topk_spatial: int,
    ):
        self.token_meta = token_meta
        self.valid_token_count = int(token_meta["total_tokens"])
        self.selected_token_indices = [int(idx) for idx in token_meta["selected_token_indices"]]
        self.query_chunk = query_chunk
        self.topk_spatial = topk_spatial
        self.stage_states: Dict[str, dict] = {}
        self.current_forward_context: Optional[dict] = None
        self._restore_stack: List[Tuple[object, str, object]] = []

    def prepare_stage(
        self,
        stage_name: str,
        model: torch.nn.Module,
        cond: torch.Tensor,
        neg_cond: torch.Tensor,
        total_steps: int,
        enabled: bool,
        query_type: str,
    ) -> None:
        num_blocks = len(getattr(model, "blocks"))
        self.stage_states[stage_name] = {
            "enabled": enabled,
            "query_type": query_type,
            "total_steps": total_steps,
            "num_blocks": num_blocks,
            "selected_steps": select_key_indices(total_steps, max_items=7),
            "selected_blocks": select_key_indices(num_blocks, max_items=3),
            "cond_signature": tensor_signature(cond),
            "neg_signature": tensor_signature(neg_cond),
            "cond_step_counter": 0,
            "step_block_token_sum": np.zeros(
                (total_steps, num_blocks, self.valid_token_count),
                dtype=np.float64,
            ),
            "step_block_query_count": np.zeros((total_steps, num_blocks), dtype=np.int64),
            "selected_maps": {},
            "overview_block_index": num_blocks - 1,
            "overview_sum_map": None,
            "overview_count": 0,
            "overview_coords": None,
        }
        if enabled:
            self._patch_model_forward(stage_name, model)
            self._patch_cross_modules(stage_name, model)

    def set_dense_grid_meta(self, stage_name: str, resolution: int, patch_size: int) -> None:
        state = self.stage_states[stage_name]
        state["dense_resolution"] = int(resolution)
        state["dense_patch_size"] = int(patch_size)

    def restore(self) -> None:
        while self._restore_stack:
            obj, attr, value = self._restore_stack.pop()
            setattr(obj, attr, value)

    def _infer_pass_kind(self, stage_name: str, cond: torch.Tensor) -> str:
        state = self.stage_states[stage_name]
        current_sig = tensor_signature(cond)
        dist_cond = float(np.max(np.abs(current_sig - state["cond_signature"])))
        dist_neg = float(np.max(np.abs(current_sig - state["neg_signature"])))
        return "cond" if dist_cond <= dist_neg else "neg"

    def _patch_model_forward(self, stage_name: str, model: torch.nn.Module) -> None:
        original_forward = model.forward
        self._restore_stack.append((model, "forward", original_forward))
        tracer = self

        def wrapped_forward(model_self, x, t, cond):
            state = tracer.stage_states[stage_name]
            pass_kind = tracer._infer_pass_kind(stage_name, cond)
            if pass_kind == "cond":
                step_index = state["cond_step_counter"]
                state["cond_step_counter"] += 1
            else:
                step_index = max(state["cond_step_counter"] - 1, 0)
            step_index = min(step_index, state["total_steps"] - 1)
            tracer.current_forward_context = {
                "stage": stage_name,
                "pass_kind": pass_kind,
                "step_index": step_index,
                "timestep": float(t[0].detach().float().cpu().item()) if torch.is_tensor(t) else float(t),
            }
            try:
                return original_forward(x, t, cond)
            finally:
                tracer.current_forward_context = None

        model.forward = types.MethodType(wrapped_forward, model)

    def _patch_cross_modules(self, stage_name: str, model: torch.nn.Module) -> None:
        block_pattern = re.compile(r"blocks\.(\d+)\.cross_attn$")
        sparse_idx = 0
        dense_idx = 0
        for module_name, module in model.named_modules():
            if isinstance(module, MultiHeadAttention) and getattr(module, "_type", None) == "cross":
                match = block_pattern.search(module_name)
                block_index = int(match.group(1)) if match else dense_idx
                dense_idx += 1
                self._patch_dense_cross(stage_name, module_name, block_index, module)
            elif isinstance(module, SparseMultiHeadAttention) and getattr(module, "_type", None) == "cross":
                match = block_pattern.search(module_name)
                block_index = int(match.group(1)) if match else sparse_idx
                sparse_idx += 1
                self._patch_sparse_cross(stage_name, module_name, block_index, module)

    def _patch_dense_cross(
        self,
        stage_name: str,
        module_name: str,
        block_index: int,
        module: MultiHeadAttention,
    ) -> None:
        original_forward = module.forward
        self._restore_stack.append((module, "forward", original_forward))
        tracer = self

        def wrapped_forward(attn_self, x, context=None, indices=None):
            batch_size, num_queries, _ = x.shape
            q = attn_self.to_q(x).reshape(batch_size, num_queries, attn_self.num_heads, -1)
            kv = attn_self.to_kv(context).reshape(batch_size, context.shape[1], 2, attn_self.num_heads, -1)
            k, v = kv.unbind(dim=2)
            if attn_self.qk_rms_norm:
                q = attn_self.q_rms_norm(q)
                k = attn_self.k_rms_norm(k)
            tracer._record_dense(stage_name, module_name, block_index, q, k, batch_size)
            h = scaled_dot_product_attention(q, k, v)
            h = h.reshape(batch_size, num_queries, -1)
            h = attn_self.to_out(h)
            return h

        module.forward = types.MethodType(wrapped_forward, module)

    def _patch_sparse_cross(
        self,
        stage_name: str,
        module_name: str,
        block_index: int,
        module: SparseMultiHeadAttention,
    ) -> None:
        original_forward = module.forward
        self._restore_stack.append((module, "forward", original_forward))
        tracer = self

        def wrapped_forward(attn_self, x, context=None):
            q = attn_self._linear(attn_self.to_q, x)
            q = attn_self._reshape_chs(q, (attn_self.num_heads, -1))
            kv = attn_self._linear(attn_self.to_kv, context)
            kv = attn_self._fused_pre(kv, num_fused=2)
            if isinstance(kv, SparseTensor):
                k, v = kv.unbind(dim=1)
            else:
                k, v = kv.unbind(dim=2)
            if attn_self.qk_rms_norm:
                q = attn_self.q_rms_norm(q)
                k = attn_self.k_rms_norm(k)
            tracer._record_sparse(stage_name, module_name, block_index, q, k)
            h = sparse_scaled_dot_product_attention(q, k, v)
            h = attn_self._reshape_chs(h, (-1,))
            h = attn_self._linear(attn_self.to_out, h)
            return h

        module.forward = types.MethodType(wrapped_forward, module)

    def _should_keep_map(self, stage_name: str, step_index: int, block_index: int) -> bool:
        state = self.stage_states[stage_name]
        return step_index in state["selected_steps"] and block_index in state["selected_blocks"]

    def _record_dense(
        self,
        stage_name: str,
        module_name: str,
        block_index: int,
        q: torch.Tensor,
        k: torch.Tensor,
        batch_size: int,
    ) -> None:
        ctx = self.current_forward_context
        if ctx is None or ctx["stage"] != stage_name or ctx["pass_kind"] != "cond":
            return
        state = self.stage_states[stage_name]
        step_index = ctx["step_index"]
        selected_map = self._should_keep_map(stage_name, step_index, block_index)
        overview_map = block_index == state["overview_block_index"]
        keep_map = selected_map or overview_map

        dense_coords = dense_query_coords(
            resolution=int(state["dense_resolution"]),
            patch_size=int(state["dense_patch_size"]),
            batch_size=batch_size,
        )
        for batch_idx in range(batch_size):
            token_sum, attn_map = compute_headmean_attention(
                q=q[batch_idx],
                k=k[batch_idx],
                query_chunk=self.query_chunk,
                keep_map=keep_map and batch_idx == 0,
            )
            state["step_block_token_sum"][step_index, block_index] += token_sum[: self.valid_token_count]
            state["step_block_query_count"][step_index, block_index] += q.shape[1]

            if batch_idx == 0 and overview_map and attn_map is not None:
                self._update_overview_map(
                    state=state,
                    coords=dense_coords[batch_idx],
                    attn_map=attn_map[:, : self.valid_token_count],
                )
            if batch_idx == 0 and selected_map and attn_map is not None:
                state["selected_maps"][(step_index, block_index)] = {
                    "module_name": module_name,
                    "step_index": step_index,
                    "block_index": block_index,
                    "timestep": ctx["timestep"],
                    "coords": dense_coords[batch_idx],
                    "attn_map": attn_map[:, : self.valid_token_count],
                }

    def _record_sparse(
        self,
        stage_name: str,
        module_name: str,
        block_index: int,
        q: SparseTensor,
        k: torch.Tensor,
    ) -> None:
        ctx = self.current_forward_context
        if ctx is None or ctx["stage"] != stage_name or ctx["pass_kind"] != "cond":
            return
        state = self.stage_states[stage_name]
        step_index = ctx["step_index"]
        selected_map = self._should_keep_map(stage_name, step_index, block_index)
        overview_map = block_index == state["overview_block_index"]
        keep_map = selected_map or overview_map

        for batch_idx in range(q.shape[0]):
            q_slice = q.feats[q.layout[batch_idx]]
            token_sum, attn_map = compute_headmean_attention(
                q=q_slice,
                k=k[batch_idx],
                query_chunk=self.query_chunk,
                keep_map=keep_map and batch_idx == 0,
            )
            state["step_block_token_sum"][step_index, block_index] += token_sum[: self.valid_token_count]
            state["step_block_query_count"][step_index, block_index] += q_slice.shape[0]

            if batch_idx == 0 and overview_map and attn_map is not None:
                self._update_overview_map(
                    state=state,
                    coords=q.coords[q.layout[batch_idx], 1:].detach().cpu().numpy().astype(np.int16),
                    attn_map=attn_map[:, : self.valid_token_count],
                )
            if batch_idx == 0 and selected_map and attn_map is not None:
                state["selected_maps"][(step_index, block_index)] = {
                    "module_name": module_name,
                    "step_index": step_index,
                    "block_index": block_index,
                    "timestep": ctx["timestep"],
                    "coords": q.coords[q.layout[batch_idx], 1:].detach().cpu().numpy().astype(np.int16),
                    "attn_map": attn_map[:, : self.valid_token_count],
                }

    def _update_overview_map(self, state: dict, coords: np.ndarray, attn_map: np.ndarray) -> None:
        if state["overview_sum_map"] is None:
            state["overview_sum_map"] = attn_map.astype(np.float32, copy=True)
            state["overview_coords"] = np.asarray(coords, dtype=np.int16)
            state["overview_count"] = 1
            return
        if state["overview_sum_map"].shape != attn_map.shape:
            return
        state["overview_sum_map"] += attn_map.astype(np.float32, copy=False)
        state["overview_count"] += 1

    def _normalized_step_block_token_mean(self, state: dict) -> np.ndarray:
        counts = np.maximum(state["step_block_query_count"].astype(np.float64)[..., None], 1.0)
        return state["step_block_token_sum"] / counts

    def export(
        self,
        root_dir: Path,
        preprocessed_image_path: Path,
        preprocessed_mask_path: Path,
        mask_overlay_path: Path,
        mesh_snapshot_path: Optional[Path] = None,
    ) -> dict:
        token_labels = list(self.token_meta["token_labels"])
        selected_token_indices = list(self.selected_token_indices)
        selected_token_labels = [token_labels[idx] for idx in selected_token_indices]
        grid_h, grid_w = self.token_meta["grid_size"]
        prefix = int(self.token_meta["prefix_token_count"])
        selected_patch_grid = np.asarray(self.token_meta["selected_patch_grid"], dtype=bool)

        manifest = {
            "preprocessed_image": preprocessed_image_path.name,
            "preprocessed_mask": preprocessed_mask_path.name,
            "mask_overlay": mask_overlay_path.name,
            "mesh_snapshot": mesh_snapshot_path.name if mesh_snapshot_path is not None else "",
            "token_labels": token_labels,
            "selected_token_indices": selected_token_indices,
            "selected_token_labels": selected_token_labels,
            "selected_patch_count": len(selected_token_indices),
            "grid_size": [int(grid_h), int(grid_w)],
            "patch_size": int(self.token_meta["patch_size"]),
            "selection_rule": self.token_meta["selection_rule"],
            "stages": {},
        }

        for stage_name, state in self.stage_states.items():
            if not state["enabled"]:
                continue
            stage_dir = ensure_dir(root_dir / stage_name)
            normalized = self._normalized_step_block_token_mean(state)
            step_token_mean = normalized.mean(axis=1)
            final_step_block_mean = normalized[-1]

            step_selected_mass = step_token_mean[:, selected_token_indices].sum(axis=1)
            final_block_selected_mass = final_step_block_mean[:, selected_token_indices].sum(axis=1)
            selected_patch_heatmap = step_token_mean[:, selected_token_indices]

            patch_attention_over_steps = step_token_mean[:, prefix : prefix + grid_h * grid_w]
            final_patch_grid = final_step_block_mean[-1, prefix : prefix + grid_h * grid_w].reshape(grid_h, grid_w)
            average_patch_grid = patch_attention_over_steps.mean(axis=0).reshape(grid_h, grid_w)

            np.save(stage_dir / "step_token_mean.npy", step_token_mean.astype(np.float32))
            np.save(stage_dir / "final_step_block_token_mean.npy", final_step_block_mean.astype(np.float32))
            np.save(stage_dir / "step_selected_token_mass.npy", step_selected_mass.astype(np.float32))
            np.save(stage_dir / "final_block_selected_token_mass.npy", final_block_selected_mass.astype(np.float32))

            save_json(
                stage_dir / "summary.json",
                {
                    "stage": stage_name,
                    "query_type": state["query_type"],
                    "total_steps": state["total_steps"],
                    "num_blocks": state["num_blocks"],
                    "selected_steps": state["selected_steps"],
                    "selected_blocks": state["selected_blocks"],
                    "selected_token_indices": selected_token_indices,
                    "selected_token_labels": selected_token_labels,
                    "selected_patch_count": len(selected_token_indices),
                    "overview_block_index": state["overview_block_index"],
                },
            )

            save_curve(
                step_selected_mass,
                stage_dir / "selected_token_step_curve.png",
                title=f"{stage_name}: average attention mass on mask-selected visual tokens across steps",
                xlabel="Sampling Step",
                ylabel="Mean Selected-Token Attention Mass",
                x_tick_labels=[f"step_{i:02d}" for i in range(step_selected_mass.shape[0])],
            )
            save_curve(
                final_block_selected_mass,
                stage_dir / "selected_token_block_curve.png",
                title=f"{stage_name}: final-step attention mass on mask-selected visual tokens across blocks",
                xlabel="Transformer Block",
                ylabel="Mean Selected-Token Attention Mass",
                x_tick_labels=[f"block_{i:02d}" for i in range(final_block_selected_mass.shape[0])],
            )

            if len(selected_token_labels) > 0:
                save_heatmap(
                    selected_patch_heatmap,
                    selected_token_labels,
                    [f"step_{i:02d}" for i in range(selected_patch_heatmap.shape[0])],
                    stage_dir / "selected_patch_heatmap.png",
                    title=f"{stage_name}: selected visual token attention over sampling steps",
                    xlabel="Mask-Selected Visual Token",
                    ylabel="Sampling Step",
                )

            save_patch_grid_heatmap(
                average_patch_grid,
                selected_patch_grid,
                stage_dir / "average_patch_grid.png",
                title=f"{stage_name}: average patch attention over all steps",
            )
            save_patch_grid_heatmap(
                final_patch_grid,
                selected_patch_grid,
                stage_dir / "final_patch_grid.png",
                title=f"{stage_name}: final-step patch attention (last block)",
            )

            selected_map_meta = []
            raw_dir = ensure_dir(stage_dir / "raw_maps")
            for (step_index, block_index), item in sorted(state["selected_maps"].items()):
                base = f"step_{step_index:02d}_block_{block_index:02d}"
                np.savez_compressed(
                    raw_dir / f"{base}.npz",
                    attn_map=item["attn_map"].astype(np.float16),
                    coords=item["coords"].astype(np.int16),
                    selected_token_indices=np.asarray(selected_token_indices, dtype=np.int32),
                )
                save_json(
                    raw_dir / f"{base}.json",
                    {
                        "stage": stage_name,
                        "module_name": item["module_name"],
                        "step_index": int(step_index),
                        "block_index": int(block_index),
                        "timestep": float(item["timestep"]),
                        "num_queries": int(item["attn_map"].shape[0]),
                        "num_tokens": int(item["attn_map"].shape[1]),
                        "selected_token_indices": selected_token_indices,
                    },
                )
                selected_map_meta.append(
                    {
                        "key": [int(step_index), int(block_index)],
                        "module_name": item["module_name"],
                        "npz": str((raw_dir / f"{base}.npz").relative_to(root_dir)),
                        "json": str((raw_dir / f"{base}.json").relative_to(root_dir)),
                    }
                )

            avg_selected_values = None
            avg_top_items: List[dict] = []
            if state["overview_sum_map"] is not None and state["overview_count"] > 0 and state["overview_coords"] is not None:
                avg_map = state["overview_sum_map"] / float(max(state["overview_count"], 1))
                avg_selected_values = avg_map[:, selected_token_indices].sum(axis=1).astype(np.float32)
                save_spatial_panels(
                    [
                        (
                            f"avg over steps, block {state['overview_block_index']}",
                            state["overview_coords"],
                            avg_selected_values,
                        )
                    ],
                    stage_dir / "masked_region_average.png",
                    title=f"{stage_name}: 3D region activated by mask-selected visual tokens",
                )
                avg_top_items = save_top_query_highlight(
                    coords=state["overview_coords"],
                    values=avg_selected_values,
                    path=stage_dir / "masked_region_top_queries.png",
                    title=f"{stage_name}: top 3D queries for mask-selected visual tokens",
                    topk=self.topk_spatial,
                )
                save_json(
                    stage_dir / "masked_region_top_queries.json",
                    {
                        "stage": stage_name,
                        "top_items": avg_top_items,
                    },
                )

            interactive_payload = build_interactive_stage_payload(
                stage_name=stage_name,
                state=state,
                selected_token_indices=selected_token_indices,
                avg_selected_values=avg_selected_values,
            )
            if interactive_payload is not None:
                save_json(stage_dir / "interactive_3d.json", interactive_payload)

            step_panels = []
            for step_index in state["selected_steps"]:
                item = state["selected_maps"].get((step_index, state["overview_block_index"]))
                if item is None:
                    continue
                values = item["attn_map"][:, selected_token_indices].sum(axis=1).astype(np.float32)
                step_panels.append(
                    (
                        f"step {step_index}, t={item['timestep']:.3g}",
                        item["coords"],
                        values,
                    )
                )
            save_spatial_panels(
                step_panels,
                stage_dir / "masked_region_step_progress.png",
                title=f"{stage_name}: masked visual token attention across steps",
            )

            block_panels = []
            last_step = state["total_steps"] - 1
            for block_index in state["selected_blocks"]:
                item = state["selected_maps"].get((last_step, block_index))
                if item is None:
                    continue
                values = item["attn_map"][:, selected_token_indices].sum(axis=1).astype(np.float32)
                block_panels.append(
                    (
                        f"block {block_index}, step {last_step}",
                        item["coords"],
                        values,
                    )
                )
            save_spatial_panels(
                block_panels,
                stage_dir / "masked_region_block_progress.png",
                title=f"{stage_name}: masked visual token attention across blocks",
            )

            manifest["stages"][stage_name] = {
                "summary": str((stage_dir / "summary.json").relative_to(root_dir)),
                "selected_token_step_curve": str((stage_dir / "selected_token_step_curve.png").relative_to(root_dir)),
                "selected_token_block_curve": str((stage_dir / "selected_token_block_curve.png").relative_to(root_dir)),
                "selected_patch_heatmap": str((stage_dir / "selected_patch_heatmap.png").relative_to(root_dir))
                if len(selected_token_labels) > 0
                else "",
                "average_patch_grid": str((stage_dir / "average_patch_grid.png").relative_to(root_dir)),
                "final_patch_grid": str((stage_dir / "final_patch_grid.png").relative_to(root_dir)),
                "masked_region_average": str((stage_dir / "masked_region_average.png").relative_to(root_dir))
                if avg_selected_values is not None
                else "",
                "masked_region_step_progress": str((stage_dir / "masked_region_step_progress.png").relative_to(root_dir)),
                "masked_region_block_progress": str((stage_dir / "masked_region_block_progress.png").relative_to(root_dir)),
                "masked_region_top_queries": str((stage_dir / "masked_region_top_queries.png").relative_to(root_dir))
                if avg_top_items
                else "",
                "masked_region_top_queries_meta": str((stage_dir / "masked_region_top_queries.json").relative_to(root_dir))
                if avg_top_items
                else "",
                "interactive_3d": str((stage_dir / "interactive_3d.json").relative_to(root_dir))
                if interactive_payload is not None
                else "",
                "selected_maps": selected_map_meta,
            }

        save_json(root_dir / "trace_manifest.json", manifest)
        return manifest


def resolve_image_paths(args: argparse.Namespace) -> Tuple[List[Path], Path]:
    if args.images.strip():
        image_paths = [Path(item.strip()) for item in args.images.split(",") if item.strip()]
    else:
        input_dir = Path(args.input_dir)
        candidates = sorted(
            [
                path
                for path in input_dir.iterdir()
                if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            ]
        )
        image_paths = [path for path in candidates if "mask" not in path.stem.lower()]
    if not image_paths:
        raise ValueError("No conditioning images were found. Please pass --images or provide an input directory.")

    if args.mask.strip():
        mask_path = Path(args.mask)
    else:
        mask_candidates = [path for path in Path(args.input_dir).iterdir() if "mask" in path.stem.lower()]
        if not mask_candidates:
            raise ValueError("No mask image was found. Please pass --mask explicitly.")
        mask_path = sorted(mask_candidates)[0]

    return image_paths, mask_path


def choose_reference_image(image_paths: Sequence[Path], explicit_path: str) -> Path:
    if explicit_path.strip():
        return Path(explicit_path)
    for path in image_paths:
        try:
            image = Image.open(path)
            if image.mode == "RGBA":
                alpha = np.asarray(image)[:, :, 3]
                if not np.all(alpha == 255):
                    return path
        except Exception:
            continue
    return image_paths[0]


def load_partial_image_pipeline(path: str) -> TrellisImageTo3DPipeline:
    import os

    if os.path.exists(f"{path}/pipeline.json"):
        config_file = f"{path}/pipeline.json"
    else:
        from huggingface_hub import hf_hub_download

        config_file = hf_hub_download(path, "pipeline.json")

    with open(config_file, "r", encoding="utf-8") as f:
        args = json.load(f)["args"]

    required_models = [
        "sparse_structure_flow_model",
        "sparse_structure_decoder",
        "slat_flow_model",
    ]
    models_dict = {}
    for model_key in required_models:
        model_ref = args["models"][model_key]
        try:
            models_dict[model_key] = trellis_models.from_pretrained(f"{path}/{model_ref}")
        except Exception:
            models_dict[model_key] = trellis_models.from_pretrained(model_ref)

    pipeline = TrellisImageTo3DPipeline(
        models=models_dict,
        sparse_structure_sampler=getattr(
            trellis_samplers,
            args["sparse_structure_sampler"]["name"],
        )(**args["sparse_structure_sampler"]["args"]),
        slat_sampler=getattr(
            trellis_samplers,
            args["slat_sampler"]["name"],
        )(**args["slat_sampler"]["args"]),
        slat_normalization=args["slat_normalization"],
        image_cond_model=args["image_cond_model"],
    )
    pipeline.sparse_structure_sampler_params = args["sparse_structure_sampler"]["params"]
    pipeline.slat_sampler_params = args["slat_sampler"]["params"]
    pipeline._pretrained_args = args
    return pipeline


def render_trace_html(case_manifest: dict, subcase_manifests: Dict[str, dict]) -> str:
    def image_card(rel_path: str, title: str, base: str) -> str:
        if not rel_path:
            return ""
        return f"""
        <div class="card">
          <p class="label">{title}</p>
          <img src="{base}/{rel_path}" alt="{title}">
        </div>
        """

    def interactive_card(rel_path: str, base: str) -> str:
        if not rel_path:
            return ""
        return f"""
        <div class="card wide interactive-card" data-interactive-3d data-src="{base}/{rel_path}">
          <div class="interactive-head">
            <div>
              <p class="label">Interactive 3D Attention</p>
              <p class="interactive-copy">
                旋转、缩放并拖动阈值 <code>t</code>。页面只显示 attention 值大于等于 <code>t</code> 的 3D 空间块。
              </p>
            </div>
            <div class="interactive-controls">
              <label class="interactive-control">
                <span>View</span>
                <select class="interactive-view-select"></select>
              </label>
              <label class="interactive-control range-control">
                <span>Threshold t</span>
                <input class="interactive-threshold-range" type="range" min="0" max="1" step="0.0001" value="0">
              </label>
              <label class="interactive-control compact">
                <span>t</span>
                <input class="interactive-threshold-number" type="number" min="0" step="0.0001" value="0">
              </label>
            </div>
          </div>
          <p class="interactive-meta muted">Loading interactive 3D viewer...</p>
          <div class="interactive-plot"></div>
        </div>
        """

    sections: List[str] = []
    nav_items: List[str] = []
    for subcase in case_manifest["subcases"]:
        slug = subcase["id"]
        nav_items.append(f'<a href="#{slug}">{subcase["label"]}</a>')
        manifest = subcase_manifests[slug]
        base = slug

        stage_blocks: List[str] = []
        for stage_name, stage_meta in manifest.get("stages", {}).items():
            stage_blocks.append(
                f"""
                <section class="stage">
                  <h3>{stage_name}</h3>
                  <div class="grid">
                    {interactive_card(stage_meta.get("interactive_3d", ""), base)}
                    {image_card(stage_meta.get("selected_token_step_curve", ""), "Selected Token Step Curve", base)}
                    {image_card(stage_meta.get("selected_token_block_curve", ""), "Selected Token Block Curve", base)}
                    {image_card(stage_meta.get("selected_patch_heatmap", ""), "Selected Patch Heatmap", base)}
                    {image_card(stage_meta.get("average_patch_grid", ""), "Average Patch Grid", base)}
                    {image_card(stage_meta.get("final_patch_grid", ""), "Final Patch Grid", base)}
                    {image_card(stage_meta.get("masked_region_average", ""), "Masked Region Average", base)}
                    {image_card(stage_meta.get("masked_region_step_progress", ""), "Masked Region Step Progress", base)}
                    {image_card(stage_meta.get("masked_region_block_progress", ""), "Masked Region Block Progress", base)}
                    {image_card(stage_meta.get("masked_region_top_queries", ""), "Masked Region Top Queries", base)}
                  </div>
                </section>
                """
            )

        sections.append(
            f"""
            <section id="{slug}" class="case">
              <div class="case-head">
                <div>
                  <p class="eyebrow">Condition Image</p>
                  <h2>{subcase["label"]}</h2>
                  <p class="muted">source: {subcase["image_path"]}</p>
                </div>
                <div class="pill">selected patch tokens: {manifest.get("selected_patch_count", 0)}</div>
              </div>

              <div class="grid">
                {image_card(manifest.get("preprocessed_image", ""), "Preprocessed Image", base)}
                {image_card(manifest.get("preprocessed_mask", ""), "Preprocessed Mask", base)}
                {image_card(manifest.get("mask_overlay", ""), "Mask + Selected Patch Tokens", base)}
                {image_card(manifest.get("mesh_snapshot", ""), "Mesh Snapshot", base)}
              </div>

              {''.join(stage_blocks)}
            </section>
            """
        )

    interactive_bootstrap = """
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <script>
      function formatAttnNumber(value) {
        if (!Number.isFinite(value)) return "n/a";
        if (Math.abs(value) >= 1e-2) return value.toFixed(4);
        if (Math.abs(value) >= 1e-4) return value.toFixed(6);
        return value.toExponential(2);
      }

      function getCoords(data, view) {
        return view.coords || data.shared_coords;
      }

      function clampThreshold(value, maxValue) {
        var numeric = Number(value);
        if (!Number.isFinite(numeric)) return 0;
        return Math.max(0, Math.min(numeric, maxValue));
      }

      function renderInteractivePlot(card, data, viewMap) {
        var viewSelect = card.querySelector(".interactive-view-select");
        var rangeInput = card.querySelector(".interactive-threshold-range");
        var numberInput = card.querySelector(".interactive-threshold-number");
        var meta = card.querySelector(".interactive-meta");
        var plot = card.querySelector(".interactive-plot");
        var view = viewMap.get(viewSelect.value) || data.views[0];
        var coords = getCoords(data, view);
        if (!coords) {
          meta.textContent = "No 3D coordinate data is available for this stage.";
          return;
        }

        var globalMax = Math.max(Number(data.global_max || 0), 1e-6);
        var threshold = clampThreshold(numberInput.value, globalMax);
        if (document.activeElement !== rangeInput) rangeInput.value = String(threshold);
        if (document.activeElement !== numberInput) numberInput.value = threshold.toFixed(6);

        var x = [];
        var y = [];
        var z = [];
        var values = [];
        for (var i = 0; i < view.values.length; i += 1) {
          var value = Number(view.values[i]);
          if (value < threshold) continue;
          x.push(coords.x[i]);
          y.push(coords.y[i]);
          z.push(coords.z[i]);
          values.push(value);
        }

        meta.textContent =
          view.label +
          " · showing " +
          values.length +
          "/" +
          view.values.length +
          " blocks with attention >= t · t=" +
          formatAttnNumber(threshold) +
          " · max=" +
          formatAttnNumber(view.stats.max) +
          " · p95=" +
          formatAttnNumber(view.stats.p95);

        Plotly.react(
          plot,
          [
            {
              type: "scatter3d",
              mode: "markers",
              x: x,
              y: y,
              z: z,
              hovertemplate: "x=%{x}<br>y=%{y}<br>z=%{z}<br>attn=%{marker.color:.6f}<extra></extra>",
              marker: {
                size: data.query_type === "dense" ? 3.2 : 4.8,
                opacity: 0.92,
                color: values,
                colorscale: "Viridis",
                cmin: 0,
                cmax: Math.max(Number(data.global_vmax || 0), 1e-6),
                colorbar: {
                  title: "Attn",
                },
              },
            },
          ],
          {
            margin: { l: 0, r: 0, b: 0, t: 8 },
            paper_bgcolor: "#ffffff",
            plot_bgcolor: "#ffffff",
            uirevision: card.dataset.src,
            scene: {
              aspectmode: "data",
              xaxis: {
                title: "x",
                backgroundcolor: "#ffffff",
                gridcolor: "#e3eaed",
                zerolinecolor: "#d7e0e4",
              },
              yaxis: {
                title: "y",
                backgroundcolor: "#ffffff",
                gridcolor: "#e3eaed",
                zerolinecolor: "#d7e0e4",
              },
              zaxis: {
                title: "z",
                backgroundcolor: "#ffffff",
                gridcolor: "#e3eaed",
                zerolinecolor: "#d7e0e4",
              },
              camera: {
                eye: { x: 1.45, y: 1.45, z: 1.1 },
              },
            },
          },
          {
            responsive: true,
            displaylogo: false,
          }
        );
      }

      async function initInteractiveCard(card) {
        var viewSelect = card.querySelector(".interactive-view-select");
        var rangeInput = card.querySelector(".interactive-threshold-range");
        var numberInput = card.querySelector(".interactive-threshold-number");
        var meta = card.querySelector(".interactive-meta");

        try {
          var response = await fetch(card.dataset.src);
          if (!response.ok) {
            throw new Error("HTTP " + response.status);
          }
          var data = await response.json();
          var viewMap = new Map();
          for (var i = 0; i < data.views.length; i += 1) {
            var view = data.views[i];
            viewMap.set(view.id, view);
            var option = document.createElement("option");
            option.value = view.id;
            option.textContent = view.label;
            viewSelect.appendChild(option);
          }

          var globalMax = Math.max(Number(data.global_max || 0), 1e-6);
          var stepSize = Math.max(globalMax / 240, 1e-6);
          rangeInput.max = String(globalMax);
          rangeInput.step = String(stepSize);
          numberInput.max = String(globalMax);
          numberInput.step = String(stepSize);

          if (data.default_view_id && viewMap.has(data.default_view_id)) {
            viewSelect.value = data.default_view_id;
          }
          var defaultThreshold = clampThreshold(data.default_threshold, globalMax);
          rangeInput.value = String(defaultThreshold);
          numberInput.value = defaultThreshold.toFixed(6);

          rangeInput.addEventListener("input", function () {
            numberInput.value = Number(rangeInput.value).toFixed(6);
            renderInteractivePlot(card, data, viewMap);
          });
          numberInput.addEventListener("input", function () {
            rangeInput.value = String(clampThreshold(numberInput.value, globalMax));
            renderInteractivePlot(card, data, viewMap);
          });
          viewSelect.addEventListener("change", function () {
            renderInteractivePlot(card, data, viewMap);
          });

          renderInteractivePlot(card, data, viewMap);
        } catch (error) {
          meta.textContent = "Failed to load interactive 3D data: " + error;
        }
      }

      document.addEventListener("DOMContentLoaded", function () {
        var cards = document.querySelectorAll("[data-interactive-3d]");
        cards.forEach(function (card) {
          initInteractiveCard(card);
        });
      });
    </script>
    """

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TRELLIS Image Cross-Attention Trace</title>
  <style>
    :root {{
      --bg: #f5f7f8;
      --panel: #ffffff;
      --line: #d7e0e4;
      --muted: #60707a;
      --ink: #1f2a33;
      --accent: #0d7a72;
      --soft: #eef3f5;
      --shadow: 0 14px 34px rgba(20, 38, 48, 0.08);
      --radius: 18px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: linear-gradient(180deg, #f8fafb 0%, #f2f5f6 100%);
      color: var(--ink);
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      padding: 24px 16px 40px;
    }}
    .page {{
      width: min(1320px, 100%);
      margin: 0 auto;
    }}
    .hero, .case, .card, .stage {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }}
    .hero, .case, .stage {{
      padding: 22px;
    }}
    .hero h1, .case h2, .stage h3 {{
      margin: 0;
    }}
    .hero p, .muted {{
      color: var(--muted);
      line-height: 1.7;
    }}
    .eyebrow {{
      margin: 0 0 8px;
      font-size: 12px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--accent);
      font-weight: 700;
    }}
    .nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 18px 0 0;
    }}
    .nav a, .pill {{
      display: inline-flex;
      align-items: center;
      text-decoration: none;
      border-radius: 999px;
      padding: 10px 14px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      font-size: 14px;
    }}
    .case {{
      margin-top: 18px;
    }}
    .case-head {{
      display: flex;
      align-items: start;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }}
    .grid {{
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    }}
    .card {{
      padding: 14px;
      background: var(--soft);
    }}
    .card.wide {{
      grid-column: 1 / -1;
    }}
    .label {{
      margin: 0 0 10px;
      font-size: 12px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
      font-weight: 700;
    }}
    img {{
      display: block;
      width: 100%;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: #fff;
    }}
    .stage {{
      margin-top: 18px;
    }}
    .stage > h3 {{
      margin-bottom: 14px;
    }}
    .interactive-card {{
      background: #f8fbfb;
    }}
    .interactive-head {{
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 12px;
    }}
    .interactive-copy {{
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.7;
    }}
    .interactive-controls {{
      display: flex;
      flex-wrap: wrap;
      align-items: end;
      gap: 12px;
    }}
    .interactive-control {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      min-width: 168px;
    }}
    .interactive-control.compact {{
      min-width: 118px;
    }}
    .interactive-control.range-control {{
      min-width: 240px;
    }}
    .interactive-control span {{
      font-size: 11px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
      font-weight: 700;
    }}
    .interactive-control select,
    .interactive-control input[type="number"] {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
      color: var(--ink);
      padding: 10px 12px;
      font: inherit;
    }}
    .interactive-control input[type="range"] {{
      width: 100%;
      margin: 0;
      accent-color: var(--accent);
    }}
    .interactive-meta {{
      margin: 0 0 12px;
      font-size: 13px;
    }}
    .interactive-plot {{
      width: 100%;
      min-height: 560px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fff;
      overflow: hidden;
    }}
    @media (max-width: 900px) {{
      .case-head {{
        display: block;
      }}
      .interactive-head {{
        display: block;
      }}
      .interactive-controls {{
        margin-top: 12px;
      }}
      .interactive-plot {{
        min-height: 440px;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <p class="eyebrow">TRELLIS</p>
      <h1>Image-to-3D Cross-Attention Trace</h1>
      <p>
        这个页面展示的是“掩码命中的 visual token”与 3D 空间 token 之间的对应关系。
        这里使用的规则是：只要一个 patch 与掩码存在任意像素重叠，该 patch 对应的整个 visual token 就会被纳入关注集合。
      </p>
      <div class="nav">
        {''.join(nav_items)}
      </div>
    </section>
    {''.join(sections)}
  </main>
  {interactive_bootstrap}
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trace image-patch to 3D cross-attention in TRELLIS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python example_image_cross_attention.py\n"
            "  python example_image_cross_attention.py --input-dir assets/example_edit\n"
            "  python example_image_cross_attention.py --images assets/example_edit/2d_render.png,assets/example_edit/2d_edit.png --mask assets/example_edit/2d_mask.png\n"
        ),
    )
    parser.add_argument("--input-dir", default="assets/example_edit", help="Directory that contains the example edit assets.")
    parser.add_argument("--images", default="", help="Comma-separated conditioning image paths. If omitted, auto-detect from --input-dir.")
    parser.add_argument("--mask", default="", help="Mask image path. If omitted, auto-detect a file with 'mask' in its name.")
    parser.add_argument("--reference-image", default="", help="Optional reference image used to compute the shared crop box.")
    parser.add_argument("--model", default="microsoft/TRELLIS-image-large", help="TRELLIS image model path or Hugging Face repo.")
    parser.add_argument("--seed", type=int, default=1, help="Sampling seed.")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of generated samples.")
    parser.add_argument("--case-name", default="", help="Output case name. Defaults to the input directory name.")
    parser.add_argument("--ss-steps", type=int, default=12, help="Sparse-structure sampling steps.")
    parser.add_argument("--slat-steps", type=int, default=12, help="SLat sampling steps.")
    parser.add_argument("--ss-cfg", type=float, default=7.5, help="Sparse-structure CFG strength.")
    parser.add_argument("--slat-cfg", type=float, default=3.0, help="SLat CFG strength.")
    parser.add_argument(
        "--trace-stages",
        default="sparse_structure,slat",
        help="Comma-separated stages to trace: sparse_structure, slat.",
    )
    parser.add_argument("--query-chunk", type=int, default=2048, help="How many spatial queries to process per attention chunk.")
    parser.add_argument("--topk-spatial", type=int, default=8, help="How many top 3D spatial queries to highlight.")
    parser.add_argument(
        "--attn-backend",
        default=_attn_backend,
        help="Optional ATTN_BACKEND value to expose before importing TRELLIS.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    image_paths, mask_path = resolve_image_paths(args)
    reference_image_path = choose_reference_image(image_paths, args.reference_image)
    trace_stages = {stage.strip() for stage in args.trace_stages.split(",") if stage.strip()}

    default_case_name = sanitize_name(Path(args.input_dir).name or "image_cross_attention")
    case_name = args.case_name.strip() or default_case_name
    root_dir = ensure_dir(build_output_layout("image_cross_attention", case_name).case_dir)

    print(f"[INFO] Output directory: {root_dir}")
    print(f"[INFO] Images: {[str(path) for path in image_paths]}")
    print(f"[INFO] Mask:   {mask_path}")
    print(f"[INFO] Ref:    {reference_image_path}")

    case_manifest = {
        "method_name": "image_cross_attention",
        "case_name": case_name,
        "output_dir": str(root_dir),
        "model": args.model,
        "mask_path": str(mask_path),
        "reference_image": str(reference_image_path),
        "selection_rule": "Any patch with any mask overlap is counted as a selected visual token.",
        "subcases": [],
    }
    subcase_manifests: Dict[str, dict] = {}

    with torch.no_grad():
        print("[1/6] Loading pipeline...")
        pipeline = load_partial_image_pipeline(args.model)
        pipeline.cuda()

        print("[2/6] Building shared crop context...")
        reference_image = Image.open(reference_image_path)
        crop_context = build_crop_context(reference_image, pipeline)
        scaled_reference = scale_image(reference_image, crop_context.scale, Image.Resampling.NEAREST)
        if has_useful_alpha(scaled_reference):
            pipeline._fallback_alpha_mask = scaled_reference.getchannel("A")
        save_json(
            root_dir / "crop_context.json",
            {
                "scale": crop_context.scale,
                "bbox": list(crop_context.bbox),
                "reference_image": str(reference_image_path),
            },
        )

        for image_path in image_paths:
            subcase_slug = sanitize_name(image_path.stem)
            subcase_dir = ensure_dir(root_dir / subcase_slug)
            print(f"[3/6] Preprocessing {image_path.name} ...")

            image = Image.open(image_path)
            mask = Image.open(mask_path)
            preprocessed_image = preprocess_condition_image(image, pipeline, crop_context)
            preprocessed_mask = preprocess_mask_image(mask, crop_context)

            preprocessed_image_path = subcase_dir / "preprocessed_image.png"
            preprocessed_mask_path = subcase_dir / "preprocessed_mask.png"
            mask_overlay_path = subcase_dir / "mask_patch_overlay.png"
            preprocessed_image.save(preprocessed_image_path)
            preprocessed_mask.save(preprocessed_mask_path)
            preview_patch_grid, _ = mask_to_patch_selection(preprocessed_mask)
            save_mask_patch_overview(
                preprocessed_image,
                preprocessed_mask,
                preview_patch_grid,
                mask_overlay_path,
                title=f"{image_path.name}: mask-selected visual tokens",
            )

            print(f"[4/6] Encoding image tokens for {image_path.name} ...")
            cond = pipeline.get_cond([preprocessed_image])
            token_meta = build_image_token_metadata(cond["cond"], preprocessed_mask)
            save_json(subcase_dir / "image_tokens.json", token_meta)

            tracer = ImageCrossAttentionTracer(
                token_meta=token_meta,
                query_chunk=args.query_chunk,
                topk_spatial=args.topk_spatial,
            )

            ss_model = pipeline.models["sparse_structure_flow_model"]
            slat_model = pipeline.models["slat_flow_model"]
            tracer.prepare_stage(
                stage_name="sparse_structure",
                model=ss_model,
                cond=cond["cond"],
                neg_cond=cond["neg_cond"],
                total_steps=args.ss_steps,
                enabled="sparse_structure" in trace_stages,
                query_type="dense",
            )
            tracer.prepare_stage(
                stage_name="slat",
                model=slat_model,
                cond=cond["cond"],
                neg_cond=cond["neg_cond"],
                total_steps=args.slat_steps,
                enabled="slat" in trace_stages,
                query_type="sparse",
            )
            tracer.set_dense_grid_meta(
                stage_name="sparse_structure",
                resolution=ss_model.resolution,
                patch_size=ss_model.patch_size,
            )

            try:
                print(f"[5/6] Sampling and tracing {image_path.name} ...")
                torch.manual_seed(args.seed)
                ss_params = dict(pipeline.sparse_structure_sampler_params)
                ss_params.update({"steps": args.ss_steps, "cfg_strength": args.ss_cfg})
                coords = pipeline.sample_sparse_structure(cond, args.num_samples, ss_params)
                np.save(subcase_dir / "sparse_structure_coords.npy", coords.detach().cpu().numpy().astype(np.int16))

                slat_params = dict(pipeline.slat_sampler_params)
                slat_params.update({"steps": args.slat_steps, "cfg_strength": args.slat_cfg})
                slat = pipeline.sample_slat(cond, coords, slat_params)
                np.save(subcase_dir / "slat_coords.npy", slat.coords.detach().cpu().numpy().astype(np.int16))
                np.save(subcase_dir / "slat_feats.npy", slat.feats.detach().cpu().numpy().astype(np.float16))

                mesh_snapshot_path: Optional[Path] = None

                manifest = tracer.export(
                    root_dir=subcase_dir,
                    preprocessed_image_path=preprocessed_image_path,
                    preprocessed_mask_path=preprocessed_mask_path,
                    mask_overlay_path=mask_overlay_path,
                    mesh_snapshot_path=mesh_snapshot_path,
                )
            finally:
                tracer.restore()
                torch.cuda.empty_cache()

            save_json(
                subcase_dir / "run_config.json",
                {
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "reference_image": str(reference_image_path),
                    "model": args.model,
                    "seed": args.seed,
                    "num_samples": args.num_samples,
                    "ss_steps": args.ss_steps,
                    "slat_steps": args.slat_steps,
                    "ss_cfg": args.ss_cfg,
                    "slat_cfg": args.slat_cfg,
                    "trace_stages": sorted(trace_stages),
                    "query_chunk": args.query_chunk,
                    "topk_spatial": args.topk_spatial,
                },
            )

            case_manifest["subcases"].append(
                {
                    "id": subcase_slug,
                    "label": image_path.stem,
                    "image_path": str(image_path),
                    "trace_manifest": f"{subcase_slug}/trace_manifest.json",
                }
            )
            subcase_manifests[subcase_slug] = manifest

        print("[6/6] Writing HTML viewer and manifests...")
        save_json(root_dir / "case_manifest.json", case_manifest)
        html = render_trace_html(case_manifest, subcase_manifests)
        (root_dir / "index.html").write_text(html, encoding="utf-8")
        (root_dir / "README.txt").write_text(
            (
                "TRELLIS Image Cross-Attention Trace\n"
                "========================================\n\n"
                f"Case:   {case_name}\n"
                f"Model:  {args.model}\n"
                f"Mask:   {mask_path}\n"
                f"Ref:    {reference_image_path}\n\n"
                "Main files:\n"
                "  case_manifest.json      all traced image cases\n"
                "  crop_context.json       shared crop box used for image/mask alignment\n"
                "  index.html              static viewer for all traced cases\n"
                "  <subcase>/trace_manifest.json  per-image trace manifest\n"
                "  <subcase>/mask_patch_overlay.png  preprocessed image + selected visual tokens\n"
                "  <subcase>/<stage>/masked_region_average.png  average 3D activation of mask-selected visual tokens\n"
                "  <subcase>/<stage>/masked_region_step_progress.png  3D activation across sampling steps\n"
                "  <subcase>/<stage>/masked_region_block_progress.png  3D activation across transformer blocks\n"
            ),
            encoding="utf-8",
        )

    print(f"[DONE] Trace written to: {root_dir}")
    print(f"[DONE] Open with a local HTTP server, then visit: {root_dir / 'index.html'}")


if __name__ == "__main__":
    main()
