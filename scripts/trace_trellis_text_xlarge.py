#!/usr/bin/env python3
"""
Trace and visualize microsoft/TRELLIS-text-xlarge generation internals.

This script expands the text-to-3D pipeline into explicit stages and saves
intermediate variables in two forms whenever possible:
1. raw artifacts for exact inspection/reuse
2. human-readable visualizations

It also traces one representative dense transformer block from the sparse
structure model and one representative sparse transformer block from the SLat
model. Repeated modules are therefore shown with one concrete example instead
of dumping every repeated block.

追踪并可视化 microsoft/TRELLIS-text-xlarge 的生成内部机制。
此脚本将文本到 3D 的流水线扩展为明确的阶段，并在可能的情况下以两种形式保存中间变量：
用于精确检查 / 复用的原始工件
人类可读的可视化内容
它还会从稀疏结构模型中追踪一个有代表性的密集 Transformer 块，并从 SLat 模型中追踪一个有代表性的稀疏 Transformer 块。因此，对于重复的模块，会展示一个具体示例，而非输出每个重复的块。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trellis.modules import sparse as sp
from trellis.modules.sparse.basic import SparseTensor

os.environ.setdefault("SPCONV_ALGO", "native")


def ensure_open3d_stub() -> None:
    if "open3d" in sys.modules:
        return
    open3d_stub = types.ModuleType("open3d")
    open3d_stub.geometry = types.SimpleNamespace(
        TriangleMesh=object,
        PointCloud=object,
        VoxelGrid=object,
    )
    open3d_stub.utility = types.SimpleNamespace(Vector3dVector=lambda array: array)
    open3d_stub.io = types.SimpleNamespace(
        write_point_cloud=lambda *args, **kwargs: False,
        write_triangle_mesh=lambda *args, **kwargs: False,
    )
    sys.modules["open3d"] = open3d_stub


STEP_SPECS: Dict[int, Dict[str, Any]] = {
    1: {
        "slug": "text_conditioning",
        "paper_step": "Text conditioning",
        "code_entry": [
            "TrellisTextTo3DPipeline.get_cond",
            "TrellisTextTo3DPipeline.encode_text",
        ],
        "inputs": ["prompt"],
        "outputs": ["token ids", "attention mask", "cond", "neg_cond"],
        "intermediates": ["tokens", "token norms"],
    },
    2: {
        "slug": "sparse_structure_sampling",
        "paper_step": "Sparse structure rectified-flow sampling",
        "code_entry": [
            "TrellisTextTo3DPipeline.sample_sparse_structure",
            "trellis.pipelines.samplers.FlowEuler*Sampler",
        ],
        "inputs": ["cond", "neg_cond", "dense noise"],
        "outputs": ["per-step x_t", "pred_v", "pred_eps", "pred_x0", "x_{t-1}", "final z_s"],
        "intermediates": ["sampling timesteps"],
    },
    3: {
        "slug": "sparse_structure_decode",
        "paper_step": "Sparse structure decode",
        "code_entry": ["sparse_structure_decoder.forward"],
        "inputs": ["z_s"],
        "outputs": ["occupancy logits", "occupancy mask", "occupied coords"],
        "intermediates": [],
    },
    4: {
        "slug": "dense_block_example",
        "paper_step": "Representative dense transformer block example",
        "code_entry": ["SparseStructureFlowModel.blocks[k]"],
        "inputs": ["x", "mod", "context"],
        "outputs": ["block output", "cross-attention summary"],
        "intermediates": [
            "adaLN chunks",
            "norm outputs",
            "self-attn output",
            "cross-attn q/k/v",
            "MLP output",
        ],
    },
    5: {
        "slug": "slat_sampling",
        "paper_step": "Structured latent rectified-flow sampling",
        "code_entry": [
            "TrellisTextTo3DPipeline.sample_slat",
            "trellis.pipelines.samplers.FlowEuler*Sampler",
        ],
        "inputs": ["cond", "neg_cond", "coords", "sparse noise"],
        "outputs": ["per-step x_t", "pred_v", "pred_eps", "pred_x0", "x_{t-1}", "final raw_slat"],
        "intermediates": ["sampling timesteps"],
    },
    6: {
        "slug": "sparse_block_example",
        "paper_step": "Representative sparse transformer block example",
        "code_entry": ["SLatFlowModel.blocks[k]"],
        "inputs": ["sparse x", "mod", "context"],
        "outputs": ["block output", "cross-attention summary"],
        "intermediates": [
            "adaLN chunks",
            "norm outputs",
            "self-attn output",
            "cross-attn q/k/v",
            "MLP output",
        ],
    },
    7: {
        "slug": "slat_normalization",
        "paper_step": "SLat denormalization",
        "code_entry": ["TrellisTextTo3DPipeline.sample_slat"],
        "inputs": ["raw_slat", "mean", "std"],
        "outputs": ["normalized_slat"],
        "intermediates": [],
    },
    8: {
        "slug": "decoded_outputs",
        "paper_step": "Decode SLat to output representations",
        "code_entry": ["TrellisTextTo3DPipeline.decode_slat"],
        "inputs": ["normalized_slat"],
        "outputs": ["mesh", "gaussian", "radiance_field"],
        "intermediates": [],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace and visualize TRELLIS text-xlarge generation internals."
    )
    parser.add_argument("--prompt", required=True, help="Text prompt for TRELLIS text-to-3D.")
    parser.add_argument(
        "--model",
        default="microsoft/TRELLIS-text-xlarge",
        help="Pretrained TRELLIS model path or Hugging Face repo.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/trellis_text_xlarge_trace",
        help="Root directory for all tracing artifacts.",
    )
    parser.add_argument(
        "--case-name",
        default="",
        help="Optional case subdirectory. Defaults to a prompt-derived name.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed used before sparse-structure sampling.")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of generated samples.")
    parser.add_argument("--ss-steps", type=int, default=None, help="Override sparse-structure sampler steps.")
    parser.add_argument("--slat-steps", type=int, default=None, help="Override SLat sampler steps.")
    parser.add_argument(
        "--ss-cfg-strength",
        type=float,
        default=None,
        help="Override sparse-structure classifier-free guidance strength.",
    )
    parser.add_argument(
        "--slat-cfg-strength",
        type=float,
        default=None,
        help="Override SLat classifier-free guidance strength.",
    )
    parser.add_argument(
        "--ss-example-block",
        type=int,
        default=-1,
        help="Representative dense block index. -1 means middle block.",
    )
    parser.add_argument(
        "--slat-example-block",
        type=int,
        default=-1,
        help="Representative sparse block index. -1 means middle block.",
    )
    parser.add_argument(
        "--ss-example-step",
        type=int,
        default=-1,
        help="Representative sparse-structure sampler step. -1 means middle step.",
    )
    parser.add_argument(
        "--slat-example-step",
        type=int,
        default=-1,
        help="Representative SLat sampler step. -1 means middle step.",
    )
    parser.add_argument(
        "--topk-attn-tokens",
        type=int,
        default=6,
        help="How many top-attended tokens to plot for representative block cross-attention.",
    )
    parser.add_argument(
        "--query-chunk",
        type=int,
        default=2048,
        help="Spatial-query chunk size used when computing explicit attention maps.",
    )
    parser.add_argument(
        "--formats",
        default="mesh,gaussian,radiance_field",
        help="Comma-separated decode formats. Example: mesh,gaussian,radiance_field",
    )
    parser.add_argument(
        "--render-resolution",
        type=int,
        default=512,
        help="Resolution for rendered decode videos.",
    )
    parser.add_argument(
        "--render-frames",
        type=int,
        default=120,
        help="Number of frames for rendered decode videos.",
    )
    parser.add_argument(
        "--export-glb",
        action="store_true",
        help="Export a GLB if both mesh and gaussian are decoded.",
    )
    parser.add_argument(
        "--skip-decode",
        action="store_true",
        help="Skip final decoding to mesh/gaussian/radiance field.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def sanitize_name(text: str, max_len: int = 96) -> str:
    text = text.strip().lower()
    text = re.sub(r"</w>", "", text)
    text = re.sub(r"[^0-9a-zA-Z_\-\u4e00-\u9fff]+", "_", text)
    text = text.strip("_")
    if not text:
        text = "artifact"
    return text[:max_len]


def make_case_name(prompt: str) -> str:
    case = sanitize_name(prompt, max_len=64)
    return case or "trellis_text_trace"


def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.size == 0:
        return np.zeros_like(array, dtype=np.uint8)
    low = float(np.nanmin(array))
    high = float(np.nanmax(array))
    if not np.isfinite(low) or not np.isfinite(high) or high - low < 1e-8:
        return np.zeros_like(array, dtype=np.uint8)
    scaled = (array - low) / (high - low)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def sample_axis_indices(length: int, max_items: int) -> np.ndarray:
    if length <= max_items:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, num=max_items).round().astype(np.int64))


def sample_matrix(array: np.ndarray, max_rows: int = 256, max_cols: int = 256) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"sample_matrix expects 2D input, got shape {arr.shape}")
    rows = sample_axis_indices(arr.shape[0], max_rows)
    cols = sample_axis_indices(arr.shape[1], max_cols)
    return arr[np.ix_(rows, cols)]


def tensor_signature(tensor: torch.Tensor, rows: int = 3, cols: int = 8) -> np.ndarray:
    if tensor.ndim == 0:
        return tensor.detach().float().cpu().reshape(1, 1).numpy()
    if tensor.ndim == 1:
        return tensor.detach().float().cpu()[: cols].reshape(1, -1).numpy()
    if tensor.ndim == 2:
        return tensor.detach().float().cpu()[:rows, :cols].numpy()
    return tensor.detach().float().cpu().reshape(tensor.shape[0], -1)[:rows, :cols].numpy()


def tensor_stats_from_array(array: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(array)
    if arr.size == 0:
        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "numel": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
        }
    arr_float = arr.astype(np.float32, copy=False)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "numel": int(arr.size),
        "min": float(np.nanmin(arr_float)),
        "max": float(np.nanmax(arr_float)),
        "mean": float(np.nanmean(arr_float)),
        "std": float(np.nanstd(arr_float)),
    }


def tensor_stats(tensor: torch.Tensor) -> Dict[str, Any]:
    return tensor_stats_from_array(tensor.detach().cpu().numpy())


def save_simple_heatmap(path: Path, array: np.ndarray, upscale: int = 4) -> None:
    arr = np.asarray(array)
    if arr.ndim == 1:
        arr = arr[None, :]
    image = Image.fromarray(normalize_to_uint8(arr))
    image = image.resize(
        (max(1, image.width * upscale), max(1, image.height * upscale)),
        Image.Resampling.NEAREST,
    )
    image.save(path)


def save_line_plot(path: Path, values: np.ndarray, title: str, xlabel: str, ylabel: str) -> None:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    width = 1200
    height = 420
    pad_left = 70
    pad_right = 20
    pad_top = 30
    pad_bottom = 45
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        [pad_left, pad_top, width - pad_right, height - pad_bottom],
        outline=(190, 190, 190),
        width=1,
    )
    if arr.size > 0:
        vmin = float(arr.min())
        vmax = float(arr.max())
        if abs(vmax - vmin) < 1e-8:
            vmax = vmin + 1.0
        x_coords = np.linspace(pad_left, width - pad_right, num=arr.size)
        y_coords = pad_top + (1.0 - (arr - vmin) / (vmax - vmin)) * (height - pad_top - pad_bottom)
        points = [(float(x), float(y)) for x, y in zip(x_coords, y_coords)]
        if len(points) >= 2:
            draw.line(points, fill=(39, 93, 173), width=2)
    draw.text((pad_left, 8), title[:100], fill=(0, 0, 0))
    draw.text((width - 180, height - 22), xlabel[:24], fill=(0, 0, 0))
    draw.text((8, height // 2), ylabel[:24], fill=(0, 0, 0))
    canvas.save(path)


def save_bar_plot(path: Path, labels: List[str], values: np.ndarray, title: str) -> None:
    arr = np.asarray(values).reshape(-1)
    width = max(1200, 70 * max(len(labels), 1))
    height = 500
    pad_left = 70
    pad_right = 20
    pad_top = 30
    pad_bottom = 100
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        [pad_left, pad_top, width - pad_right, height - pad_bottom],
        outline=(190, 190, 190),
        width=1,
    )
    vmax = float(arr.max()) if arr.size > 0 else 1.0
    vmax = max(vmax, 1e-6)
    bar_w = max(8, int((width - pad_left - pad_right) / max(len(arr), 1) * 0.7))
    gap = max(2, int((width - pad_left - pad_right) / max(len(arr), 1) * 0.3))
    x = pad_left + gap
    for idx, value in enumerate(arr):
        bar_h = int((float(value) / vmax) * (height - pad_top - pad_bottom))
        draw.rectangle(
            [x, height - pad_bottom - bar_h, x + bar_w, height - pad_bottom],
            fill=(58, 124, 165),
            outline=(34, 80, 104),
        )
        if idx < len(labels):
            draw.text((x, height - pad_bottom + 6), labels[idx][:10], fill=(0, 0, 0))
        x += bar_w + gap
    draw.text((pad_left, 8), title[:100], fill=(0, 0, 0))
    canvas.save(path)


def make_weighted_projection_strip(coords: np.ndarray, values: np.ndarray, size: int = 256) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if coords.shape[0] == 0:
        blank = np.zeros((size, size), dtype=np.uint8)
        return np.concatenate([blank, blank, blank], axis=1)

    mins = coords.min(axis=0, keepdims=True)
    maxs = coords.max(axis=0, keepdims=True)
    denom = np.maximum(maxs - mins, 1e-6)
    norm = (coords - mins) / denom

    def project(a: int, b: int) -> np.ndarray:
        canvas = np.zeros((size, size), dtype=np.float32)
        xy = np.stack([norm[:, a], norm[:, b]], axis=1)
        ij = np.clip((xy * (size - 1)).round().astype(np.int32), 0, size - 1)
        for idx, (ii, jj) in enumerate(ij):
            canvas[size - 1 - jj, ii] += values[idx]
        return normalize_to_uint8(canvas)

    return np.concatenate([project(0, 1), project(0, 2), project(1, 2)], axis=1)


def make_projection_strip(coords: np.ndarray, size: int = 256) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    if coords.shape[0] == 0:
        blank = np.zeros((size, size), dtype=np.uint8)
        return np.concatenate([blank, blank, blank], axis=1)

    mins = coords.min(axis=0, keepdims=True)
    maxs = coords.max(axis=0, keepdims=True)
    denom = np.maximum(maxs - mins, 1e-6)
    norm = (coords - mins) / denom

    def project(a: int, b: int) -> np.ndarray:
        canvas = np.zeros((size, size), dtype=np.uint8)
        xy = np.stack([norm[:, a], norm[:, b]], axis=1)
        ij = np.clip((xy * (size - 1)).round().astype(np.int32), 0, size - 1)
        canvas[size - 1 - ij[:, 1], ij[:, 0]] = 255
        return canvas

    xy = project(0, 1)
    xz = project(0, 2)
    yz = project(1, 2)
    return np.concatenate([xy, xz, yz], axis=1)


def compute_feature_colors(features: np.ndarray, max_rows: int = 4096, max_dims: int = 64) -> np.ndarray:
    feats = np.asarray(features, dtype=np.float32)
    if feats.ndim != 2 or feats.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    dim_idx = sample_axis_indices(feats.shape[1], min(max_dims, feats.shape[1]))
    feats_small = feats[:, dim_idx]
    row_idx = sample_axis_indices(feats_small.shape[0], min(max_rows, feats_small.shape[0]))
    sample = feats_small[row_idx]
    sample_center = sample.mean(axis=0, keepdims=True)
    centered_sample = sample - sample_center

    try:
        _, _, vh = np.linalg.svd(centered_sample, full_matrices=False)
        basis = vh[:3].T
        proj = (feats_small - sample_center) @ basis
    except np.linalg.LinAlgError:
        proj = feats_small[:, : min(3, feats_small.shape[1])]

    if proj.shape[1] < 3:
        proj = np.pad(proj, ((0, 0), (0, 3 - proj.shape[1])))

    proj_min = proj.min(axis=0, keepdims=True)
    proj_max = proj.max(axis=0, keepdims=True)
    denom = np.maximum(proj_max - proj_min, 1e-6)
    return (proj - proj_min) / denom


def save_point_cloud(path: Path, points: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
    pts = np.asarray(points, dtype=np.float32)
    cols = None
    if colors is not None and len(colors) == len(points):
        cols = np.clip(np.asarray(colors, dtype=np.float32), 0.0, 1.0)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if cols is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        for idx, point in enumerate(pts):
            if cols is None:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")
            else:
                rgb = (cols[idx] * 255.0).round().astype(np.uint8)
                f.write(f"{point[0]} {point[1]} {point[2]} {int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n")


def save_mesh(path: Path, mesh: Any) -> None:
    vertices = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {vertices.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {faces.shape[0]}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for vertex in vertices:
            f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def save_video(path: Path, frames: List[np.ndarray], fps: int = 30) -> None:
    try:
        import imageio
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Saving videos requires the optional dependency 'imageio'. "
            "Install it or rerun with --skip-decode."
        ) from exc
    imageio.mimsave(path, frames, fps=fps)


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
    keep_map: bool = True,
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


def save_spatial_panels(
    panels: List[Tuple[str, np.ndarray, np.ndarray]],
    path: Path,
    title: str,
) -> None:
    if not panels:
        return
    rendered = []
    for panel_title, coords, values in panels:
        strip = make_weighted_projection_strip(coords, values, size=256)
        panel = Image.new("RGB", (strip.shape[1], strip.shape[0] + 26), "white")
        panel.paste(Image.fromarray(strip).convert("RGB"), (0, 26))
        draw = ImageDraw.Draw(panel)
        draw.text((4, 4), panel_title[:40], fill=(0, 0, 0))
        rendered.append(np.array(panel))
    canvas = Image.new("RGB", (rendered[0].shape[1] * len(rendered), rendered[0].shape[0] + 28), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), title[:120], fill=(0, 0, 0))
    offset_x = 0
    for panel in rendered:
        canvas.paste(Image.fromarray(panel), (offset_x, 28))
        offset_x += panel.shape[1]
    canvas.save(path)


def render_volume_panels(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 5:
        arr = arr[0]
    if arr.ndim == 4:
        channel0 = arr[0]
        channel_mean = arr.mean(axis=0)
    elif arr.ndim == 3:
        channel0 = arr
        channel_mean = arr
    else:
        raise ValueError(f"render_volume_panels expects 3D/4D/5D input, got {arr.shape}")

    def panels(volume: np.ndarray) -> List[np.ndarray]:
        d, h, w = volume.shape
        return [
            volume[d // 2],
            volume[:, h // 2, :],
            volume[:, :, w // 2],
            volume.max(axis=0),
            volume.max(axis=1),
            volume.max(axis=2),
        ]

    parts = []
    for label, volume in [("c0", channel0), ("mean", channel_mean)]:
        ims = []
        for panel in panels(volume):
            panel_img = Image.fromarray(normalize_to_uint8(panel))
            panel_img = panel_img.resize((256, 256), Image.Resampling.NEAREST)
            ims.append(np.array(panel_img))
        row_top = np.concatenate(ims[:3], axis=1)
        row_bottom = np.concatenate(ims[3:], axis=1)
        strip = np.concatenate([row_top, row_bottom], axis=0)
        parts.append(strip)
    return np.concatenate(parts, axis=1)


def clean_token_label(token: str) -> str:
    label = token.replace("</w>", "")
    label = label.replace("<|startoftext|>", "[BOS]")
    label = label.replace("<|endoftext|>", "[EOS]")
    label = label.strip()
    return label if label else token


def build_word_groups(
    raw_tokens: List[str],
    display_tokens: List[str],
    special_mask: List[bool],
    valid_token_count: int,
) -> Tuple[List[str], List[List[int]]]:
    word_labels: List[str] = []
    word_token_indices: List[List[int]] = []
    current_label_parts: List[str] = []
    current_indices: List[int] = []

    def flush_current() -> None:
        if not current_indices:
            return
        label = "".join(current_label_parts).strip()
        if re.search(r"[0-9A-Za-z\u4e00-\u9fff]", label):
            word_labels.append(label)
            word_token_indices.append(current_indices.copy())
        current_label_parts.clear()
        current_indices.clear()

    for idx in range(valid_token_count):
        if special_mask[idx]:
            flush_current()
            continue
        token = raw_tokens[idx]
        current_label_parts.append(display_tokens[idx])
        current_indices.append(idx)
        if token.endswith("</w>"):
            flush_current()

    flush_current()
    return word_labels, word_token_indices


def build_token_metadata(tokenizer: Any, prompt: str) -> Dict[str, Any]:
    encoding = tokenizer(
        [prompt],
        max_length=77,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"][0].tolist()
    attention_mask = encoding["attention_mask"][0].tolist()
    raw_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    display_tokens = [clean_token_label(token) for token in raw_tokens]
    valid_token_count = int(sum(attention_mask))

    special_mask: List[bool] = []
    for idx, token in enumerate(raw_tokens):
        is_special = (
            token in getattr(tokenizer, "all_special_tokens", [])
            or token in {"<|startoftext|>", "<|endoftext|>"}
            or idx >= valid_token_count
        )
        special_mask.append(bool(is_special))

    word_labels, word_token_indices = build_word_groups(
        raw_tokens=raw_tokens,
        display_tokens=display_tokens,
        special_mask=special_mask,
        valid_token_count=valid_token_count,
    )

    return {
        "prompt": prompt,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "raw_tokens": raw_tokens,
        "display_tokens": display_tokens,
        "valid_token_count": valid_token_count,
        "special_mask": special_mask,
        "word_labels": word_labels,
        "word_token_indices": word_token_indices,
    }


def select_focus_token_indices(token_meta: Dict[str, Any], reference_scores: np.ndarray, topk: int) -> List[int]:
    valid_token_count = token_meta["valid_token_count"]
    special_mask = token_meta["special_mask"]
    display_tokens = token_meta["display_tokens"]

    candidates = []
    for idx in range(valid_token_count):
        if special_mask[idx]:
            continue
        if re.search(r"[0-9A-Za-z\u4e00-\u9fff]", display_tokens[idx]):
            candidates.append(idx)
    if not candidates:
        candidates = [idx for idx in range(valid_token_count) if not special_mask[idx]]
    if not candidates:
        return []
    ranked = sorted(candidates, key=lambda idx: float(reference_scores[idx]), reverse=True)
    return ranked[:topk]


class ArtifactWriter:
    def __init__(self, root_dir: Path, step_specs: Dict[int, Dict[str, Any]]) -> None:
        self.root_dir = ensure_dir(root_dir)
        self.step_specs = step_specs
        self.step_dirs = {
            step: ensure_dir(self.root_dir / f"step_{step:02d}_{spec['slug']}")
            for step, spec in step_specs.items()
        }
        self.role_counters: Dict[Tuple[int, str], int] = defaultdict(int)
        self.manifest: Dict[str, Any] = {
            "root_dir": str(self.root_dir),
            "steps": {
                str(step): {
                    "slug": spec["slug"],
                    "paper_step": spec["paper_step"],
                    "code_entry": spec["code_entry"],
                    "artifacts": [],
                }
                for step, spec in step_specs.items()
            },
            "artifacts": [],
        }

    def _next_stem(self, step: int, role: str, logical_name: str) -> Tuple[Path, int]:
        key = (step, role)
        self.role_counters[key] += 1
        index = self.role_counters[key]
        stem = self.step_dirs[step] / f"第{step}步_{role}{index}_{sanitize_name(logical_name)}"
        return stem, index

    def _register(
        self,
        step: int,
        role: str,
        index: int,
        logical_name: str,
        artifact_type: str,
        files: Dict[str, str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = {
            "step": step,
            "role": role,
            "index": index,
            "logical_name": logical_name,
            "artifact_type": artifact_type,
            "files": files,
            "metadata": metadata or {},
        }
        self.manifest["artifacts"].append(entry)
        self.manifest["steps"][str(step)]["artifacts"].append(entry)

    def save_json_payload(
        self,
        step: int,
        role: str,
        logical_name: str,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        stem, index = self._next_stem(step, role, logical_name)
        raw_path = stem.with_suffix(".json")
        save_json(raw_path, payload)
        self._register(
            step=step,
            role=role,
            index=index,
            logical_name=logical_name,
            artifact_type="json",
            files={"json": str(raw_path.relative_to(self.root_dir))},
            metadata=metadata,
        )
        return raw_path

    def save_tensor(
        self,
        step: int,
        role: str,
        logical_name: str,
        tensor: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        array = tensor.detach().cpu().numpy()
        stem, index = self._next_stem(step, role, logical_name)
        raw_path = stem.with_suffix(".npy")
        stats_path = stem.with_name(stem.name + "_stats.json")
        np.save(raw_path, array)
        stats = tensor_stats_from_array(array)
        if torch.is_tensor(tensor):
            stats["device"] = str(tensor.device)
        save_json(stats_path, {**stats, **(metadata or {})})

        files = {
            "raw": str(raw_path.relative_to(self.root_dir)),
            "stats": str(stats_path.relative_to(self.root_dir)),
        }

        if array.ndim in (1, 2):
            vis_path = stem.with_name(stem.name + "_heatmap.png")
            vis_arr = sample_matrix(array, max_rows=256, max_cols=256)
            save_simple_heatmap(vis_path, vis_arr, upscale=4)
            files["visualization"] = str(vis_path.relative_to(self.root_dir))
            if array.ndim == 1:
                line_path = stem.with_name(stem.name + "_curve.png")
                save_line_plot(
                    line_path,
                    array,
                    title=logical_name,
                    xlabel="Index",
                    ylabel="Value",
                )
                files["curve"] = str(line_path.relative_to(self.root_dir))
        elif array.ndim in (3, 4, 5):
            vis_path = stem.with_name(stem.name + "_panels.png")
            Image.fromarray(render_volume_panels(array)).save(vis_path)
            files["visualization"] = str(vis_path.relative_to(self.root_dir))
        elif array.ndim == 0:
            text_path = stem.with_name(stem.name + "_value.json")
            save_json(text_path, {"value": float(array.item())})
            files["value"] = str(text_path.relative_to(self.root_dir))

        self._register(
            step=step,
            role=role,
            index=index,
            logical_name=logical_name,
            artifact_type="tensor",
            files=files,
            metadata=metadata,
        )
        return raw_path

    def save_sparse_tensor(
        self,
        step: int,
        role: str,
        logical_name: str,
        tensor: SparseTensor,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        coords = tensor.coords.detach().cpu().numpy()
        feats = tensor.feats.detach().cpu().numpy()
        stem, index = self._next_stem(step, role, logical_name)

        coords_path = stem.with_name(stem.name + "_coords.npy")
        feats_path = stem.with_name(stem.name + "_feats.npy")
        summary_path = stem.with_name(stem.name + "_summary.json")
        np.save(coords_path, coords)
        np.save(feats_path, feats)

        summary = {
            "coords": tensor_stats_from_array(coords),
            "feats": tensor_stats_from_array(feats),
            "device": str(tensor.device),
        }
        if metadata:
            summary.update(metadata)
        save_json(summary_path, summary)

        files = {
            "coords": str(coords_path.relative_to(self.root_dir)),
            "feats": str(feats_path.relative_to(self.root_dir)),
            "summary": str(summary_path.relative_to(self.root_dir)),
        }

        xyz = coords[:, -3:].astype(np.float32) if coords.size else np.zeros((0, 3), dtype=np.float32)
        projection_path = stem.with_name(stem.name + "_coords_projections.png")
        Image.fromarray(make_projection_strip(xyz, size=256)).save(projection_path)
        files["projection"] = str(projection_path.relative_to(self.root_dir))

        heatmap_path = stem.with_name(stem.name + "_feats_heatmap.png")
        heatmap_arr = sample_matrix(feats, max_rows=256, max_cols=256) if feats.ndim == 2 else sample_matrix(feats.reshape(feats.shape[0], -1), max_rows=256, max_cols=256)
        save_simple_heatmap(heatmap_path, heatmap_arr, upscale=4)
        files["heatmap"] = str(heatmap_path.relative_to(self.root_dir))

        if xyz.shape[0] > 0:
            ply_path = stem.with_name(stem.name + "_pca_colors.ply")
            colors = compute_feature_colors(feats.reshape(feats.shape[0], -1))
            save_point_cloud(ply_path, xyz, colors)
            files["point_cloud"] = str(ply_path.relative_to(self.root_dir))

        self._register(
            step=step,
            role=role,
            index=index,
            logical_name=logical_name,
            artifact_type="sparse_tensor",
            files=files,
            metadata=metadata,
        )
        return coords_path

    def save_attention_bundle(
        self,
        step: int,
        role: str,
        logical_name: str,
        attn_map: np.ndarray,
        coords: np.ndarray,
        token_meta: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        topk_tokens: int = 6,
    ) -> Path:
        valid = token_meta["valid_token_count"]
        labels = token_meta["display_tokens"][:valid]
        attn_arr = np.asarray(attn_map, dtype=np.float32)[:, :valid]
        coords_arr = np.asarray(coords, dtype=np.int16)

        stem, index = self._next_stem(step, role, logical_name)
        npy_path = stem.with_name(stem.name + "_attn_map.npy")
        coords_path = stem.with_name(stem.name + "_coords.npy")
        stats_path = stem.with_name(stem.name + "_summary.json")
        np.save(npy_path, attn_arr)
        np.save(coords_path, coords_arr)

        token_mean = attn_arr.mean(axis=0) if attn_arr.size else np.zeros((valid,), dtype=np.float32)
        focus_indices = select_focus_token_indices(token_meta, token_mean, topk_tokens)
        summary = {
            "shape": list(attn_arr.shape),
            "token_labels": labels,
            "focus_indices": focus_indices,
            "focus_labels": [labels[idx] for idx in focus_indices],
            "token_mean": token_mean.tolist(),
        }
        if metadata:
            summary.update(metadata)
        save_json(stats_path, summary)

        files = {
            "attn_map": str(npy_path.relative_to(self.root_dir)),
            "coords": str(coords_path.relative_to(self.root_dir)),
            "summary": str(stats_path.relative_to(self.root_dir)),
        }

        bar_path = stem.with_name(stem.name + "_token_mean_bar.png")
        save_bar_plot(bar_path, labels, token_mean, title=logical_name)
        files["token_mean_bar"] = str(bar_path.relative_to(self.root_dir))

        if focus_indices:
            panels = [
                (
                    f"{labels[idx]} ({token_mean[idx]:.4f})",
                    coords_arr,
                    attn_arr[:, idx],
                )
                for idx in focus_indices
            ]
            panel_path = stem.with_name(stem.name + "_focus_tokens.png")
            save_spatial_panels(
                panels,
                panel_path,
                title=f"{logical_name}: representative token-to-space attention",
            )
            files["focus_tokens"] = str(panel_path.relative_to(self.root_dir))

        self._register(
            step=step,
            role=role,
            index=index,
            logical_name=logical_name,
            artifact_type="attention",
            files=files,
            metadata=metadata,
        )
        return npy_path

    def write_manifest(self) -> Path:
        manifest_path = self.root_dir / "trace_manifest.json"
        save_json(manifest_path, self.manifest)
        return manifest_path


def resolve_example_index(specified: int, total: int) -> int:
    if total <= 0:
        return 0
    if specified < 0:
        return total // 2
    return max(0, min(specified, total - 1))


def merge_sampler_params(
    defaults: Dict[str, Any],
    steps_override: Optional[int] = None,
    cfg_override: Optional[float] = None,
) -> Dict[str, Any]:
    params = dict(defaults)
    if steps_override is not None:
        params["steps"] = steps_override
    if cfg_override is not None:
        params["cfg_strength"] = cfg_override
    return params


class RepresentativeBlockTracer:
    def __init__(
        self,
        writer: ArtifactWriter,
        token_meta: Dict[str, Any],
        topk_attn_tokens: int,
        query_chunk: int,
    ) -> None:
        self.writer = writer
        self.token_meta = token_meta
        self.topk_attn_tokens = topk_attn_tokens
        self.query_chunk = query_chunk
        self.restore_stack: List[Tuple[object, str, object]] = []
        self.stage_specs: Dict[str, Dict[str, Any]] = {}
        self.current_runtime: Dict[str, Any] = {}
        self.current_forward_context: Optional[Dict[str, Any]] = None

    def prepare_stage(
        self,
        stage_name: str,
        model: torch.nn.Module,
        cond: torch.Tensor,
        neg_cond: torch.Tensor,
        selected_step: int,
        selected_block: int,
        step_id: int,
        resolution: Optional[int] = None,
        patch_size: Optional[int] = None,
    ) -> None:
        self.stage_specs[stage_name] = {
            "selected_step": int(selected_step),
            "selected_block": int(selected_block),
            "step_id": int(step_id),
            "resolution": resolution,
            "patch_size": patch_size,
            "cond_signature": tensor_signature(cond),
            "neg_signature": tensor_signature(neg_cond),
            "captured": False,
        }
        self._patch_model_forward(stage_name, model)
        block = model.blocks[selected_block]
        if stage_name == "sparse_structure":
            self._patch_dense_block(stage_name, block)
        elif stage_name == "slat":
            self._patch_sparse_block(stage_name, block)
        else:
            raise ValueError(f"Unknown stage: {stage_name}")

    def set_runtime_context(self, stage_name: str, step_index: int, timestep: float) -> None:
        self.current_runtime = {
            "stage_name": stage_name,
            "step_index": int(step_index),
            "timestep": float(timestep),
        }

    def clear_runtime_context(self) -> None:
        self.current_runtime = {}
        self.current_forward_context = None

    def restore(self) -> None:
        while self.restore_stack:
            obj, attr, value = self.restore_stack.pop()
            setattr(obj, attr, value)

    def _infer_pass_kind(self, stage_name: str, cond: torch.Tensor) -> str:
        state = self.stage_specs[stage_name]
        current_sig = tensor_signature(cond)
        dist_cond = float(np.max(np.abs(current_sig - state["cond_signature"])))
        dist_neg = float(np.max(np.abs(current_sig - state["neg_signature"])))
        return "cond" if dist_cond <= dist_neg else "neg"

    def _patch_model_forward(self, stage_name: str, model: torch.nn.Module) -> None:
        original_forward = model.forward
        self.restore_stack.append((model, "forward", original_forward))
        tracer = self

        def wrapped_forward(model_self, x, t, cond):
            if tracer.current_runtime.get("stage_name") != stage_name:
                return original_forward(x, t, cond)
            tracer.current_forward_context = {
                "stage_name": stage_name,
                "step_index": tracer.current_runtime["step_index"],
                "timestep": tracer.current_runtime["timestep"],
                "pass_kind": tracer._infer_pass_kind(stage_name, cond),
            }
            try:
                return original_forward(x, t, cond)
            finally:
                tracer.current_forward_context = None

        model.forward = types.MethodType(wrapped_forward, model)

    def _patch_dense_block(self, stage_name: str, block: torch.nn.Module) -> None:
        original_forward = block.forward
        self.restore_stack.append((block, "forward", original_forward))
        tracer = self

        def wrapped_forward(block_self, x, mod, context):
            spec = tracer.stage_specs[stage_name]
            ctx = tracer.current_forward_context
            should_capture = (
                ctx is not None
                and ctx["stage_name"] == stage_name
                and ctx["pass_kind"] == "cond"
                and ctx["step_index"] == spec["selected_step"]
                and not spec["captured"]
            )
            if not should_capture:
                return original_forward(x, mod, context)

            step_id = spec["step_id"]
            metadata = {
                "stage_name": stage_name,
                "selected_block": spec["selected_block"],
                "selected_step": spec["selected_step"],
                "timestep": ctx["timestep"],
            }
            tracer.writer.save_json_payload(step_id, "input", f"{stage_name}_block_example_meta", metadata)
            tracer.writer.save_tensor(step_id, "input", f"{stage_name}_block_input_x", x, metadata)
            tracer.writer.save_tensor(step_id, "input", f"{stage_name}_block_input_mod", mod, metadata)
            tracer.writer.save_tensor(step_id, "input", f"{stage_name}_block_input_context", context, metadata)

            if block_self.share_mod:
                ada = mod
            else:
                ada = block_self.adaLN_modulation(mod)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada.chunk(6, dim=1)

            h_norm1 = block_self.norm1(x)
            h_msa_in = h_norm1 * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
            h_self = block_self.self_attn(h_msa_in)
            h_self_gated = h_self * gate_msa.unsqueeze(1)
            x_after_self = x + h_self_gated
            h_norm2 = block_self.norm2(x_after_self)

            cross_attn = block_self.cross_attn
            q = cross_attn.to_q(h_norm2).reshape(h_norm2.shape[0], h_norm2.shape[1], cross_attn.num_heads, -1)
            kv = cross_attn.to_kv(context).reshape(context.shape[0], context.shape[1], 2, cross_attn.num_heads, -1)
            k, v = kv.unbind(dim=2)
            if cross_attn.qk_rms_norm:
                q = cross_attn.q_rms_norm(q)
                k = cross_attn.k_rms_norm(k)
            _, attn_map = compute_headmean_attention(
                q=q[0],
                k=k[0],
                query_chunk=self.query_chunk,
                keep_map=True,
            )
            h_cross = cross_attn(h_norm2, context)
            x_after_cross = x_after_self + h_cross
            h_norm3 = block_self.norm3(x_after_cross)
            h_mlp_in = h_norm3 * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
            h_mlp = block_self.mlp(h_mlp_in)
            h_mlp_gated = h_mlp * gate_mlp.unsqueeze(1)
            out = x_after_cross + h_mlp_gated

            for name, tensor in [
                ("adaln_full", ada),
                ("shift_msa", shift_msa),
                ("scale_msa", scale_msa),
                ("gate_msa", gate_msa),
                ("shift_mlp", shift_mlp),
                ("scale_mlp", scale_mlp),
                ("gate_mlp", gate_mlp),
                ("norm1_out", h_norm1),
                ("msa_modulated_input", h_msa_in),
                ("self_attn_out", h_self),
                ("self_attn_gated", h_self_gated),
                ("after_self_attn", x_after_self),
                ("norm2_out", h_norm2),
                ("cross_attn_q", q),
                ("cross_attn_k", k),
                ("cross_attn_v", v),
                ("cross_attn_out", h_cross),
                ("after_cross_attn", x_after_cross),
                ("norm3_out", h_norm3),
                ("mlp_modulated_input", h_mlp_in),
                ("mlp_out", h_mlp),
                ("mlp_gated", h_mlp_gated),
                ("block_out", out),
            ]:
                tracer.writer.save_tensor(step_id, "output", f"{stage_name}_{name}", tensor, metadata)

            coords = dense_query_coords(
                resolution=int(spec["resolution"]),
                patch_size=int(spec["patch_size"]),
                batch_size=x.shape[0],
            )[0]
            if attn_map is not None:
                tracer.writer.save_attention_bundle(
                    step=step_id,
                    role="output",
                    logical_name=f"{stage_name}_cross_attention_map",
                    attn_map=attn_map,
                    coords=coords,
                    token_meta=tracer.token_meta,
                    metadata=metadata,
                    topk_tokens=tracer.topk_attn_tokens,
                )

            spec["captured"] = True
            return out

        block.forward = types.MethodType(wrapped_forward, block)

    def _patch_sparse_block(self, stage_name: str, block: torch.nn.Module) -> None:
        original_forward = block.forward
        self.restore_stack.append((block, "forward", original_forward))
        tracer = self

        def wrapped_forward(block_self, x, mod, context):
            spec = tracer.stage_specs[stage_name]
            ctx = tracer.current_forward_context
            should_capture = (
                ctx is not None
                and ctx["stage_name"] == stage_name
                and ctx["pass_kind"] == "cond"
                and ctx["step_index"] == spec["selected_step"]
                and not spec["captured"]
            )
            if not should_capture:
                return original_forward(x, mod, context)

            step_id = spec["step_id"]
            metadata = {
                "stage_name": stage_name,
                "selected_block": spec["selected_block"],
                "selected_step": spec["selected_step"],
                "timestep": ctx["timestep"],
            }
            tracer.writer.save_json_payload(step_id, "input", f"{stage_name}_block_example_meta", metadata)
            tracer.writer.save_sparse_tensor(step_id, "input", f"{stage_name}_block_input_x", x, metadata)
            tracer.writer.save_tensor(step_id, "input", f"{stage_name}_block_input_mod", mod, metadata)
            tracer.writer.save_tensor(step_id, "input", f"{stage_name}_block_input_context", context, metadata)

            if block_self.share_mod:
                ada = mod
            else:
                ada = block_self.adaLN_modulation(mod)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada.chunk(6, dim=1)

            h_norm1 = x.replace(block_self.norm1(x.feats))
            h_msa_in = h_norm1 * (1 + scale_msa) + shift_msa
            h_self = block_self.self_attn(h_msa_in)
            h_self_gated = h_self * gate_msa
            x_after_self = x + h_self_gated
            h_norm2 = x_after_self.replace(block_self.norm2(x_after_self.feats))

            cross_attn = block_self.cross_attn
            q = cross_attn._linear(cross_attn.to_q, h_norm2)
            q = cross_attn._reshape_chs(q, (cross_attn.num_heads, -1))
            kv = cross_attn._linear(cross_attn.to_kv, context)
            kv = cross_attn._fused_pre(kv, num_fused=2)
            k, v = kv.unbind(dim=2)
            if cross_attn.qk_rms_norm:
                q = cross_attn.q_rms_norm(q)
                k = cross_attn.k_rms_norm(k)

            q_slice = q.feats[q.layout[0]]
            _, attn_map = compute_headmean_attention(
                q=q_slice,
                k=k[0],
                query_chunk=self.query_chunk,
                keep_map=True,
            )

            h_cross = cross_attn(h_norm2, context)
            x_after_cross = x_after_self + h_cross
            h_norm3 = x_after_cross.replace(block_self.norm3(x_after_cross.feats))
            h_mlp_in = h_norm3 * (1 + scale_mlp) + shift_mlp
            h_mlp = block_self.mlp(h_mlp_in)
            h_mlp_gated = h_mlp * gate_mlp
            out = x_after_cross + h_mlp_gated

            for name, value in [
                ("adaln_full", ada),
                ("shift_msa", shift_msa),
                ("scale_msa", scale_msa),
                ("gate_msa", gate_msa),
                ("shift_mlp", shift_mlp),
                ("scale_mlp", scale_mlp),
                ("gate_mlp", gate_mlp),
            ]:
                tracer.writer.save_tensor(step_id, "output", f"{stage_name}_{name}", value, metadata)

            for name, value in [
                ("norm1_out", h_norm1),
                ("msa_modulated_input", h_msa_in),
                ("self_attn_out", h_self),
                ("self_attn_gated", h_self_gated),
                ("after_self_attn", x_after_self),
                ("norm2_out", h_norm2),
                ("cross_attn_q", q),
                ("cross_attn_out", h_cross),
                ("after_cross_attn", x_after_cross),
                ("norm3_out", h_norm3),
                ("mlp_modulated_input", h_mlp_in),
                ("mlp_out", h_mlp),
                ("mlp_gated", h_mlp_gated),
                ("block_out", out),
            ]:
                tracer.writer.save_sparse_tensor(step_id, "output", f"{stage_name}_{name}", value, metadata)

            tracer.writer.save_tensor(step_id, "output", f"{stage_name}_cross_attn_k", k, metadata)
            tracer.writer.save_tensor(step_id, "output", f"{stage_name}_cross_attn_v", v, metadata)

            if attn_map is not None:
                coords = q.coords[q.layout[0], 1:].detach().cpu().numpy().astype(np.int16)
                tracer.writer.save_attention_bundle(
                    step=step_id,
                    role="output",
                    logical_name=f"{stage_name}_cross_attention_map",
                    attn_map=attn_map,
                    coords=coords,
                    token_meta=tracer.token_meta,
                    metadata=metadata,
                    topk_tokens=tracer.topk_attn_tokens,
                )

            spec["captured"] = True
            return out

        block.forward = types.MethodType(wrapped_forward, block)


def trace_dense_sampler(
    writer: ArtifactWriter,
    tracer: RepresentativeBlockTracer,
    pipeline: TrellisTextTo3DPipeline,
    cond: Dict[str, torch.Tensor],
    sampler_params: Dict[str, Any],
) -> torch.Tensor:
    flow_model = pipeline.models["sparse_structure_flow_model"]
    sampler = pipeline.sparse_structure_sampler
    reso = flow_model.resolution
    steps = int(sampler_params.get("steps", 50))
    rescale_t = float(sampler_params.get("rescale_t", 1.0))

    noise = torch.randn(
        cond["cond"].shape[0] if cond["cond"].shape[0] > 1 else 1,
        flow_model.in_channels,
        reso,
        reso,
        reso,
        device=pipeline.device,
    )
    if noise.shape[0] != 1:
        noise = noise[:1]
    if cond["cond"].shape[0] == 1 and sampler_params.get("num_samples", None) is not None:
        pass
    if sampler_params.get("_num_samples") is not None and sampler_params["_num_samples"] > 1:
        noise = torch.randn(
            sampler_params["_num_samples"],
            flow_model.in_channels,
            reso,
            reso,
            reso,
            device=pipeline.device,
        )

    writer.save_tensor(
        step=2,
        role="input",
        logical_name="sparse_structure_initial_noise",
        tensor=noise,
        metadata={"source_function": "sample_sparse_structure"},
    )

    params_for_model = {
        k: v
        for k, v in sampler_params.items()
        if k not in {"steps", "rescale_t", "verbose", "_num_samples"}
    }

    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    sample = noise
    for step_idx, (t, t_prev) in enumerate((t_seq[i], t_seq[i + 1]) for i in range(steps)):
        tracer.set_runtime_context("sparse_structure", step_idx, float(t))
        pred_x0, pred_eps, pred_v = sampler._get_model_prediction(
            flow_model,
            sample,
            float(t),
            cond=cond["cond"],
            neg_cond=cond["neg_cond"],
            **params_for_model,
        )
        pred_x_prev = sample - (float(t) - float(t_prev)) * pred_v

        meta = {
            "sampler_step": step_idx,
            "t": float(t),
            "t_prev": float(t_prev),
            "source_function": "FlowEulerSampler._get_model_prediction",
        }
        writer.save_tensor(2, "output", f"sparse_structure_step_{step_idx:02d}_x_t", sample, meta)
        writer.save_tensor(2, "output", f"sparse_structure_step_{step_idx:02d}_pred_v", pred_v, meta)
        writer.save_tensor(2, "output", f"sparse_structure_step_{step_idx:02d}_pred_eps", pred_eps, meta)
        writer.save_tensor(2, "output", f"sparse_structure_step_{step_idx:02d}_pred_x0", pred_x0, meta)
        writer.save_tensor(2, "output", f"sparse_structure_step_{step_idx:02d}_x_prev", pred_x_prev, meta)
        sample = pred_x_prev

    tracer.clear_runtime_context()
    writer.save_tensor(
        step=2,
        role="output",
        logical_name="sparse_structure_final_latent_z_s",
        tensor=sample,
        metadata={"source_function": "sample_sparse_structure"},
    )
    return sample


def decode_sparse_structure(
    writer: ArtifactWriter,
    pipeline: TrellisTextTo3DPipeline,
    z_s: torch.Tensor,
) -> torch.Tensor:
    decoder = pipeline.models["sparse_structure_decoder"]
    logits = decoder(z_s)
    occupancy = logits > 0
    coords = torch.argwhere(occupancy)[:, [0, 2, 3, 4]].int()

    writer.save_tensor(
        step=3,
        role="input",
        logical_name="sparse_structure_latent_z_s",
        tensor=z_s,
        metadata={"source_function": "sparse_structure_decoder"},
    )
    writer.save_tensor(
        step=3,
        role="output",
        logical_name="occupancy_logits",
        tensor=logits,
        metadata={"source_function": "sparse_structure_decoder"},
    )
    writer.save_tensor(
        step=3,
        role="output",
        logical_name="occupancy_mask",
        tensor=occupancy.float(),
        metadata={"source_function": "occupancy > 0"},
    )
    writer.save_tensor(
        step=3,
        role="output",
        logical_name="occupied_coords",
        tensor=coords,
        metadata={"source_function": "torch.argwhere"},
    )

    coords_np = coords.detach().cpu().numpy().astype(np.float32)
    if coords_np.size > 0:
        stem = writer.save_json_payload(
            step=3,
            role="output",
            logical_name="occupied_coords_summary",
            payload={
                "num_coords": int(coords_np.shape[0]),
                "coord_min": coords_np.min(axis=0).tolist(),
                "coord_max": coords_np.max(axis=0).tolist(),
            },
            metadata={"source_function": "torch.argwhere"},
        )
        del stem

    return coords


def trace_sparse_sampler(
    writer: ArtifactWriter,
    tracer: RepresentativeBlockTracer,
    pipeline: TrellisTextTo3DPipeline,
    cond: Dict[str, torch.Tensor],
    coords: torch.Tensor,
    sampler_params: Dict[str, Any],
) -> SparseTensor:
    flow_model = pipeline.models["slat_flow_model"]
    sampler = pipeline.slat_sampler
    steps = int(sampler_params.get("steps", 50))
    rescale_t = float(sampler_params.get("rescale_t", 1.0))

    noise = sp.SparseTensor(
        feats=torch.randn(coords.shape[0], flow_model.in_channels, device=pipeline.device),
        coords=coords,
    )
    writer.save_sparse_tensor(
        step=5,
        role="input",
        logical_name="slat_initial_noise",
        tensor=noise,
        metadata={"source_function": "sample_slat"},
    )

    params_for_model = {
        k: v
        for k, v in sampler_params.items()
        if k not in {"steps", "rescale_t", "verbose"}
    }

    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    sample = noise
    for step_idx, (t, t_prev) in enumerate((t_seq[i], t_seq[i + 1]) for i in range(steps)):
        tracer.set_runtime_context("slat", step_idx, float(t))
        pred_x0, pred_eps, pred_v = sampler._get_model_prediction(
            flow_model,
            sample,
            float(t),
            cond=cond["cond"],
            neg_cond=cond["neg_cond"],
            **params_for_model,
        )
        pred_x_prev = sample - (float(t) - float(t_prev)) * pred_v

        meta = {
            "sampler_step": step_idx,
            "t": float(t),
            "t_prev": float(t_prev),
            "source_function": "FlowEulerSampler._get_model_prediction",
        }
        writer.save_sparse_tensor(5, "output", f"slat_step_{step_idx:02d}_x_t", sample, meta)
        writer.save_sparse_tensor(5, "output", f"slat_step_{step_idx:02d}_pred_v", pred_v, meta)
        writer.save_sparse_tensor(5, "output", f"slat_step_{step_idx:02d}_pred_eps", pred_eps, meta)
        writer.save_sparse_tensor(5, "output", f"slat_step_{step_idx:02d}_pred_x0", pred_x0, meta)
        writer.save_sparse_tensor(5, "output", f"slat_step_{step_idx:02d}_x_prev", pred_x_prev, meta)
        sample = pred_x_prev

    tracer.clear_runtime_context()
    writer.save_sparse_tensor(
        step=5,
        role="output",
        logical_name="slat_final_raw_slat",
        tensor=sample,
        metadata={"source_function": "sample_slat"},
    )
    return sample


def normalize_slat(
    writer: ArtifactWriter,
    pipeline: TrellisTextTo3DPipeline,
    raw_slat: SparseTensor,
) -> SparseTensor:
    std = torch.tensor(pipeline.slat_normalization["std"], device=raw_slat.device)[None]
    mean = torch.tensor(pipeline.slat_normalization["mean"], device=raw_slat.device)[None]
    normalized = raw_slat * std + mean

    writer.save_sparse_tensor(
        step=7,
        role="input",
        logical_name="raw_slat",
        tensor=raw_slat,
        metadata={"source_function": "slat normalization"},
    )
    writer.save_tensor(
        step=7,
        role="input",
        logical_name="slat_normalization_std",
        tensor=std,
        metadata={"source_function": "slat_normalization"},
    )
    writer.save_tensor(
        step=7,
        role="input",
        logical_name="slat_normalization_mean",
        tensor=mean,
        metadata={"source_function": "slat_normalization"},
    )
    writer.save_sparse_tensor(
        step=7,
        role="output",
        logical_name="normalized_slat",
        tensor=normalized,
        metadata={"source_function": "slat normalization"},
    )
    return normalized


def save_text_conditioning(
    writer: ArtifactWriter,
    pipeline: TrellisTextTo3DPipeline,
    prompt: str,
    cond: Dict[str, torch.Tensor],
    token_meta: Dict[str, Any],
) -> None:
    writer.save_json_payload(
        step=1,
        role="input",
        logical_name="prompt_and_tokens",
        payload=token_meta,
        metadata={"source_function": "tokenizer"},
    )
    writer.save_tensor(
        step=1,
        role="output",
        logical_name="text_cond_embedding",
        tensor=cond["cond"],
        metadata={"source_function": "encode_text"},
    )
    writer.save_tensor(
        step=1,
        role="output",
        logical_name="text_neg_cond_embedding",
        tensor=cond["neg_cond"],
        metadata={"source_function": "encode_text([''])"},
    )
    token_norms = cond["cond"][0].detach().float().norm(dim=-1)
    writer.save_tensor(
        step=1,
        role="output",
        logical_name="text_token_norms",
        tensor=token_norms,
        metadata={"source_function": "cond.norm(dim=-1)"},
    )
    writer.save_json_payload(
        step=1,
        role="output",
        logical_name="prompt_summary",
        payload={
            "prompt": prompt,
            "cond_stats": tensor_stats(cond["cond"]),
            "neg_cond_stats": tensor_stats(cond["neg_cond"]),
        },
        metadata={"source_function": "get_cond"},
    )


def save_decoded_outputs(
    writer: ArtifactWriter,
    outputs: Dict[str, List[Any]],
    resolution: int,
    num_frames: int,
    export_glb: bool,
) -> None:
    from trellis.utils import postprocessing_utils, render_utils

    sample_dir = ensure_dir(writer.step_dirs[8] / "decoded_sample_0")
    summary: Dict[str, Any] = {}

    if "mesh" in outputs:
        mesh = outputs["mesh"][0]
        mesh_path = sample_dir / "mesh.ply"
        save_mesh(mesh_path, mesh)
        mesh_video = render_utils.render_video(mesh, resolution=resolution, num_frames=num_frames)["normal"]
        mesh_video_path = sample_dir / "mesh.mp4"
        save_video(mesh_video_path, mesh_video)
        summary["mesh"] = {
            "mesh_path": str(mesh_path.relative_to(writer.root_dir)),
            "video_path": str(mesh_video_path.relative_to(writer.root_dir)),
            "num_vertices": int(mesh.vertices.shape[0]),
            "num_faces": int(mesh.faces.shape[0]),
            "success": bool(mesh.success),
        }

    if "gaussian" in outputs:
        gaussian = outputs["gaussian"][0]
        gaussian_path = sample_dir / "gaussian.ply"
        gaussian.save_ply(gaussian_path)
        gaussian_video = render_utils.render_video(gaussian, resolution=resolution, num_frames=num_frames)["color"]
        gaussian_video_path = sample_dir / "gaussian.mp4"
        save_video(gaussian_video_path, gaussian_video)
        summary["gaussian"] = {
            "ply_path": str(gaussian_path.relative_to(writer.root_dir)),
            "video_path": str(gaussian_video_path.relative_to(writer.root_dir)),
            "num_points": int(gaussian.get_xyz.shape[0]),
            "xyz_stats": tensor_stats(gaussian.get_xyz),
            "opacity_stats": tensor_stats(gaussian.get_opacity),
            "scaling_stats": tensor_stats(gaussian.get_scaling),
        }

    if "radiance_field" in outputs:
        radiance_field = outputs["radiance_field"][0]
        rf_video = render_utils.render_video(radiance_field, resolution=resolution, num_frames=num_frames)["color"]
        rf_video_path = sample_dir / "radiance_field.mp4"
        save_video(rf_video_path, rf_video)
        summary["radiance_field"] = {
            "video_path": str(rf_video_path.relative_to(writer.root_dir)),
        }

    if export_glb and "gaussian" in outputs and "mesh" in outputs:
        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            simplify=0.95,
            texture_size=1024,
        )
        glb_path = sample_dir / "sample.glb"
        glb.export(glb_path)
        summary["glb"] = {"path": str(glb_path.relative_to(writer.root_dir))}

    writer.save_json_payload(
        step=8,
        role="output",
        logical_name="decoded_outputs_summary",
        payload=summary,
        metadata={"source_function": "decode_slat"},
    )


def build_step_table() -> List[Dict[str, Any]]:
    table = []
    for step, spec in STEP_SPECS.items():
        table.append(
            {
                "step": step,
                "paper_step": spec["paper_step"],
                "code_entry": spec["code_entry"],
                "inputs": spec["inputs"],
                "outputs": spec["outputs"],
                "intermediates": spec["intermediates"],
            }
        )
    return table


@torch.no_grad()
def main() -> None:
    args = parse_args()
    case_name = args.case_name.strip() or make_case_name(args.prompt)
    root_dir = ensure_dir(Path(args.output_dir) / case_name)
    writer = ArtifactWriter(root_dir=root_dir, step_specs=STEP_SPECS)

    print(f"[INFO] Output directory: {root_dir}")
    print(f"[INFO] Model:  {args.model}")
    print(f"[INFO] Prompt: {args.prompt}")

    ensure_open3d_stub()
    from trellis.pipelines import TrellisTextTo3DPipeline

    formats = [item.strip() for item in args.formats.split(",") if item.strip()]

    save_json(root_dir / "trace_step_table.json", {"steps": build_step_table()})

    tracer: Optional[RepresentativeBlockTracer] = None
    try:
        pipeline = TrellisTextTo3DPipeline.from_pretrained(args.model)
        pipeline.cuda()

        tokenizer = pipeline.text_cond_model["tokenizer"]
        token_meta = build_token_metadata(tokenizer, args.prompt)
        cond = pipeline.get_cond([args.prompt])

        ss_sampler_params = merge_sampler_params(
            pipeline.sparse_structure_sampler_params,
            steps_override=args.ss_steps,
            cfg_override=args.ss_cfg_strength,
        )
        slat_sampler_params = merge_sampler_params(
            pipeline.slat_sampler_params,
            steps_override=args.slat_steps,
            cfg_override=args.slat_cfg_strength,
        )
        ss_sampler_params["_num_samples"] = args.num_samples

        ss_model = pipeline.models["sparse_structure_flow_model"]
        slat_model = pipeline.models["slat_flow_model"]

        ss_steps = int(ss_sampler_params.get("steps", 50))
        slat_steps = int(slat_sampler_params.get("steps", 50))
        ss_block_idx = resolve_example_index(args.ss_example_block, len(ss_model.blocks))
        slat_block_idx = resolve_example_index(args.slat_example_block, len(slat_model.blocks))
        ss_step_idx = resolve_example_index(args.ss_example_step, ss_steps)
        slat_step_idx = resolve_example_index(args.slat_example_step, slat_steps)

        run_config = {
            "model": args.model,
            "prompt": args.prompt,
            "seed": args.seed,
            "num_samples": args.num_samples,
            "formats": formats,
            "skip_decode": args.skip_decode,
            "ss_sampler_params": ss_sampler_params,
            "slat_sampler_params": slat_sampler_params,
            "representative_examples": {
                "sparse_structure": {
                    "block_index": ss_block_idx,
                    "sampling_step": ss_step_idx,
                },
                "slat": {
                    "block_index": slat_block_idx,
                    "sampling_step": slat_step_idx,
                },
            },
        }
        save_json(root_dir / "run_config.json", run_config)

        save_text_conditioning(writer, pipeline, args.prompt, cond, token_meta)

        tracer = RepresentativeBlockTracer(
            writer=writer,
            token_meta=token_meta,
            topk_attn_tokens=args.topk_attn_tokens,
            query_chunk=args.query_chunk,
        )
        tracer.prepare_stage(
            stage_name="sparse_structure",
            model=ss_model,
            cond=cond["cond"],
            neg_cond=cond["neg_cond"],
            selected_step=ss_step_idx,
            selected_block=ss_block_idx,
            step_id=4,
            resolution=ss_model.resolution,
            patch_size=ss_model.patch_size,
        )
        tracer.prepare_stage(
            stage_name="slat",
            model=slat_model,
            cond=cond["cond"],
            neg_cond=cond["neg_cond"],
            selected_step=slat_step_idx,
            selected_block=slat_block_idx,
            step_id=6,
        )

        torch.manual_seed(args.seed)
        dense_noise = torch.randn(
            args.num_samples,
            ss_model.in_channels,
            ss_model.resolution,
            ss_model.resolution,
            ss_model.resolution,
            device=pipeline.device,
        )
        writer.save_tensor(
            step=2,
            role="input",
            logical_name="sparse_structure_initial_noise",
            tensor=dense_noise,
            metadata={"source_function": "torch.randn"},
        )

        params_for_model = {
            k: v
            for k, v in ss_sampler_params.items()
            if k not in {"steps", "rescale_t", "verbose", "_num_samples"}
        }
        steps = int(ss_sampler_params.get("steps", 50))
        rescale_t = float(ss_sampler_params.get("rescale_t", 1.0))
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        sample_dense = dense_noise
        for step_idx, (t, t_prev) in enumerate((t_seq[i], t_seq[i + 1]) for i in range(steps)):
            tracer.set_runtime_context("sparse_structure", step_idx, float(t))
            pred_x0, pred_eps, pred_v = pipeline.sparse_structure_sampler._get_model_prediction(
                ss_model,
                sample_dense,
                float(t),
                cond=cond["cond"],
                neg_cond=cond["neg_cond"],
                **params_for_model,
            )
            pred_x_prev = sample_dense - (float(t) - float(t_prev)) * pred_v
            meta = {
                "sampler_step": step_idx,
                "t": float(t),
                "t_prev": float(t_prev),
                "source_function": "FlowEulerSampler._get_model_prediction",
            }
            writer.save_tensor(2, "output", f"sparse_structure_step_{step_idx:02d}_x_t", sample_dense, meta)
            writer.save_tensor(2, "output", f"sparse_structure_step_{step_idx:02d}_pred_v", pred_v, meta)
            writer.save_tensor(2, "output", f"sparse_structure_step_{step_idx:02d}_pred_eps", pred_eps, meta)
            writer.save_tensor(2, "output", f"sparse_structure_step_{step_idx:02d}_pred_x0", pred_x0, meta)
            writer.save_tensor(2, "output", f"sparse_structure_step_{step_idx:02d}_x_prev", pred_x_prev, meta)
            sample_dense = pred_x_prev
        tracer.clear_runtime_context()
        writer.save_tensor(
            step=2,
            role="output",
            logical_name="sparse_structure_final_latent_z_s",
            tensor=sample_dense,
            metadata={"source_function": "sample_sparse_structure"},
        )

        coords = decode_sparse_structure(writer, pipeline, sample_dense)
        if coords.numel() == 0:
            writer.save_json_payload(
                step=5,
                role="input",
                logical_name="slat_skipped_empty_sparse_structure",
                payload={
                    "reason": "Sparse-structure decode produced zero occupied voxels.",
                    "suggestion": "Increase --ss-steps, adjust --seed, or try a simpler prompt.",
                },
                metadata={"source_function": "empty occupied coords guard"},
            )
            raise RuntimeError(
                "Sparse-structure decode produced zero occupied voxels, so SLat tracing cannot continue. "
                "Increase --ss-steps, adjust --seed, or try a simpler prompt."
            )

        sparse_coords = coords
        sparse_noise = sp.SparseTensor(
            feats=torch.randn(sparse_coords.shape[0], slat_model.in_channels, device=pipeline.device),
            coords=sparse_coords,
        )
        writer.save_sparse_tensor(
            step=5,
            role="input",
            logical_name="slat_initial_noise",
            tensor=sparse_noise,
            metadata={"source_function": "torch.randn"},
        )
        writer.save_tensor(
            step=5,
            role="input",
            logical_name="slat_coords",
            tensor=sparse_coords,
            metadata={"source_function": "occupied coords"},
        )

        params_for_sparse_model = {
            k: v for k, v in slat_sampler_params.items() if k not in {"steps", "rescale_t", "verbose"}
        }
        sparse_steps = int(slat_sampler_params.get("steps", 50))
        sparse_rescale_t = float(slat_sampler_params.get("rescale_t", 1.0))
        sparse_t_seq = np.linspace(1, 0, sparse_steps + 1)
        sparse_t_seq = sparse_rescale_t * sparse_t_seq / (1 + (sparse_rescale_t - 1) * sparse_t_seq)
        sample_sparse = sparse_noise
        for step_idx, (t, t_prev) in enumerate((sparse_t_seq[i], sparse_t_seq[i + 1]) for i in range(sparse_steps)):
            tracer.set_runtime_context("slat", step_idx, float(t))
            pred_x0, pred_eps, pred_v = pipeline.slat_sampler._get_model_prediction(
                slat_model,
                sample_sparse,
                float(t),
                cond=cond["cond"],
                neg_cond=cond["neg_cond"],
                **params_for_sparse_model,
            )
            pred_x_prev = sample_sparse - (float(t) - float(t_prev)) * pred_v
            meta = {
                "sampler_step": step_idx,
                "t": float(t),
                "t_prev": float(t_prev),
                "source_function": "FlowEulerSampler._get_model_prediction",
            }
            writer.save_sparse_tensor(5, "output", f"slat_step_{step_idx:02d}_x_t", sample_sparse, meta)
            writer.save_sparse_tensor(5, "output", f"slat_step_{step_idx:02d}_pred_v", pred_v, meta)
            writer.save_sparse_tensor(5, "output", f"slat_step_{step_idx:02d}_pred_eps", pred_eps, meta)
            writer.save_sparse_tensor(5, "output", f"slat_step_{step_idx:02d}_pred_x0", pred_x0, meta)
            writer.save_sparse_tensor(5, "output", f"slat_step_{step_idx:02d}_x_prev", pred_x_prev, meta)
            sample_sparse = pred_x_prev
        tracer.clear_runtime_context()
        writer.save_sparse_tensor(
            step=5,
            role="output",
            logical_name="slat_final_raw_slat",
            tensor=sample_sparse,
            metadata={"source_function": "sample_slat"},
        )

        normalized_slat = normalize_slat(writer, pipeline, sample_sparse)

        if not args.skip_decode:
            outputs = pipeline.decode_slat(normalized_slat, formats=formats)
            save_decoded_outputs(
                writer=writer,
                outputs=outputs,
                resolution=args.render_resolution,
                num_frames=args.render_frames,
                export_glb=args.export_glb,
            )

    finally:
        if tracer is not None:
            tracer.restore()

    manifest_path = writer.write_manifest()
    print(f"[INFO] Trace manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
