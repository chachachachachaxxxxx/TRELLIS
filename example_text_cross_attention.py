#!/usr/bin/env python3
"""
在 TRELLIS 文本生成 3D 的流程中，追踪并可视化文本 token 与空间 token 的 cross-attention。

使用方法：
1. 先进入已经安装好 TRELLIS 及其依赖的 Python 环境。
2. 直接运行：
   python example_text_cross_attention.py --prompt "a red chair"
3. 如需额外关注某些词，可以指定：
   python example_text_cross_attention.py --prompt "a red chair" --focus-words chair,red
4. 运行完成后，到下面目录查看导出的热力图、注意力 map 和说明文件：
   output/<case_name>/cross_attention_trace/

常用参数：
- --model: 选择模型路径或 Hugging Face repo。
- --trace-stages: 选择追踪 sparse_structure、slat 或二者。
- --ss-steps / --slat-steps: 控制两个阶段的采样步数。
- --topk-focus-tokens: 自动选择并重点展示多少个 token。

该脚本不会修改 TRELLIS 原始源码，而是通过运行时 monkey patch 的方式捕获注意力摘要，
并将原始数据和可读的可视化结果保存到 output/<case_name>/cross_attention_trace/。
"""

import os
import sys


def _peek_arg(flag: str, default: str = "") -> str:
    """在 argparse 初始化前，先从命令行里取出某个参数值。"""
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
import json
import math
import re
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from trellis.modules.attention.full_attn import scaled_dot_product_attention
from trellis.modules.attention.modules import MultiHeadAttention
from trellis.modules.sparse.attention.full_attn import sparse_scaled_dot_product_attention
from trellis.modules.sparse.attention.modules import SparseMultiHeadAttention
from trellis.modules.sparse.basic import SparseTensor
from trellis.pipelines import TrellisTextTo3DPipeline


def ensure_dir(path: Path) -> Path:
    """确保目录存在，并返回该目录路径，便于链式调用。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, data: dict) -> None:
    """以 UTF-8 编码保存 JSON 文件，方便后续查看和调试。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def sanitize_name(text: str, max_len: int = 48) -> str:
    """把任意文本清洗成适合做文件名或目录名的短字符串。"""
    text = text.strip().lower()
    text = re.sub(r"</w>", "", text)
    text = re.sub(r"[^0-9a-zA-Z_\u4e00-\u9fff]+", "_", text)
    text = text.strip("_")
    if not text:
        text = "token"
    return text[:max_len]


def tensor_signature(tensor: torch.Tensor, rows: int = 3, cols: int = 8) -> np.ndarray:
    """截取张量前部的小块样本，作为区分 cond / neg_cond 的轻量签名。"""
    sample = tensor[:1, :rows, :cols].detach().float().cpu().numpy()
    return sample


def clean_token_label(token: str) -> str:
    """把 tokenizer 的原始 token 处理成更适合展示的标签。"""
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
    """把 BPE token 按词边界合并成单词级分组，便于做词级注意力统计。"""
    word_labels: List[str] = []
    word_token_indices: List[List[int]] = []
    current_label_parts: List[str] = []
    current_indices: List[int] = []

    def flush_current() -> None:
        """把当前正在累积的 token 片段提交成一个完整词。"""
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

    if word_labels:
        return word_labels, word_token_indices

    fallback_labels = []
    fallback_indices = []
    for idx in range(valid_token_count):
        if special_mask[idx]:
            continue
        label = display_tokens[idx]
        if re.search(r"[0-9A-Za-z\u4e00-\u9fff]", label):
            fallback_labels.append(label)
            fallback_indices.append([idx])
    return fallback_labels, fallback_indices


def build_subword_groups(
    display_tokens: List[str],
    special_mask: List[bool],
    valid_token_count: int,
) -> Tuple[List[str], List[List[int]], Dict[int, int]]:
    """为每个有效 token 建立一对一的 subword 分组及反向索引。"""
    subword_labels: List[str] = []
    subword_token_indices: List[List[int]] = []
    token_to_subword_group: Dict[int, int] = {}

    for idx in range(valid_token_count):
        if special_mask[idx]:
            continue
        token_to_subword_group[idx] = len(subword_labels)
        subword_labels.append(display_tokens[idx])
        subword_token_indices.append([idx])

    return subword_labels, subword_token_indices, token_to_subword_group


def build_token_metadata(tokenizer, prompt: str) -> dict:
    """对 prompt 做分词，并生成后续导出可视化所需的 token 元信息。"""
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
    special_mask = []
    for i, token in enumerate(raw_tokens):
        is_special = (
            token in getattr(tokenizer, "all_special_tokens", [])
            or token in {"<|startoftext|>", "<|endoftext|>"}
            or i >= valid_token_count
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


def select_key_indices(total: int, max_items: int = 3) -> List[int]:
    """从一个长度为 total 的序列里均匀挑出若干关键索引。"""
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


def build_content_token_index_info(
    special_mask: List[bool],
    valid_token_count: int,
) -> Tuple[List[int], List[int], Dict[int, int]]:
    """建立内容 token / 特殊 token 的索引及内容 token 的反向映射。"""
    content_token_indices = [
        idx for idx in range(valid_token_count)
        if not special_mask[idx]
    ]
    special_token_indices = [
        idx for idx in range(valid_token_count)
        if special_mask[idx]
    ]
    full_to_content_index = {
        token_idx: content_idx
        for content_idx, token_idx in enumerate(content_token_indices)
    }
    return content_token_indices, special_token_indices, full_to_content_index


def dense_query_coords(resolution: int, patch_size: int, batch_size: int) -> List[np.ndarray]:
    """根据 dense 网格分辨率恢复每个 query 对应的三维坐标。"""
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
    content_token_indices: List[int],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """分块计算 query 对文本 token 的平均注意力，并按需保留完整 attention map。"""
    q = q.detach()
    k = k.detach()
    num_queries = q.shape[0]
    num_tokens = k.shape[0]
    head_dim = q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)

    k_float = k.float()
    token_sum = np.zeros((num_tokens,), dtype=np.float64)
    token_sum_content_renorm = np.zeros((num_tokens,), dtype=np.float64)
    map_chunks: List[np.ndarray] = []
    content_token_index_tensor: Optional[torch.Tensor] = None
    if content_token_indices:
        content_token_index_tensor = torch.as_tensor(
            content_token_indices,
            device=q.device,
            dtype=torch.long,
        )

    for start in range(0, num_queries, query_chunk):
        end = min(start + query_chunk, num_queries)
        q_chunk = q[start:end].float()
        scores = torch.einsum("qhd,khd->qhk", q_chunk, k_float) * scale
        attn = torch.softmax(scores, dim=-1)
        attn_mean = attn.mean(dim=1)
        token_sum += attn_mean.sum(dim=0).cpu().numpy()

        if content_token_index_tensor is not None and content_token_index_tensor.numel() > 0:
            content_attn = attn_mean.index_select(dim=-1, index=content_token_index_tensor)
            content_denom = content_attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            content_attn_renorm = content_attn / content_denom
            token_sum_content_renorm[content_token_indices] += (
                content_attn_renorm.sum(dim=0).cpu().numpy()
            )

        if keep_map:
            map_chunks.append(attn_mean.cpu().to(torch.float16).numpy())

    attn_map = np.concatenate(map_chunks, axis=0) if keep_map else None
    return token_sum, token_sum_content_renorm, attn_map


def save_heatmap(
    matrix: np.ndarray,
    x_labels: List[str],
    y_labels: List[str],
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """把二维注意力统计保存成热力图。"""
    fig_w = max(12, 0.28 * len(x_labels))
    fig_h = max(4, 0.35 * len(y_labels))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=65, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_token_curves(
    step_token_mean: np.ndarray,
    token_indices: List[int],
    token_labels: List[str],
    path: Path,
    title: str,
) -> None:
    """绘制若干重点 token 在采样步骤上的注意力曲线。"""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(step_token_mean.shape[0])
    for token_idx in token_indices:
        ax.plot(x, step_token_mean[:, token_idx], marker="o", label=token_labels[token_idx])
    ax.set_title(title)
    ax.set_xlabel("Sampling Step")
    ax.set_ylabel("Mean Attention")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_spatial_panels(
    panels: List[Tuple[str, np.ndarray, np.ndarray]],
    path: Path,
    title: str,
) -> None:
    """把多个空间注意力分布并排绘制成 3D 面板图。"""
    if not panels:
        return
    fig = plt.figure(figsize=(6 * len(panels), 5.5))
    for idx, (panel_title, coords, values) in enumerate(panels, start=1):
        coords = np.asarray(coords)
        values = np.asarray(values).reshape(-1)
        if coords.shape[0] != values.shape[0]:
            raise ValueError(
                f"save_spatial_panels expected matching coords/values for '{panel_title}', "
                f"got {coords.shape[0]} and {values.shape[0]}"
            )
        ax = fig.add_subplot(1, len(panels), idx, projection="3d")
        vmax = float(np.percentile(values, 99.5)) if values.size > 0 else 1.0
        vmax = max(vmax, float(values.max()) if values.size > 0 else 1.0)
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=values,
            s=3,
            alpha=0.9,
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(panel_title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=28, azim=42)
        fig.colorbar(scatter, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_bar(values: np.ndarray, labels: List[str], path: Path, title: str) -> None:
    """将单个空间位置对应的文本注意力分布画成柱状图。"""
    values = np.asarray(values).reshape(-1)
    if values.shape[0] != len(labels):
        raise ValueError(
            f"save_bar expected len(values) == len(labels), got {values.shape[0]} and {len(labels)}"
        )
    fig_w = max(12, 0.32 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))
    ax.bar(np.arange(len(labels)), values, color="#3A7CA5")
    ax.set_title(title)
    ax.set_xlabel("Text Token")
    ax.set_ylabel("Attention")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def slice_token_values(values: np.ndarray, token_indices: List[int]) -> np.ndarray:
    """按指定 token 索引抽取最后一维，得到内容 token 子集。"""
    array = np.asarray(values)
    if not token_indices:
        return np.zeros(array.shape[:-1] + (0,), dtype=np.float32)
    return np.asarray(array[..., token_indices], dtype=np.float32)


def renormalize_token_values(
    values: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """对最后一维重新归一化；若该维和为 0，则保持为 0。"""
    array = np.asarray(values, dtype=np.float32)
    denom = array.sum(axis=-1, keepdims=True)
    safe = np.where(denom > eps, denom, 1.0)
    renorm = array / safe
    return np.where(denom > eps, renorm, 0.0).astype(np.float32)


def aggregate_token_values_by_groups(
    values: np.ndarray,
    token_groups: List[List[int]],
    reduce: str = "mean",
) -> np.ndarray:
    """按给定 token 分组把 token 级统计聚合成 subword 或 word 级统计。"""
    array = np.asarray(values)
    if array.ndim not in (1, 2):
        raise ValueError(
            f"aggregate_token_values_by_groups expects 1D or 2D input, got {array.ndim}D"
        )
    if reduce not in {"mean", "sum"}:
        raise ValueError(f"Unsupported reduce mode: {reduce}")
    if not token_groups:
        return np.zeros((array.shape[0], 0), dtype=np.float32) if array.ndim == 2 else np.zeros((0,), dtype=np.float32)

    aggregated = []
    for group in token_groups:
        if not group:
            continue
        group_values = array[..., group]
        reduced = group_values.mean(axis=-1) if reduce == "mean" else group_values.sum(axis=-1)
        aggregated.append(np.asarray(reduced, dtype=np.float32))

    if not aggregated:
        return np.zeros((array.shape[0], 0), dtype=np.float32) if array.ndim == 2 else np.zeros((0,), dtype=np.float32)
    if array.ndim == 1:
        return np.stack(aggregated, axis=0)
    return np.stack(aggregated, axis=1)


def resolve_focus_word_indices(
    word_labels: List[str],
    reference_scores: np.ndarray,
    focus_words: List[str],
    topk: int,
) -> List[int]:
    """解析用户指定的关注词；若未命中，则回退到得分最高的若干词。"""
    requested: List[int] = []
    lowered_labels = [label.lower() for label in word_labels]
    for word in focus_words:
        word_lower = word.lower().strip()
        if not word_lower:
            continue
        for idx, label in enumerate(lowered_labels):
            if word_lower in label and idx not in requested:
                requested.append(idx)

    if requested:
        return requested

    if not word_labels:
        return []

    ranked = np.argsort(np.asarray(reference_scores).reshape(-1))[::-1]
    return [int(idx) for idx in ranked[:topk]]


def compute_attention_vmax(values_list: List[np.ndarray], percentile: float = 99.5) -> float:
    """为多张注意力图计算共享的颜色上限，避免单图极值主导显示。"""
    flattened = []
    for values in values_list:
        array = np.asarray(values).reshape(-1)
        if array.size == 0:
            continue
        flattened.append(array)
    if not flattened:
        return 1.0
    merged = np.concatenate(flattened, axis=0)
    return max(float(np.percentile(merged, percentile)), 1e-6)


def plot_spatial_attention_subplot(
    ax,
    coords: np.ndarray,
    values: np.ndarray,
    title: str,
    vmax: float,
    point_size: float = 5.0,
) -> None:
    """在给定的 3D 子图上绘制一张空间注意力散点图。"""
    coords = np.asarray(coords)
    values = np.asarray(values).reshape(-1)
    if coords.shape[0] != values.shape[0]:
        raise ValueError(
            f"plot_spatial_attention_subplot expected matching coords/values, "
            f"got {coords.shape[0]} and {values.shape[0]}"
        )

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


def save_word_attention_overview(
    avg_coords: np.ndarray,
    avg_word_maps: np.ndarray,
    word_labels: List[str],
    focus_word_indices: List[int],
    step_panels: List[dict],
    path: Path,
    title: str,
    block_index: int,
) -> None:
    """导出总览图：上方展示平均词级注意力，下方展示重点词的分步变化。"""
    if avg_word_maps.size == 0 or not word_labels:
        return

    num_words = len(word_labels)
    num_focus_words = len(focus_word_indices)
    num_step_panels = len(step_panels)
    has_bottom = num_focus_words > 0 and num_step_panels > 0

    top_width = max(2.7 * num_words, 8.5)
    bottom_width = max(2.7 * max(num_step_panels, 1), 8.5)
    fig_width = max(top_width, bottom_width)
    fig_height = 4.6 + (2.8 * num_focus_words if has_bottom else 0.0)

    fig = plt.figure(figsize=(fig_width, fig_height))
    outer = fig.add_gridspec(
        2 if has_bottom else 1,
        1,
        height_ratios=[1.15, max(num_focus_words, 1)] if has_bottom else [1.0],
        hspace=0.2,
    )

    fig.suptitle(title, fontsize=16, y=0.985)
    fig.text(
        0.5,
        0.92 if has_bottom else 0.94,
        f"Average attention maps across all traced sampling steps (final cross-attn block {block_index})",
        ha="center",
        va="center",
        fontsize=12,
    )

    top_grid = outer[0].subgridspec(1, num_words, wspace=0.04)
    top_vmax = compute_attention_vmax([avg_word_maps[:, idx] for idx in range(num_words)])
    for idx, label in enumerate(word_labels):
        ax = fig.add_subplot(top_grid[0, idx], projection="3d")
        plot_spatial_attention_subplot(
            ax=ax,
            coords=avg_coords,
            values=avg_word_maps[:, idx],
            title=f"\"{label}\"",
            vmax=top_vmax,
            point_size=4.0,
        )

    if has_bottom:
        fig.text(
            0.5,
            0.54,
            "Attention maps for individual sampling steps",
            ha="center",
            va="center",
            fontsize=12,
        )
        bottom_grid = outer[1].subgridspec(num_focus_words, num_step_panels, wspace=0.04, hspace=0.1)
        for row_idx, word_idx in enumerate(focus_word_indices):
            row_maps = [panel["word_maps"][:, word_idx] for panel in step_panels]
            row_vmax = compute_attention_vmax(row_maps)
            for col_idx, panel in enumerate(step_panels):
                ax = fig.add_subplot(bottom_grid[row_idx, col_idx], projection="3d")
                title_text = f"step {panel['step_index']}\nt={panel['timestep']:.3g}"
                plot_spatial_attention_subplot(
                    ax=ax,
                    coords=panel["coords"],
                    values=panel["word_maps"][:, word_idx],
                    title=title_text,
                    vmax=row_vmax,
                    point_size=4.0,
                )
                if col_idx == 0:
                    ax.text2D(
                        -0.16,
                        0.5,
                        word_labels[word_idx],
                        transform=ax.transAxes,
                        rotation=90,
                        va="center",
                        ha="center",
                        fontsize=12,
                    )

    fig.subplots_adjust(
        left=0.03,
        right=0.99,
        bottom=0.03,
        top=0.89 if has_bottom else 0.91,
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def resolve_focus_token_indices(
    token_meta: dict,
    reference_scores: np.ndarray,
    focus_words: List[str],
    topk: int,
) -> List[int]:
    """解析需要重点查看的 token 索引；未指定时自动挑选高注意力 token。"""
    display_tokens = token_meta["display_tokens"]
    special_mask = token_meta["special_mask"]
    valid_token_count = token_meta["valid_token_count"]

    requested: List[int] = []
    lowered_tokens = [token.lower() for token in display_tokens[:valid_token_count]]
    for word in focus_words:
        word_lower = word.lower().strip()
        if not word_lower:
            continue
        for idx, token in enumerate(lowered_tokens):
            if word_lower in token and idx not in requested:
                requested.append(idx)

    if requested:
        return requested

    candidates = []
    for idx in range(valid_token_count):
        if special_mask[idx]:
            continue
        if re.search(r"[0-9A-Za-z\u4e00-\u9fff]", display_tokens[idx]):
            candidates.append(idx)

    if not candidates:
        candidates = [idx for idx in range(valid_token_count) if not special_mask[idx]]
    if not candidates:
        candidates = list(range(valid_token_count))

    ranked = sorted(candidates, key=lambda idx: float(reference_scores[idx]), reverse=True)
    return ranked[:topk]


class CrossAttentionTracer:
    """在运行时挂钩 cross-attention 模块，并导出聚合统计与可视化结果。"""

    def __init__(
        self,
        token_meta: dict,
        query_chunk: int,
        focus_words: List[str],
        topk_focus_tokens: int,
        topk_spatial: int,
    ):
        """初始化追踪器的全局配置与待恢复的 monkey patch 栈。"""
        self.token_meta = token_meta
        self.valid_token_count = int(token_meta["valid_token_count"])
        (
            self.content_token_indices,
            self.special_token_indices,
            self.full_to_content_index,
        ) = build_content_token_index_info(
            special_mask=token_meta["special_mask"],
            valid_token_count=self.valid_token_count,
        )
        self.query_chunk = query_chunk
        self.focus_words = focus_words
        self.topk_focus_tokens = topk_focus_tokens
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
        """为某个采样阶段建立状态，并在需要时把 hook 挂到模型上。"""
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
            "forward_counter": 0,
            "step_block_token_sum": np.zeros(
                (total_steps, num_blocks, self.token_meta["valid_token_count"]),
                dtype=np.float64,
            ),
            "step_block_token_sum_content_renorm": np.zeros(
                (total_steps, num_blocks, self.token_meta["valid_token_count"]),
                dtype=np.float64,
            ),
            "step_block_query_count": np.zeros((total_steps, num_blocks), dtype=np.int64),
            "selected_maps": {},
            "overview_block_index": num_blocks - 1,
            "overview_sum_map": None,
            "overview_sum_map_content_renorm": None,
            "overview_count": 0,
            "overview_coords": None,
        }
        if enabled:
            self._patch_model_forward(stage_name, model)
            self._patch_cross_modules(stage_name, model)

    def restore(self) -> None:
        """恢复所有被替换过的 forward，确保脚本结束后模型状态干净。"""
        while self._restore_stack:
            obj, attr, value = self._restore_stack.pop()
            setattr(obj, attr, value)

    def _infer_pass_kind(self, stage_name: str, cond: torch.Tensor) -> str:
        """根据 cond 的签名判断当前 forward 属于条件分支还是负条件分支。"""
        state = self.stage_states[stage_name]
        current_sig = tensor_signature(cond)
        dist_cond = float(np.max(np.abs(current_sig - state["cond_signature"])))
        dist_neg = float(np.max(np.abs(current_sig - state["neg_signature"])))
        return "cond" if dist_cond <= dist_neg else "neg"

    def _patch_model_forward(self, stage_name: str, model: torch.nn.Module) -> None:
        """包装 stage 级 forward，用来记录当前采样步和上下文信息。"""
        original_forward = model.forward
        self._restore_stack.append((model, "forward", original_forward))
        tracer = self

        def wrapped_forward(model_self, x, t, cond):
            """在原始 forward 前后维护当前 tracing 上下文。"""
            state = tracer.stage_states[stage_name]
            pass_kind = tracer._infer_pass_kind(stage_name, cond)
            if pass_kind == "cond":
                step_index = state["cond_step_counter"]
                state["cond_step_counter"] += 1
            else:
                step_index = max(state["cond_step_counter"] - 1, 0)
            step_index = min(step_index, state["total_steps"] - 1)
            state["forward_counter"] += 1
            tracer.current_forward_context = {
                "stage": stage_name,
                "pass_kind": pass_kind,
                "step_index": step_index,
                "timestep": float(t[0].detach().float().cpu().item()) if torch.is_tensor(t) else float(t),
                "x": x,
                "batch_size": x.shape[0] if not isinstance(x, SparseTensor) else x.shape[0],
            }
            try:
                return original_forward(x, t, cond)
            finally:
                tracer.current_forward_context = None

        model.forward = types.MethodType(wrapped_forward, model)

    def _patch_cross_modules(self, stage_name: str, model: torch.nn.Module) -> None:
        """遍历模型里的 cross-attention 模块，并按 dense / sparse 分别挂钩。"""
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
        """包装 dense cross-attention 模块，在前向时拦截 q/k 并记录统计。"""
        original_forward = module.forward
        self._restore_stack.append((module, "forward", original_forward))
        tracer = self

        def wrapped_forward(attn_self, x, context=None, indices=None):
            """复用原始 attention 计算路径，同时额外保存 dense 注意力摘要。"""
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
        """包装 sparse cross-attention 模块，在前向时拦截 q/k 并记录统计。"""
        original_forward = module.forward
        self._restore_stack.append((module, "forward", original_forward))
        tracer = self

        def wrapped_forward(attn_self, x, context=None):
            """复用原始 sparse attention 计算路径，同时额外保存稀疏注意力摘要。"""
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
        """判断当前 step / block 是否属于需要完整导出的采样子集。"""
        state = self.stage_states[stage_name]
        return (
            step_index in state["selected_steps"]
            and block_index in state["selected_blocks"]
        )

    def _record_dense(
        self,
        stage_name: str,
        module_name: str,
        block_index: int,
        q: torch.Tensor,
        k: torch.Tensor,
        batch_size: int,
    ) -> None:
        """记录 dense cross-attention 的 token 统计和可选的完整 map。"""
        ctx = self.current_forward_context
        if ctx is None or ctx["stage"] != stage_name or ctx["pass_kind"] != "cond":
            return

        state = self.stage_states[stage_name]
        step_index = ctx["step_index"]
        selected_map = self._should_keep_map(stage_name, step_index, block_index)
        overview_map = block_index == state["overview_block_index"]
        keep_map = selected_map or overview_map
        valid_token_count = self.token_meta["valid_token_count"]

        dense_coords = dense_query_coords(
            resolution=self._dense_resolution_from_state(stage_name),
            patch_size=self._dense_patch_size_from_state(stage_name),
            batch_size=batch_size,
        )

        for batch_idx in range(batch_size):
            token_sum, token_sum_content_renorm, attn_map = compute_headmean_attention(
                q=q[batch_idx],
                k=k[batch_idx],
                query_chunk=self.query_chunk,
                keep_map=keep_map and batch_idx == 0,
                content_token_indices=self.content_token_indices,
            )
            state["step_block_token_sum"][step_index, block_index] += token_sum[:valid_token_count]
            state["step_block_token_sum_content_renorm"][step_index, block_index] += (
                token_sum_content_renorm[:valid_token_count]
            )
            state["step_block_query_count"][step_index, block_index] += q.shape[1]

            if batch_idx == 0 and overview_map and attn_map is not None:
                self._update_overview_map(
                    state=state,
                    coords=dense_coords[batch_idx],
                    attn_map=attn_map[:, :valid_token_count],
                )

            if selected_map and batch_idx == 0 and attn_map is not None:
                state["selected_maps"][(step_index, block_index)] = {
                    "module_name": module_name,
                    "step_index": step_index,
                    "block_index": block_index,
                    "timestep": ctx["timestep"],
                    "coords": dense_coords[batch_idx],
                    "attn_map": attn_map[:, :valid_token_count],
                }

    def _record_sparse(
        self,
        stage_name: str,
        module_name: str,
        block_index: int,
        q: SparseTensor,
        k: torch.Tensor,
    ) -> None:
        """记录 sparse cross-attention 的 token 统计和可选的完整 map。"""
        ctx = self.current_forward_context
        if ctx is None or ctx["stage"] != stage_name or ctx["pass_kind"] != "cond":
            return

        state = self.stage_states[stage_name]
        step_index = ctx["step_index"]
        selected_map = self._should_keep_map(stage_name, step_index, block_index)
        overview_map = block_index == state["overview_block_index"]
        keep_map = selected_map or overview_map
        valid_token_count = self.token_meta["valid_token_count"]

        for batch_idx in range(q.shape[0]):
            q_slice = q.feats[q.layout[batch_idx]]
            token_sum, token_sum_content_renorm, attn_map = compute_headmean_attention(
                q=q_slice,
                k=k[batch_idx],
                query_chunk=self.query_chunk,
                keep_map=keep_map and batch_idx == 0,
                content_token_indices=self.content_token_indices,
            )
            state["step_block_token_sum"][step_index, block_index] += token_sum[:valid_token_count]
            state["step_block_token_sum_content_renorm"][step_index, block_index] += (
                token_sum_content_renorm[:valid_token_count]
            )
            state["step_block_query_count"][step_index, block_index] += q_slice.shape[0]

            if batch_idx == 0 and overview_map and attn_map is not None:
                self._update_overview_map(
                    state=state,
                    coords=q.coords[q.layout[batch_idx], 1:].detach().cpu().numpy().astype(np.int16),
                    attn_map=attn_map[:, :valid_token_count],
                )

            if selected_map and batch_idx == 0 and attn_map is not None:
                coords = q.coords[q.layout[batch_idx], 1:].detach().cpu().numpy().astype(np.int16)
                state["selected_maps"][(step_index, block_index)] = {
                    "module_name": module_name,
                    "step_index": step_index,
                    "block_index": block_index,
                    "timestep": ctx["timestep"],
                    "coords": coords,
                    "attn_map": attn_map[:, :valid_token_count],
                }

    def _update_overview_map(
        self,
        state: dict,
        coords: np.ndarray,
        attn_map: np.ndarray,
    ) -> None:
        """把某个 block 的 attention map 累加到总览统计里。"""
        content_attn_map = slice_token_values(attn_map, self.content_token_indices)
        content_attn_map_renorm = renormalize_token_values(content_attn_map)
        attn_map_content_renorm_full = np.zeros_like(attn_map, dtype=np.float32)
        if self.content_token_indices:
            attn_map_content_renorm_full[:, self.content_token_indices] = content_attn_map_renorm

        if state["overview_sum_map"] is None:
            state["overview_sum_map"] = attn_map.astype(np.float32, copy=True)
            state["overview_sum_map_content_renorm"] = attn_map_content_renorm_full
            state["overview_coords"] = np.asarray(coords, dtype=np.int16)
            state["overview_count"] = 1
            return
        if state["overview_sum_map"].shape != attn_map.shape:
            return
        state["overview_sum_map"] += attn_map.astype(np.float32, copy=False)
        if state["overview_sum_map_content_renorm"] is None:
            state["overview_sum_map_content_renorm"] = attn_map_content_renorm_full
        elif state["overview_sum_map_content_renorm"].shape == attn_map_content_renorm_full.shape:
            state["overview_sum_map_content_renorm"] += attn_map_content_renorm_full
        state["overview_count"] += 1

    def _dense_resolution_from_state(self, stage_name: str) -> int:
        """读取 dense 阶段当前使用的网格分辨率。"""
        state = self.stage_states[stage_name]
        return int(state["dense_resolution"])

    def _dense_patch_size_from_state(self, stage_name: str) -> int:
        """读取 dense 阶段当前使用的 patch 大小。"""
        state = self.stage_states[stage_name]
        return int(state["dense_patch_size"])

    def set_dense_grid_meta(self, stage_name: str, resolution: int, patch_size: int) -> None:
        """补充 dense query 坐标恢复所需的网格元信息。"""
        state = self.stage_states[stage_name]
        state["dense_resolution"] = int(resolution)
        state["dense_patch_size"] = int(patch_size)

    def export(self, root_dir: Path) -> None:
        """把所有阶段的聚合统计、原始 map 和可视化结果导出到磁盘。"""
        token_labels = self.token_meta["display_tokens"][: self.valid_token_count]
        special_mask = self.token_meta["special_mask"]
        valid_token_count = self.valid_token_count
        content_token_indices = list(self.content_token_indices)
        special_token_indices = list(self.special_token_indices)
        content_token_labels = [token_labels[idx] for idx in content_token_indices]
        special_token_labels = [token_labels[idx] for idx in special_token_indices]
        word_labels = list(self.token_meta.get("word_labels", []))
        word_token_indices = [list(group) for group in self.token_meta.get("word_token_indices", [])]
        subword_labels, subword_token_indices, token_to_subword_group = build_subword_groups(
            display_tokens=self.token_meta["display_tokens"],
            special_mask=special_mask,
            valid_token_count=valid_token_count,
        )
        manifest = {
            "prompt": self.token_meta["prompt"],
            "token_labels": token_labels,
            "content_token_indices": content_token_indices,
            "content_token_labels": content_token_labels,
            "special_token_indices": special_token_indices,
            "special_token_labels": special_token_labels,
            "attention_export_modes": {
                "content_raw": "Normalize over all tokens first, then drop special tokens for display/export.",
                "content_renorm": "Drop special tokens first, then renormalize remaining content-token attention to sum to 1.",
            },
            "subword_labels": subword_labels,
            "word_labels": word_labels,
            "stages": {},
        }

        focus_token_indices: Optional[List[int]] = None
        focus_subword_indices: Optional[List[int]] = None
        focus_word_indices: Optional[List[int]] = None
        preferred_stage = "slat" if "slat" in self.stage_states and self.stage_states["slat"]["enabled"] else None
        if preferred_stage is None:
            for stage_name, state in self.stage_states.items():
                if state["enabled"]:
                    preferred_stage = stage_name
                    break

        if preferred_stage is not None:
            ref_matrix = self._normalized_step_block_token_mean(self.stage_states[preferred_stage])
            ref_scores = ref_matrix[-1].mean(axis=0)
            focus_token_indices = resolve_focus_token_indices(
                self.token_meta,
                ref_scores,
                self.focus_words,
                self.topk_focus_tokens,
            )
            focus_subword_indices = []
            for token_idx in focus_token_indices:
                subword_idx = token_to_subword_group.get(token_idx)
                if subword_idx is not None and subword_idx not in focus_subword_indices:
                    focus_subword_indices.append(subword_idx)
            ref_word_scores = aggregate_token_values_by_groups(ref_scores, word_token_indices, reduce="mean")
            focus_word_indices = resolve_focus_word_indices(
                word_labels,
                ref_word_scores,
                self.focus_words,
                self.topk_focus_tokens,
            )
        else:
            focus_token_indices = []
            focus_subword_indices = []
            focus_word_indices = []

        for stage_name, state in self.stage_states.items():
            if not state["enabled"]:
                continue

            stage_dir = ensure_dir(root_dir / stage_name)
            normalized = self._normalized_step_block_token_mean(state)
            normalized_content_renorm = self._normalized_step_block_token_mean(state, renorm_content=True)
            step_token_mean = normalized.mean(axis=1)
            final_step_block_mean = normalized[-1]
            step_token_mean_content_raw = slice_token_values(step_token_mean, content_token_indices)
            final_step_block_mean_content_raw = slice_token_values(final_step_block_mean, content_token_indices)
            step_token_mean_content_renorm = slice_token_values(
                normalized_content_renorm.mean(axis=1),
                content_token_indices,
            )
            final_step_block_mean_content_renorm = slice_token_values(
                normalized_content_renorm[-1],
                content_token_indices,
            )

            save_json(
                stage_dir / "summary.json",
                {
                    "stage": stage_name,
                    "query_type": state["query_type"],
                    "total_steps": state["total_steps"],
                    "num_blocks": state["num_blocks"],
                    "selected_steps": state["selected_steps"],
                    "selected_blocks": state["selected_blocks"],
                    "content_token_indices": content_token_indices,
                    "content_token_labels": content_token_labels,
                    "special_token_indices": special_token_indices,
                    "special_token_labels": special_token_labels,
                    "attention_export_modes": {
                        "content_raw": "Normalize over all tokens first, then drop special tokens for display/export.",
                        "content_renorm": "Drop special tokens first, then renormalize remaining content-token attention to sum to 1.",
                    },
                    "focus_token_indices": focus_token_indices,
                    "focus_token_labels": [token_labels[i] for i in focus_token_indices],
                    "subword_labels": subword_labels,
                    "subword_token_indices": subword_token_indices,
                    "focus_subword_indices": focus_subword_indices,
                    "focus_subword_labels": [subword_labels[i] for i in focus_subword_indices],
                    "word_labels": word_labels,
                    "word_token_indices": word_token_indices,
                    "focus_word_indices": focus_word_indices,
                    "focus_word_labels": [word_labels[i] for i in focus_word_indices],
                },
            )
            np.save(stage_dir / "step_token_mean.npy", step_token_mean.astype(np.float32))
            np.save(stage_dir / "final_step_block_token_mean.npy", final_step_block_mean.astype(np.float32))
            np.save(stage_dir / "step_token_mean_content_raw.npy", step_token_mean_content_raw.astype(np.float32))
            np.save(
                stage_dir / "final_step_block_token_mean_content_raw.npy",
                final_step_block_mean_content_raw.astype(np.float32),
            )
            np.save(
                stage_dir / "step_token_mean_content_renorm.npy",
                step_token_mean_content_renorm.astype(np.float32),
            )
            np.save(
                stage_dir / "final_step_block_token_mean_content_renorm.npy",
                final_step_block_mean_content_renorm.astype(np.float32),
            )

            save_heatmap(
                step_token_mean,
                token_labels,
                [f"step_{i:02d}" for i in range(step_token_mean.shape[0])],
                stage_dir / "step_token_heatmap.png",
                title=f"{stage_name}: mean token attention over sampling steps (all valid tokens)",
                xlabel="Text Token",
                ylabel="Sampling Step",
            )
            save_heatmap(
                final_step_block_mean,
                token_labels,
                [f"block_{i:02d}" for i in range(final_step_block_mean.shape[0])],
                stage_dir / "final_step_block_token_heatmap.png",
                title=f"{stage_name}: final-step token attention over blocks (all valid tokens)",
                xlabel="Text Token",
                ylabel="Transformer Block",
            )
            save_heatmap(
                step_token_mean_content_raw,
                content_token_labels,
                [f"step_{i:02d}" for i in range(step_token_mean_content_raw.shape[0])],
                stage_dir / "step_token_heatmap_content_raw.png",
                title=f"{stage_name}: content-token attention over sampling steps (drop special tokens only)",
                xlabel="Content Token",
                ylabel="Sampling Step",
            )
            save_heatmap(
                final_step_block_mean_content_raw,
                content_token_labels,
                [f"block_{i:02d}" for i in range(final_step_block_mean_content_raw.shape[0])],
                stage_dir / "final_step_block_token_heatmap_content_raw.png",
                title=f"{stage_name}: final-step content-token attention over blocks (drop special tokens only)",
                xlabel="Content Token",
                ylabel="Transformer Block",
            )
            save_heatmap(
                step_token_mean_content_renorm,
                content_token_labels,
                [f"step_{i:02d}" for i in range(step_token_mean_content_renorm.shape[0])],
                stage_dir / "step_token_heatmap_content_renorm.png",
                title=f"{stage_name}: content-token attention over sampling steps (renorm after dropping special tokens)",
                xlabel="Content Token",
                ylabel="Sampling Step",
            )
            save_heatmap(
                final_step_block_mean_content_renorm,
                content_token_labels,
                [f"block_{i:02d}" for i in range(final_step_block_mean_content_renorm.shape[0])],
                stage_dir / "final_step_block_token_heatmap_content_renorm.png",
                title=f"{stage_name}: final-step content-token attention over blocks (renorm after dropping special tokens)",
                xlabel="Content Token",
                ylabel="Transformer Block",
            )

            if focus_token_indices:
                save_token_curves(
                    step_token_mean,
                    focus_token_indices,
                    token_labels,
                    stage_dir / "focus_token_step_curves.png",
                    title=f"{stage_name}: focus-token attention across steps",
                )
                focus_content_token_indices = [
                    self.full_to_content_index[token_idx]
                    for token_idx in focus_token_indices
                    if token_idx in self.full_to_content_index
                ]
                save_token_curves(
                    step_token_mean_content_raw,
                    focus_content_token_indices,
                    content_token_labels,
                    stage_dir / "focus_token_step_curves_content_raw.png",
                    title=f"{stage_name}: focus-token attention across steps (drop special tokens only)",
                )
                save_token_curves(
                    step_token_mean_content_renorm,
                    focus_content_token_indices,
                    content_token_labels,
                    stage_dir / "focus_token_step_curves_content_renorm.png",
                    title=f"{stage_name}: focus-token attention across steps (renorm after dropping special tokens)",
                )

            selected_map_meta = []
            for (step_index, block_index), item in sorted(state["selected_maps"].items()):
                raw_dir = ensure_dir(stage_dir / "raw_maps")
                attn_map = item["attn_map"][:, : self.token_meta["valid_token_count"]]
                content_attn_map_raw = slice_token_values(attn_map, content_token_indices)
                content_attn_map_renorm = renormalize_token_values(content_attn_map_raw)
                coords = item["coords"]
                base = f"step_{step_index:02d}_block_{block_index:02d}"
                np.savez_compressed(
                    raw_dir / f"{base}.npz",
                    attn_map=attn_map.astype(np.float16),
                    content_attn_map_raw=content_attn_map_raw.astype(np.float16),
                    content_attn_map_renorm=content_attn_map_renorm.astype(np.float16),
                    coords=coords.astype(np.int16),
                )
                save_json(
                    raw_dir / f"{base}.json",
                    {
                        "stage": stage_name,
                        "module_name": item["module_name"],
                        "step_index": step_index,
                        "block_index": block_index,
                        "timestep": item["timestep"],
                        "num_queries": int(attn_map.shape[0]),
                        "num_tokens": int(attn_map.shape[1]),
                        "content_token_indices": content_token_indices,
                        "content_token_labels": content_token_labels,
                        "arrays": {
                            "attn_map": "Raw attention over all valid tokens, including special tokens.",
                            "content_attn_map_raw": "Attention over content tokens after dropping special tokens only.",
                            "content_attn_map_renorm": "Attention over content tokens after dropping special tokens and renormalizing per query.",
                        },
                    },
                )
                selected_map_meta.append(
                    {
                        "key": [step_index, block_index],
                        "module_name": item["module_name"],
                        "npz": str((raw_dir / f"{base}.npz").relative_to(root_dir)),
                        "json": str((raw_dir / f"{base}.json").relative_to(root_dir)),
                    }
                )

            self._export_focus_views(
                stage_dir=stage_dir,
                state=state,
                token_labels=token_labels,
                focus_token_indices=focus_token_indices,
            )

            subword_overview_paths = self._export_group_overview(
                stage_dir=stage_dir,
                state=state,
                group_labels=subword_labels,
                group_token_indices=subword_token_indices,
                focus_group_indices=focus_subword_indices,
                file_prefix="subword_attention_content_raw",
                title_label="subword-to-3D",
                aggregation_name="bpe_subword",
                renorm_content=False,
            )
            subword_renorm_overview_paths = self._export_group_overview(
                stage_dir=stage_dir,
                state=state,
                group_labels=subword_labels,
                group_token_indices=subword_token_indices,
                focus_group_indices=focus_subword_indices,
                file_prefix="subword_attention_content_renorm",
                title_label="subword-to-3D",
                aggregation_name="bpe_subword",
                renorm_content=True,
            )
            overview_paths = self._export_word_overview(
                stage_dir=stage_dir,
                state=state,
                word_labels=word_labels,
                word_token_indices=word_token_indices,
                focus_word_indices=focus_word_indices,
                renorm_content=False,
            )
            overview_renorm_paths = self._export_word_overview(
                stage_dir=stage_dir,
                state=state,
                word_labels=word_labels,
                word_token_indices=word_token_indices,
                focus_word_indices=focus_word_indices,
                renorm_content=True,
            )

            manifest["stages"][stage_name] = {
                "summary": str((stage_dir / "summary.json").relative_to(root_dir)),
                "step_token_mean": str((stage_dir / "step_token_mean.npy").relative_to(root_dir)),
                "final_step_block_token_mean": str((stage_dir / "final_step_block_token_mean.npy").relative_to(root_dir)),
                "step_token_mean_content_raw": str((stage_dir / "step_token_mean_content_raw.npy").relative_to(root_dir)),
                "final_step_block_token_mean_content_raw": str(
                    (stage_dir / "final_step_block_token_mean_content_raw.npy").relative_to(root_dir)
                ),
                "step_token_mean_content_renorm": str(
                    (stage_dir / "step_token_mean_content_renorm.npy").relative_to(root_dir)
                ),
                "final_step_block_token_mean_content_renorm": str(
                    (stage_dir / "final_step_block_token_mean_content_renorm.npy").relative_to(root_dir)
                ),
                "selected_maps": selected_map_meta,
                **subword_overview_paths,
                **subword_renorm_overview_paths,
                **overview_paths,
                **overview_renorm_paths,
            }

        save_json(root_dir / "trace_manifest.json", manifest)

    def _normalized_step_block_token_mean(self, state: dict, renorm_content: bool = False) -> np.ndarray:
        """用 query 数量对累计注意力求平均，得到可比较的 step/block 统计。"""
        counts = state["step_block_query_count"].astype(np.float64)
        counts = np.maximum(counts[..., None], 1.0)
        source = (
            state["step_block_token_sum_content_renorm"]
            if renorm_content
            else state["step_block_token_sum"]
        )
        return source / counts

    def _export_focus_views(
        self,
        stage_dir: Path,
        state: dict,
        token_labels: List[str],
        focus_token_indices: List[int],
    ) -> None:
        """为重点 token 导出分步、分块以及 top 空间位置的细粒度视图。"""
        if not focus_token_indices:
            return

        content_token_labels = [
            token_labels[token_idx]
            for token_idx in self.content_token_indices
        ]
        last_step = state["total_steps"] - 1
        last_block = state["num_blocks"] - 1
        selected_steps = state["selected_steps"]
        selected_blocks = state["selected_blocks"]

        final_map = state["selected_maps"].get((last_step, last_block))
        if final_map is None:
            return

        for token_idx in focus_token_indices:
            token_label = token_labels[token_idx]
            token_slug = f"token_{token_idx:02d}_{sanitize_name(token_label)}"
            focus_dir = ensure_dir(stage_dir / token_slug)

            step_panels = []
            for step_index in selected_steps:
                item = state["selected_maps"].get((step_index, last_block))
                if item is None:
                    continue
                step_panels.append(
                    (
                        f"step {step_index}, block {last_block}",
                        item["coords"],
                        item["attn_map"][:, token_idx].astype(np.float32),
                    )
                )
            save_spatial_panels(
                step_panels,
                focus_dir / "step_progress.png",
                title=f"{stage_dir.name}: spatial attention for token '{token_label}' across steps",
            )

            block_panels = []
            for block_index in selected_blocks:
                item = state["selected_maps"].get((last_step, block_index))
                if item is None:
                    continue
                block_panels.append(
                    (
                        f"step {last_step}, block {block_index}",
                        item["coords"],
                        item["attn_map"][:, token_idx].astype(np.float32),
                    )
                )
            save_spatial_panels(
                block_panels,
                focus_dir / "block_progress.png",
                title=f"{stage_dir.name}: spatial attention for token '{token_label}' across blocks",
            )

            focus_scores = final_map["attn_map"][:, token_idx].astype(np.float32)
            top_indices = np.argsort(focus_scores)[::-1][: self.topk_spatial]
            top_items = []
            for rank, query_idx in enumerate(top_indices, start=1):
                coord = final_map["coords"][query_idx].tolist()
                text_dist = final_map["attn_map"][query_idx, :].astype(np.float32)
                content_text_dist_raw = slice_token_values(text_dist, self.content_token_indices)
                content_text_dist_renorm = renormalize_token_values(content_text_dist_raw)
                save_bar(
                    text_dist,
                    token_labels,
                    focus_dir / f"top_spatial_rank_{rank:02d}_text_distribution.png",
                    title=(
                        f"{stage_dir.name}: spatial token #{rank} at {coord} "
                        f"(focus token '{token_label}', attn={focus_scores[query_idx]:.4f})"
                    ),
                )
                save_bar(
                    content_text_dist_raw,
                    content_token_labels,
                    focus_dir / f"top_spatial_rank_{rank:02d}_text_distribution_content_raw.png",
                    title=(
                        f"{stage_dir.name}: spatial token #{rank} at {coord} "
                        f"(drop special only, focus token '{token_label}')"
                    ),
                )
                save_bar(
                    content_text_dist_renorm,
                    content_token_labels,
                    focus_dir / f"top_spatial_rank_{rank:02d}_text_distribution_content_renorm.png",
                    title=(
                        f"{stage_dir.name}: spatial token #{rank} at {coord} "
                        f"(content renorm, focus token '{token_label}')"
                    ),
                )
                top_items.append(
                    {
                        "rank": rank,
                        "query_index": int(query_idx),
                        "coord": coord,
                        "focus_token_attention": float(focus_scores[query_idx]),
                        "bar_path": f"{token_slug}/top_spatial_rank_{rank:02d}_text_distribution.png",
                        "bar_path_content_raw": (
                            f"{token_slug}/top_spatial_rank_{rank:02d}_text_distribution_content_raw.png"
                        ),
                        "bar_path_content_renorm": (
                            f"{token_slug}/top_spatial_rank_{rank:02d}_text_distribution_content_renorm.png"
                        ),
                    }
                )

            save_json(
                focus_dir / "top_spatial_tokens.json",
                {
                    "focus_token_index": token_idx,
                    "focus_token_label": token_label,
                    "final_step": last_step,
                    "final_block": last_block,
                    "items": top_items,
                },
            )

    def _export_word_overview(
        self,
        stage_dir: Path,
        state: dict,
        word_labels: List[str],
        word_token_indices: List[List[int]],
        focus_word_indices: List[int],
        renorm_content: bool,
    ) -> dict:
        """以单词分组为单位导出总览图和元数据。"""
        return self._export_group_overview(
            stage_dir=stage_dir,
            state=state,
            group_labels=word_labels,
            group_token_indices=word_token_indices,
            focus_group_indices=focus_word_indices,
            file_prefix=f"word_attention_{'content_renorm' if renorm_content else 'content_raw'}",
            title_label="word-to-3D",
            aggregation_name="word",
            renorm_content=renorm_content,
        )

    def _export_group_overview(
        self,
        stage_dir: Path,
        state: dict,
        group_labels: List[str],
        group_token_indices: List[List[int]],
        focus_group_indices: List[int],
        file_prefix: str,
        title_label: str,
        aggregation_name: str,
        renorm_content: bool,
    ) -> dict:
        """复用同一套逻辑导出 subword / word 两类分组总览结果。"""
        if not group_labels or not group_token_indices:
            return {}
        overview_sum_key = "overview_sum_map_content_renorm" if renorm_content else "overview_sum_map"
        if state[overview_sum_key] is None or state["overview_count"] <= 0 or state["overview_coords"] is None:
            return {}

        avg_map = state[overview_sum_key] / float(max(state["overview_count"], 1))
        avg_group_maps = aggregate_token_values_by_groups(avg_map, group_token_indices, reduce="mean")

        step_panels = []
        for step_index in state["selected_steps"]:
            item = state["selected_maps"].get((step_index, state["overview_block_index"]))
            if item is None:
                continue
            item_attn_map = item["attn_map"][:, : self.valid_token_count]
            if renorm_content:
                item_attn_map_content = slice_token_values(item_attn_map, self.content_token_indices)
                item_attn_map_content_renorm = renormalize_token_values(item_attn_map_content)
                item_attn_map = np.zeros_like(item_attn_map, dtype=np.float32)
                if self.content_token_indices:
                    item_attn_map[:, self.content_token_indices] = item_attn_map_content_renorm
            step_panels.append(
                {
                    "step_index": int(step_index),
                    "timestep": float(item["timestep"]),
                    "coords": item["coords"],
                    "word_maps": aggregate_token_values_by_groups(
                        item_attn_map,
                        group_token_indices,
                        reduce="mean",
                    ),
                }
            )

        overview_image = stage_dir / f"{file_prefix}_overview.png"
        overview_npz = stage_dir / f"{file_prefix}_average_maps.npz"
        overview_json = stage_dir / f"{file_prefix}_overview.json"

        save_word_attention_overview(
            avg_coords=state["overview_coords"],
            avg_word_maps=avg_group_maps,
            word_labels=group_labels,
            focus_word_indices=focus_group_indices,
            step_panels=step_panels,
            path=overview_image,
            title=(
                f"{stage_dir.name}: {title_label} cross-attention overview "
                f"({'content renorm' if renorm_content else 'content raw'})"
            ),
            block_index=state["overview_block_index"],
        )

        np.savez_compressed(
            overview_npz,
            coords=np.asarray(state["overview_coords"], dtype=np.int16),
            attention=avg_group_maps.astype(np.float16),
            labels=np.asarray(group_labels),
        )
        save_json(
            overview_json,
            {
                "stage": stage_dir.name,
                "aggregation": aggregation_name,
                "overview_block_index": int(state["overview_block_index"]),
                "overview_count": int(state["overview_count"]),
                "labels": group_labels,
                "token_index_groups": group_token_indices,
                "attention_mode": "content_renorm" if renorm_content else "content_raw",
                "focus_indices": focus_group_indices,
                "focus_labels": [group_labels[i] for i in focus_group_indices],
                "selected_step_indices": [panel["step_index"] for panel in step_panels],
            },
        )

        return {
            f"{file_prefix}_overview": str(overview_image.relative_to(stage_dir.parent)),
            f"{file_prefix}_average_maps": str(overview_npz.relative_to(stage_dir.parent)),
            f"{file_prefix}_overview_meta": str(overview_json.relative_to(stage_dir.parent)),
        }


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器，并在 --help 中展示常用运行示例。"""
    parser = argparse.ArgumentParser(
        description="Trace text-to-space cross-attention in TRELLIS without modifying the original codebase.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python example_text_cross_attention.py --prompt \"a red chair\"\n"
            "  python example_text_cross_attention.py --prompt \"a red chair\" --focus-words chair,red\n"
            "  python example_text_cross_attention.py --prompt \"a wooden table\" "
            "--trace-stages slat --slat-steps 16\n"
            "\n"
            "Outputs will be written to:\n"
            "  output/<case_name>/cross_attention_trace/\n"
        ),
    )
    parser.add_argument("--prompt", required=True, help="Text prompt for TRELLIS text-to-3D.")
    parser.add_argument(
        "--model",
        default="microsoft/TRELLIS-text-xlarge",
        help="Pretrained TRELLIS text model path or Hugging Face repo.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Sampling seed.")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of generated samples.")
    parser.add_argument("--case-name", default="", help="Output case name. Defaults to a prompt-based name.")
    parser.add_argument("--ss-steps", type=int, default=12, help="Sparse-structure sampling steps.")
    parser.add_argument("--slat-steps", type=int, default=12, help="SLat sampling steps.")
    parser.add_argument("--ss-cfg", type=float, default=7.5, help="Sparse-structure CFG strength.")
    parser.add_argument("--slat-cfg", type=float, default=3.0, help="SLat CFG strength.")
    parser.add_argument(
        "--trace-stages",
        default="sparse_structure,slat",
        help="Comma-separated stages to trace: sparse_structure, slat.",
    )
    parser.add_argument(
        "--focus-words",
        default="",
        help="Comma-separated focus words/tokens. If omitted, auto-select the most attended tokens.",
    )
    parser.add_argument(
        "--topk-focus-tokens",
        type=int,
        default=4,
        help="How many focus tokens to visualize when auto-selecting.",
    )
    parser.add_argument(
        "--topk-spatial",
        type=int,
        default=4,
        help="How many final spatial tokens to inspect for each focus token.",
    )
    parser.add_argument(
        "--query-chunk",
        type=int,
        default=2048,
        help="How many spatial queries to process per attention chunk while tracing.",
    )
    parser.add_argument(
        "--attn-backend",
        default=_attn_backend,
        help="Optional ATTN_BACKEND value to expose in the environment before importing TRELLIS.",
    )
    return parser


def make_case_name(prompt: str) -> str:
    """根据 prompt 生成默认输出目录名。"""
    trimmed = sanitize_name(prompt, max_len=64)
    return trimmed or "cross_attention_trace"


def main() -> None:
    """运行完整的 cross-attention tracing 流程并导出结果。"""
    parser = build_parser()
    args = parser.parse_args()

    trace_stages = {
        stage.strip()
        for stage in args.trace_stages.split(",")
        if stage.strip()
    }
    focus_words = [word.strip() for word in args.focus_words.split(",") if word.strip()]
    case_name = args.case_name.strip() or make_case_name(args.prompt)

    root_dir = ensure_dir(Path("output") / case_name / "cross_attention_trace")
    print(f"[INFO] Output directory: {root_dir}")
    print(f"[INFO] Prompt: {args.prompt}")
    print(f"[INFO] Model:  {args.model}")

    tracer: Optional[CrossAttentionTracer] = None
    try:
        with torch.no_grad():
            print("[1/5] Loading pipeline...")
            pipeline = TrellisTextTo3DPipeline.from_pretrained(args.model)
            pipeline.cuda()

            print("[2/5] Encoding text prompt...")
            token_meta = build_token_metadata(pipeline.text_cond_model["tokenizer"], args.prompt)
            save_json(root_dir / "prompt_tokens.json", token_meta)
            cond = pipeline.get_cond([args.prompt])

            tracer = CrossAttentionTracer(
                token_meta=token_meta,
                query_chunk=args.query_chunk,
                focus_words=focus_words,
                topk_focus_tokens=args.topk_focus_tokens,
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

            print("[3/5] Sampling sparse structure and tracing cross-attention...")
            torch.manual_seed(args.seed)
            ss_params = dict(pipeline.sparse_structure_sampler_params)
            ss_params.update({"steps": args.ss_steps, "cfg_strength": args.ss_cfg})
            coords = pipeline.sample_sparse_structure(cond, args.num_samples, ss_params)
            np.save(root_dir / "sparse_structure_coords.npy", coords.detach().cpu().numpy().astype(np.int16))

            print("[4/5] Sampling SLat and tracing cross-attention...")
            slat_params = dict(pipeline.slat_sampler_params)
            slat_params.update({"steps": args.slat_steps, "cfg_strength": args.slat_cfg})
            slat = pipeline.sample_slat(cond, coords, slat_params)
            np.save(root_dir / "slat_coords.npy", slat.coords.detach().cpu().numpy().astype(np.int16))
            np.save(root_dir / "slat_feats.npy", slat.feats.detach().cpu().numpy().astype(np.float16))

            print("[5/5] Exporting visualizations...")
            tracer.export(root_dir)
    finally:
        if tracer is not None:
            tracer.restore()

    save_json(
        root_dir / "run_config.json",
        {
            "prompt": args.prompt,
            "model": args.model,
            "seed": args.seed,
            "num_samples": args.num_samples,
            "ss_steps": args.ss_steps,
            "slat_steps": args.slat_steps,
            "ss_cfg": args.ss_cfg,
            "slat_cfg": args.slat_cfg,
            "trace_stages": sorted(trace_stages),
            "focus_words": focus_words,
            "query_chunk": args.query_chunk,
        },
    )

    with open(root_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write("TRELLIS Text Cross-Attention Trace\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Prompt: {args.prompt}\n")
        f.write(f"Model:  {args.model}\n")
        f.write(f"Seed:   {args.seed}\n\n")
        f.write("Main files:\n")
        f.write("  prompt_tokens.json                   tokenized prompt\n")
        f.write("  trace_manifest.json                  all exported trace artifacts\n")
        f.write("  sparse_structure_coords.npy          sparse structure coordinates\n")
        f.write("  slat_coords.npy                      final SLat coordinates\n")
        f.write("  slat_feats.npy                       final SLat features\n")
        f.write("  <stage>/step_token_mean_content_raw.npy      content-token means after dropping special tokens only\n")
        f.write("  <stage>/step_token_mean_content_renorm.npy   content-token means after dropping special tokens and renormalizing\n")
        f.write("  <stage>/final_step_block_token_mean_content_raw.npy  final-step block means, drop special tokens only\n")
        f.write("  <stage>/final_step_block_token_mean_content_renorm.npy  final-step block means, content renorm\n")
        f.write("  <stage>/step_token_heatmap.png       token importance over sampling steps (all valid tokens)\n")
        f.write("  <stage>/final_step_block_token_heatmap.png  token importance over transformer blocks (all valid tokens)\n")
        f.write("  <stage>/step_token_heatmap_content_raw.png   content-token heatmap after dropping special tokens only\n")
        f.write("  <stage>/step_token_heatmap_content_renorm.png  content-token heatmap after dropping special tokens and renormalizing\n")
        f.write("  <stage>/focus_token_step_curves.png  focus-token curves over steps (all valid tokens)\n")
        f.write("  <stage>/focus_token_step_curves_content_raw.png  focus-token curves with special tokens removed only\n")
        f.write("  <stage>/focus_token_step_curves_content_renorm.png  focus-token curves with content-token renormalization\n")
        f.write("  <stage>/subword_attention_content_raw_overview.png  subword-to-3D overview, drop special tokens only\n")
        f.write("  <stage>/subword_attention_content_renorm_overview.png  subword-to-3D overview, renorm after dropping special tokens\n")
        f.write("  <stage>/word_attention_content_raw_overview.png  word-to-3D overview, drop special tokens only\n")
        f.write("  <stage>/word_attention_content_renorm_overview.png  word-to-3D overview, renorm after dropping special tokens\n")
        f.write("  <stage>/token_*/step_progress.png    spatial attention for one text token across steps\n")
        f.write("  <stage>/token_*/block_progress.png   spatial attention for one text token across blocks\n")
        f.write("  <stage>/token_*/top_spatial_tokens.json      top spatial tokens in the final map, with extra raw/renorm bar paths\n")
        f.write("  <stage>/raw_maps/*.npz               raw maps plus content_raw/content_renorm attention arrays\n")

    print(f"[DONE] Trace saved to: {root_dir}")


if __name__ == "__main__":
    main()
