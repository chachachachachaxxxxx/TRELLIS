#!/usr/bin/env python3
from __future__ import annotations
"""
TRELLIS Prompt-to-Prompt text editing example.

This script keeps TRELLIS core code untouched and injects Prompt-to-Prompt
cross-attention edits at runtime via monkey patching.

Usage examples:
  python example_text_prompt_to_prompt.py \
    --source-prompt "a cute cat statue" \
    --edit-prompt "a cute tiger statue" \
    --case-name "cat_to_tiger"

  python example_text_prompt_to_prompt.py \
    --source-prompt "a red chair with wooden legs" \
    --edit-prompt "a blue chair with wooden legs" \
    --inject-stages st \
    --ss-t-start 1.0 --ss-t-end 0.3

  python example_text_prompt_to_prompt.py \
    --source-prompt "a ceramic teapot with flowers" \
    --edit-prompt "a ceramic teapot with dragons" \
    --inject-stages slat \
    --slat-t-start 0.8 --slat-t-end 0.0
"""

import argparse
import gc
import importlib.util
import json
import math
import os
import re
import sys
import types
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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

import torch


MultiHeadAttention = None
SparseMultiHeadAttention = None
SparseTensor = None
TrellisTextTo3DPipeline = None


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def configure_attention_backend(requested_backend: str) -> str:
    backend = requested_backend.strip().lower()
    if backend:
        if backend not in {"flash_attn", "xformers"}:
            raise RuntimeError(
                "TRELLIS text-to-3D requires sparse attention, so "
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
                "Neither flash_attn nor xformers is installed, but TRELLIS text-to-3D "
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


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def detach_object_tensors(obj) -> None:
    for name, value in vars(obj).items():
        if torch.is_tensor(value):
            setattr(obj, name, value.detach())


def detach_outputs(outputs: dict) -> None:
    for samples in outputs.values():
        for sample in samples:
            detach_object_tensors(sample)


def slugify(text: str, max_len: int = 96) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^0-9a-zA-Z_\u4e00-\u9fff]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    if not text:
        text = "prompt_to_prompt_edit"
    return text[:max_len]


def clean_token_label(token: str) -> str:
    label = token.replace("</w>", "")
    label = label.replace("<|startoftext|>", "[BOS]")
    label = label.replace("<|endoftext|>", "[EOS]")
    label = label.strip()
    return label if label else token


def build_token_metadata(tokenizer, prompt: str) -> dict:
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

    return {
        "prompt": prompt,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "raw_tokens": raw_tokens,
        "display_tokens": display_tokens,
        "valid_token_count": valid_token_count,
        "special_mask": special_mask,
    }


def build_prompt_alignment(source_meta: dict, edit_meta: dict) -> dict:
    src_content_indices = [
        idx
        for idx in range(source_meta["valid_token_count"])
        if not source_meta["special_mask"][idx]
    ]
    edit_content_indices = [
        idx
        for idx in range(edit_meta["valid_token_count"])
        if not edit_meta["special_mask"][idx]
    ]

    src_seq = [source_meta["raw_tokens"][idx] for idx in src_content_indices]
    edit_seq = [edit_meta["raw_tokens"][idx] for idx in edit_content_indices]

    matcher = SequenceMatcher(a=src_seq, b=edit_seq, autojunk=False)
    src_keep_indices: List[int] = []
    edit_keep_indices: List[int] = []
    aligned_pairs: List[dict] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            continue
        for offset in range(i2 - i1):
            src_full_idx = src_content_indices[i1 + offset]
            edit_full_idx = edit_content_indices[j1 + offset]
            src_keep_indices.append(src_full_idx)
            edit_keep_indices.append(edit_full_idx)
            aligned_pairs.append(
                {
                    "source_token_index": src_full_idx,
                    "edit_token_index": edit_full_idx,
                    "source_token": source_meta["display_tokens"][src_full_idx],
                    "edit_token": edit_meta["display_tokens"][edit_full_idx],
                }
            )

    src_keep_set = set(src_keep_indices)
    edit_keep_set = set(edit_keep_indices)
    edited_source_indices = [
        idx
        for idx in src_content_indices
        if idx not in src_keep_set
    ]
    edited_edit_indices = [
        idx
        for idx in edit_content_indices
        if idx not in edit_keep_set
    ]

    return {
        "source_prompt": source_meta["prompt"],
        "edit_prompt": edit_meta["prompt"],
        "source_tokens": source_meta["display_tokens"],
        "edit_tokens": edit_meta["display_tokens"],
        "source_keep_indices": src_keep_indices,
        "edit_keep_indices": edit_keep_indices,
        "source_keep_tokens": [source_meta["display_tokens"][idx] for idx in src_keep_indices],
        "edit_keep_tokens": [edit_meta["display_tokens"][idx] for idx in edit_keep_indices],
        "source_edited_indices": edited_source_indices,
        "edit_edited_indices": edited_edit_indices,
        "source_edited_tokens": [source_meta["display_tokens"][idx] for idx in edited_source_indices],
        "edit_edited_tokens": [edit_meta["display_tokens"][idx] for idx in edited_edit_indices],
        "aligned_pairs": aligned_pairs,
    }


def tensor_signature(tensor: torch.Tensor, rows: int = 3, cols: int = 8) -> torch.Tensor:
    return tensor[:1, :rows, :cols].detach().float().cpu()


def signature_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.max(torch.abs(a - b)).item())


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


class PromptToPromptEditor:
    def __init__(
        self,
        source_cond: torch.Tensor,
        edit_cond: torch.Tensor,
        neg_cond: torch.Tensor,
        alignment: dict,
        stage_configs: Dict[str, StageConfig],
        query_chunk: int = 1024,
    ):
        self.source_cond = source_cond
        self.edit_cond_signature = tensor_signature(edit_cond)
        self.neg_cond_signature = tensor_signature(neg_cond)
        self.alignment = alignment
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
                self.alignment["source_keep_indices"],
                device=device,
                dtype=torch.long,
            )
            edit_idx = torch.tensor(
                self.alignment["edit_keep_indices"],
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
    parser = argparse.ArgumentParser(description="Prompt-to-Prompt editing for TRELLIS text-to-3D.")
    parser.add_argument("--model", default="microsoft/TRELLIS-text-xlarge", help="Pipeline checkpoint or HF repo.")
    parser.add_argument("--source-prompt", required=True, help="Original source prompt.")
    parser.add_argument("--edit-prompt", required=True, help="Edited target prompt.")
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
    parser.add_argument("--case-name", default="", help="Optional output case name.")
    parser.add_argument("--skip-source", action="store_true", help="Skip generating the original source model.")
    parser.add_argument("--skip-render", action="store_true", help="Skip rendering mp4 previews.")
    parser.add_argument(
        "--skip-radiance-field-render",
        action="store_true",
        help="Skip radiance-field mp4 preview rendering while keeping gaussian / mesh previews.",
    )
    parser.add_argument("--skip-glb", action="store_true", help="Skip exporting GLB.")
    parser.add_argument("--skip-ply", action="store_true", help="Skip exporting gaussian PLY.")
    parser.add_argument(
        "--attn-backend",
        default=_attn_backend or "",
        help="Attention backend override: flash_attn or xformers.",
    )
    return parser.parse_args()


def save_preview_video(render_utils, imageio, sample, out_path: Path, channel: str, label: str) -> bool:
    release_cuda_memory()
    try:
        video = render_utils.render_video(sample)[channel]
        imageio.mimsave(str(out_path), video, fps=30)
        return True
    except Exception as exc:
        message = str(exc)
        message_lower = message.lower()
        extra = ""
        if "out of memory" in message_lower:
            extra = (
                " This preview step ran out of VRAM. "
                "Re-run with --skip-radiance-field-render or --skip-render if you only need the generated assets."
            )
        elif "operation not supported on global/shared address space" in message_lower and label == "radiance_field":
            extra = (
                " This often points to a diffoctreerast/CUDA compatibility issue. "
                "Re-run with --skip-radiance-field-render or --skip-render if you only need the generated assets."
            )
        print(
            f"Warning: failed to render {label} preview at {out_path.name}: {message}.{extra}",
            file=sys.stderr,
        )
        return False
    finally:
        if "video" in locals():
            del video
        release_cuda_memory()


def save_outputs(
    outputs: dict,
    out_dir: Path,
    skip_render: bool,
    skip_radiance_field_render: bool,
    skip_glb: bool,
    skip_ply: bool,
) -> None:
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
                save_preview_video(
                    render_utils,
                    imageio,
                    outputs["gaussian"][sample_idx],
                    out_dir / f"{prefix}_gs.mp4",
                    "color",
                    "gaussian",
                )
            if "radiance_field" in outputs and not skip_radiance_field_render:
                save_preview_video(
                    render_utils,
                    imageio,
                    outputs["radiance_field"][sample_idx],
                    out_dir / f"{prefix}_rf.mp4",
                    "color",
                    "radiance_field",
                )
            if "mesh" in outputs:
                save_preview_video(
                    render_utils,
                    imageio,
                    outputs["mesh"][sample_idx],
                    out_dir / f"{prefix}_mesh.mp4",
                    "normal",
                    "mesh",
                )

        if not skip_glb and "gaussian" in outputs and "mesh" in outputs:
            release_cuda_memory()
            glb = postprocessing_utils.to_glb(
                outputs["gaussian"][sample_idx],
                outputs["mesh"][sample_idx],
                simplify=0.95,
                texture_size=1024,
            )
            glb.export(str(out_dir / f"{prefix}.glb"))
            del glb
            release_cuda_memory()

        if not skip_ply and "gaussian" in outputs:
            release_cuda_memory()
            outputs["gaussian"][sample_idx].save_ply(str(out_dir / f"{prefix}.ply"))


def main() -> int:
    args = parse_args()
    backend = configure_attention_backend(args.attn_backend)

    global MultiHeadAttention
    global SparseMultiHeadAttention
    global SparseTensor
    global TrellisTextTo3DPipeline

    from trellis.modules.attention.modules import MultiHeadAttention as _MultiHeadAttention
    from trellis.modules.sparse.attention.modules import SparseMultiHeadAttention as _SparseMultiHeadAttention
    from trellis.modules.sparse.basic import SparseTensor as _SparseTensor
    from trellis.pipelines import TrellisTextTo3DPipeline as _TrellisTextTo3DPipeline

    MultiHeadAttention = _MultiHeadAttention
    SparseMultiHeadAttention = _SparseMultiHeadAttention
    SparseTensor = _SparseTensor
    TrellisTextTo3DPipeline = _TrellisTextTo3DPipeline

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

    pipeline = TrellisTextTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()

    tokenizer = pipeline.text_cond_model["tokenizer"]
    source_meta = build_token_metadata(tokenizer, args.source_prompt)
    edit_meta = build_token_metadata(tokenizer, args.edit_prompt)
    alignment = build_prompt_alignment(source_meta, edit_meta)
    if not alignment["aligned_pairs"]:
        print("Warning: no unchanged token pairs were aligned; injection will behave like a normal edit run.")

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
        case_name = slugify(f"{args.source_prompt}_to_{args.edit_prompt}")
    out_dir = ensure_dir(Path("output") / case_name / "prompt_to_prompt_edit")
    source_out_dir = Path("output") / case_name / "source_original"

    save_json(
        out_dir / "config.json",
        {
            "model": args.model,
            "attn_backend": backend,
            "source_prompt": args.source_prompt,
            "edit_prompt": args.edit_prompt,
            "seed": args.seed,
            "num_samples": args.num_samples,
            "inject_stages": inject_stages,
            "query_chunk": args.query_chunk,
            "skip_radiance_field_render": args.skip_radiance_field_render,
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
    save_json(out_dir / "source_tokens.json", source_meta)
    save_json(out_dir / "edit_tokens.json", edit_meta)
    save_json(out_dir / "token_alignment.json", alignment)

    source_outputs = None
    with torch.no_grad():
        source_cond_dict = pipeline.get_cond([args.source_prompt])
        edit_cond = pipeline.get_cond([args.edit_prompt])

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
            detach_outputs(source_outputs)
            del source_coords, source_slat
            release_cuda_memory()

        editor = PromptToPromptEditor(
            source_cond=source_cond_dict["cond"],
            edit_cond=edit_cond["cond"],
            neg_cond=edit_cond["neg_cond"],
            alignment=alignment,
            stage_configs=stage_configs,
            query_chunk=args.query_chunk,
        )

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

    detach_outputs(outputs)
    del source_cond_dict
    del edit_cond
    del coords
    del slat
    del editor
    del tokenizer
    del pipeline
    release_cuda_memory()

    if source_outputs is not None:
        ensure_dir(source_out_dir)
        save_outputs(
            outputs=source_outputs,
            out_dir=source_out_dir,
            skip_render=args.skip_render,
            skip_radiance_field_render=args.skip_radiance_field_render,
            skip_glb=args.skip_glb,
            skip_ply=args.skip_ply,
        )
        print(f"Saved source original results to: {source_out_dir}")

    save_outputs(
        outputs=outputs,
        out_dir=out_dir,
        skip_render=args.skip_render,
        skip_radiance_field_render=args.skip_radiance_field_render,
        skip_glb=args.skip_glb,
        skip_ply=args.skip_ply,
    )

    print(f"Saved Prompt-to-Prompt edit results to: {out_dir}")
    print(f"Kept token pairs: {len(alignment['aligned_pairs'])}")
    print(f"Injected stages: {inject_stages if inject_stages else ['none']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
