from __future__ import annotations

import math
import types
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from .base import AttentionHook


@dataclass(frozen=True)
class StageConfig:
    """Configuration for a pipeline stage."""
    name: str
    enabled: bool
    t_start: float
    t_end: float
    strength: float

    def contains(self, t_norm: float) -> bool:
        """Check if normalized timestep is within this stage's range.

        Args:
            t_norm: Normalized timestep (0.0 to 1.0)

        Returns:
            True if timestep is within range and stage is enabled
        """
        lo = min(self.t_start, self.t_end)
        hi = max(self.t_start, self.t_end)
        return self.enabled and lo <= t_norm <= hi and self.strength > 0.0


class PromptToPromptHook(AttentionHook):
    """Prompt-to-Prompt attention injection hook.

    Injects source attention for unmasked tokens during denoising.
    """

    def __init__(
        self,
        source_cond: torch.Tensor,
        edit_cond: torch.Tensor,
        neg_cond: torch.Tensor,
        token_meta: dict,
        stage_configs: Dict[str, StageConfig],
        query_chunk: int = 1024,
    ):
        """Initialize Prompt-to-Prompt hook.

        Args:
            source_cond: Source condition tensor
            edit_cond: Edit condition tensor
            neg_cond: Negative condition tensor
            token_meta: Token metadata from build_image_token_metadata
            stage_configs: Stage configurations
            query_chunk: Query chunk size for attention computation
        """
        super().__init__()
        self.source_cond = source_cond
        self.edit_cond_signature = self._tensor_signature(edit_cond)
        self.neg_cond_signature = self._tensor_signature(neg_cond)
        self.token_meta = token_meta
        self.stage_configs = stage_configs
        self.query_chunk = max(1, int(query_chunk))

        self.current_forward_context: Optional[dict] = None
        self._index_cache: Dict[Tuple[str, str], Tuple[torch.Tensor, torch.Tensor]] = {}

    def patch_model(self, model: torch.nn.Module, stage_name: str) -> None:
        """Patch model for Prompt-to-Prompt editing.

        Args:
            model: Model to patch (sparse_structure_flow_model or slat_flow_model)
            stage_name: Stage name ("sparse_structure" or "slat")
        """
        config = self.stage_configs[stage_name]
        if not config.enabled:
            return

        self._patch_model_forward(stage_name, model)
        self._patch_cross_modules(stage_name, model)

    def _patch_model_forward(self, stage_name: str, model: torch.nn.Module) -> None:
        """Patch model forward to track context."""
        original_forward = model.forward
        self._save_original(model, "forward", original_forward)
        hook = self

        def wrapped_forward(model_self, x, t, cond):
            pass_kind = hook._infer_pass_kind(cond)
            t_value = float(t[0].detach().float().cpu().item()) if torch.is_tensor(t) else float(t)
            hook.current_forward_context = {
                "stage": stage_name,
                "pass_kind": pass_kind,
                "t_raw": t_value,
                "t_norm": t_value / 1000.0,
            }
            try:
                return original_forward(x, t, cond)
            finally:
                hook.current_forward_context = None

        model.forward = types.MethodType(wrapped_forward, model)

    def _patch_cross_modules(self, stage_name: str, model: torch.nn.Module) -> None:
        """Patch cross-attention modules."""
        # Import here to avoid circular dependency
        try:
            from trellis.modules.attention import MultiHeadAttention, SparseMultiHeadAttention
        except ImportError:
            # Fallback for different import paths
            MultiHeadAttention = None
            SparseMultiHeadAttention = None
            for _, module in model.named_modules():
                if module.__class__.__name__ == "MultiHeadAttention":
                    MultiHeadAttention = module.__class__
                elif module.__class__.__name__ == "SparseMultiHeadAttention":
                    SparseMultiHeadAttention = module.__class__
                if MultiHeadAttention and SparseMultiHeadAttention:
                    break

        for _, module in model.named_modules():
            if MultiHeadAttention and isinstance(module, MultiHeadAttention):
                if getattr(module, "_type", None) == "cross":
                    self._patch_dense_cross(stage_name, module)
            elif SparseMultiHeadAttention and isinstance(module, SparseMultiHeadAttention):
                if getattr(module, "_type", None) == "cross":
                    self._patch_sparse_cross(stage_name, module)

    def _patch_dense_cross(self, stage_name: str, module) -> None:
        """Patch dense cross-attention module."""
        original_forward = module.forward
        self._save_original(module, "forward", original_forward)
        hook = self

        def wrapped_forward(attn_self, x, context=None, indices=None):
            strength = hook._active_strength(stage_name)
            if strength <= 0.0 or context is None:
                return original_forward(x, context, indices)

            batch_size, num_queries, _ = x.shape
            q = attn_self.to_q(x).reshape(batch_size, num_queries, attn_self.num_heads, -1)
            kv_edit = attn_self.to_kv(context).reshape(batch_size, context.shape[1], 2, attn_self.num_heads, -1)
            k_edit, v_edit = kv_edit.unbind(dim=2)

            source_context = hook._source_context_for_batch(
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

            h = hook._chunked_prompt_to_prompt_attention(q, k_src, k_edit, v_edit, strength)
            h = h.reshape(batch_size, num_queries, -1)
            h = attn_self.to_out(h)
            return h

        module.forward = types.MethodType(wrapped_forward, module)

    def _patch_sparse_cross(self, stage_name: str, module) -> None:
        """Patch sparse cross-attention module."""
        original_forward = module.forward
        self._save_original(module, "forward", original_forward)
        hook = self

        def wrapped_forward(attn_self, x, context=None):
            strength = hook._active_strength(stage_name)
            if strength <= 0.0 or context is None:
                return original_forward(x, context)

            q = attn_self._linear(attn_self.to_q, x)
            q = attn_self._reshape_chs(q, (attn_self.num_heads, -1))

            kv_edit = attn_self._linear(attn_self.to_kv, context)
            kv_edit = attn_self._fused_pre(kv_edit, num_fused=2)
            k_edit, v_edit = kv_edit.unbind(dim=2)

            source_context = hook._source_context_for_batch(
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

            h = hook._chunked_sparse_prompt_to_prompt_attention(q, k_src, k_edit, v_edit, strength)
            h = attn_self._reshape_chs(h, (-1,))
            h = attn_self._linear(attn_self.to_out, h)
            return h

        module.forward = types.MethodType(wrapped_forward, module)

    def _active_strength(self, stage_name: str) -> float:
        """Get active strength for current context."""
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
        """Get source context repeated for batch."""
        source_context = self.source_cond.to(device=device, dtype=dtype)
        if source_context.shape[0] == 1 and batch_size > 1:
            source_context = source_context.repeat(batch_size, 1, 1)
        return source_context

    def _token_indices(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached token indices for device."""
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
        """Mix source and edit attention maps."""
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
        """Compute Prompt-to-Prompt attention in chunks."""
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
        q,  # SparseTensor
        k_src: torch.Tensor,
        k_edit: torch.Tensor,
        v_edit: torch.Tensor,
        strength: float,
    ):
        """Compute Prompt-to-Prompt attention for sparse tensors."""
        # Import SparseTensor here to avoid circular dependency
        try:
            from trellis.modules.sparse.basic import SparseTensor
        except ImportError:
            SparseTensor = type(q)

        new_feats = torch.empty_like(q.feats)
        head_dim = k_src.shape[-1]
        scale = 1.0 / math.sqrt(head_dim)

        for batch_idx in range(q.shape[0]):
            batch_slice = q.layout[batch_idx]
            q_batch = q.feats[batch_slice]
            num_queries = q_batch.shape[0]

            k_src_batch = k_src[batch_idx].permute(1, 2, 0).float()
            k_edit_batch = k_edit[batch_idx].permute(1, 2, 0).float()
            v_edit_batch = v_edit[batch_idx].permute(1, 0, 2).float()

            out_chunks = []
            for start in range(0, num_queries, self.query_chunk):
                end = min(start + self.query_chunk, num_queries)
                q_chunk = q_batch[start:end].permute(1, 0, 2).float()

                scores_src = torch.matmul(q_chunk, k_src_batch) * scale
                scores_edit = torch.matmul(q_chunk, k_edit_batch) * scale
                scores_src = scores_src - scores_src.amax(dim=-1, keepdim=True)
                scores_edit = scores_edit - scores_edit.amax(dim=-1, keepdim=True)

                attn_src = torch.softmax(scores_src, dim=-1)
                attn_edit = torch.softmax(scores_edit, dim=-1)
                attn_mix = self._mix_attention_maps(attn_src, attn_edit, strength)
                out_chunk = torch.matmul(attn_mix, v_edit_batch)
                out_chunks.append(out_chunk.permute(1, 0, 2).to(q.dtype))

            new_feats[batch_slice] = torch.cat(out_chunks, dim=0)

        return SparseTensor(feats=new_feats, coords=q.coords)

    @staticmethod
    def _tensor_signature(tensor: torch.Tensor, rows: int = 3, cols: int = 8) -> torch.Tensor:
        """Extract tensor signature for comparison."""
        return tensor[:1, :rows, :cols].detach().float().cpu()

    def _infer_pass_kind(self, cond: torch.Tensor) -> str:
        """Infer whether this is a conditional or negative pass."""
        from editing.utils import signature_distance

        sig = self._tensor_signature(cond)
        if signature_distance(sig, self.edit_cond_signature) <= signature_distance(sig, self.neg_cond_signature):
            return "cond"
        return "neg"
