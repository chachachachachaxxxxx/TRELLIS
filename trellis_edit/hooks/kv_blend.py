from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from trellis_edit.samplers import SparseLatentBlendMask
from trellis_edit.utils import signature_distance, tensor_signature

from .base import AttentionHook


@dataclass(frozen=True)
class KVBlendStageConfig:
    name: str
    enabled: bool
    t_start: float = 1.0
    t_end: float = 0.0
    self_attention: bool = True
    cross_attention: bool = True

    def contains(self, t_norm: float) -> bool:
        lo = min(self.t_start, self.t_end)
        hi = max(self.t_start, self.t_end)
        return self.enabled and lo <= t_norm <= hi


class KVBlendHook(AttentionHook):
    """VoxHammer-style KV cache/blend hook for SS and SLAT stages."""

    def __init__(
        self,
        *,
        neg_cond: torch.Tensor,
        stage_configs: Dict[str, KVBlendStageConfig],
        stage_masks: Dict[str, Dict[str, Any]],
    ):
        super().__init__()
        self.neg_cond_signature = tensor_signature(neg_cond)
        self.stage_configs = stage_configs
        self.stage_masks = stage_masks
        self.current_phase: Optional[str] = None
        self.current_eval_spec: Optional[Any] = None
        self.current_source_capture: Optional[dict[str, Any]] = None
        self.current_forward_context: Optional[dict] = None
        self._pending_step_source_snapshots: Dict[str, Dict[str, Any]] = {}
        self._active_step_source_snapshots: Dict[str, Dict[str, Any]] = {}
        self._active_step_source_times: Dict[str, float] = {}
        self._dense_mask_cache: Dict[tuple[str, str, str, int], torch.Tensor] = {}
        self._sparse_mask_cache: Dict[tuple[str, str, str], Any] = {}

    @staticmethod
    def _eval_spec_field(eval_spec: Any, field_name: str) -> Any:
        if isinstance(eval_spec, dict):
            return eval_spec[field_name]
        return getattr(eval_spec, field_name)

    def set_eval_context(self, phase: Optional[str], *, eval_spec: Optional[Any] = None) -> None:
        if phase not in {None, "inversion", "denoise", "source_capture"}:
            raise ValueError(f"Unknown KV blend phase: {phase}")
        if phase is not None and eval_spec is None:
            raise ValueError("KV blend eval context requires eval_spec when phase is set.")
        self.current_phase = phase
        self.current_eval_spec = eval_spec

    def requires_step_source_snapshot(self, stage_name: str, logical_t: float) -> bool:
        config = self.stage_configs.get(stage_name)
        if config is None:
            return False
        return bool(config.contains(float(logical_t)) and (config.self_attention or config.cross_attention))

    def begin_step_source_capture(self, *, stage_name: str, logical_t: float) -> None:
        self.current_source_capture = {
            "stage": stage_name,
            "logical_t": float(logical_t),
        }
        self._pending_step_source_snapshots[stage_name] = {}

    def cancel_step_source_capture(self, *, stage_name: str) -> None:
        if self.current_source_capture is not None and self.current_source_capture.get("stage") == stage_name:
            self.current_source_capture = None
        self._pending_step_source_snapshots.pop(stage_name, None)

    def end_step_source_capture(self, *, stage_name: str, logical_t: float) -> Dict[str, Any]:
        if self.current_source_capture is None or self.current_source_capture.get("stage") != stage_name:
            raise RuntimeError(f"No pending KV blend source capture for stage={stage_name!r}.")
        capture_t = float(self.current_source_capture["logical_t"])
        if abs(capture_t - float(logical_t)) > 1e-6:
            raise RuntimeError(
                f"KV blend source capture logical_t mismatch for stage={stage_name!r}: "
                f"expected {capture_t}, got {float(logical_t)}"
            )
        self.current_source_capture = None
        return self._pending_step_source_snapshots.pop(stage_name, {})

    def set_step_source_snapshot(
        self,
        *,
        stage_name: str,
        logical_t: float,
        snapshot: Dict[str, Any],
    ) -> None:
        self._active_step_source_snapshots[stage_name] = snapshot
        self._active_step_source_times[stage_name] = float(logical_t)

    def clear_step_source_snapshot(self, *, stage_name: str | None = None) -> None:
        if stage_name is None:
            self._active_step_source_snapshots.clear()
            self._active_step_source_times.clear()
            return
        self._active_step_source_snapshots.pop(stage_name, None)
        self._active_step_source_times.pop(stage_name, None)

    def patch_model(self, model: torch.nn.Module, stage_name: str) -> None:
        config = self.stage_configs.get(stage_name)
        if config is None or not config.enabled:
            return
        self._patch_model_forward(stage_name, model)
        for layer_idx, block in enumerate(getattr(model, "blocks", [])):
            if hasattr(block, "self_attn"):
                self._patch_attention_module(stage_name, block.self_attn, layer_idx, "self")
            if hasattr(block, "cross_attn"):
                self._patch_attention_module(stage_name, block.cross_attn, layer_idx, "cross")

    def _patch_model_forward(self, stage_name: str, model: torch.nn.Module) -> None:
        original_forward = model.forward
        self._save_original(model, "forward", original_forward)
        hook = self

        def wrapped_forward(model_self, x, t, cond):
            if hook.current_phase is not None and hook.current_eval_spec is not None:
                logical_t = float(hook._eval_spec_field(hook.current_eval_spec, "logical_t"))
                phase = hook.current_phase
                eval_idx = int(hook._eval_spec_field(hook.current_eval_spec, "eval_idx"))
            else:
                source_capture = hook.current_source_capture
                if source_capture is not None and source_capture.get("stage") == stage_name:
                    logical_t = float(source_capture["logical_t"])
                    phase = "source_capture"
                    eval_idx = int(source_capture.get("eval_idx", 0))
                else:
                    return original_forward(x, t, cond)
            t_value = float(t[0].detach().float().cpu().item()) if torch.is_tensor(t) else float(t)
            actual_t = t_value / 1000.0
            hook.current_forward_context = {
                "stage": stage_name,
                "phase": phase,
                "pass_kind": hook._infer_pass_kind(cond),
                "actual_t": actual_t,
                "logical_t": logical_t,
                "eval_idx": eval_idx,
            }
            try:
                return original_forward(x, t, cond)
            finally:
                hook.current_forward_context = None

        model.forward = types.MethodType(wrapped_forward, model)

    def _patch_attention_module(self, stage_name: str, module, layer_idx: int, attn_type: str) -> None:
        if hasattr(module, "_linear") and hasattr(module, "_fused_pre"):
            if attn_type == "self":
                self._patch_sparse_self(stage_name, module, layer_idx)
            else:
                self._patch_sparse_cross(stage_name, module, layer_idx)
            return

        if attn_type == "self":
            self._patch_dense_self(stage_name, module, layer_idx)
        else:
            self._patch_dense_cross(stage_name, module, layer_idx)

    def _is_active(self, stage_name: str, attn_type: str) -> bool:
        ctx = self.current_forward_context
        if ctx is None or ctx["stage"] != stage_name:
            return False
        config = self.stage_configs[stage_name]
        if not config.contains(float(ctx["logical_t"])):
            return False
        if attn_type == "self":
            return config.self_attention
        return config.cross_attention

    @staticmethod
    def _snapshot_key(pass_kind: str, eval_idx: int, layer_idx: int, attn_type: str) -> str:
        return "|".join((pass_kind, str(eval_idx), str(layer_idx), attn_type))

    def _capture_dense_kv(
        self,
        stage_name: str,
        layer_idx: int,
        attn_type: str,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        ctx = self.current_forward_context
        assert ctx is not None
        pending_snapshot = self._pending_step_source_snapshots.setdefault(stage_name, {})
        key = self._snapshot_key(ctx["pass_kind"], int(ctx["eval_idx"]), layer_idx, attn_type)
        pending_snapshot[f"{key}|k"] = k.detach().cpu()
        pending_snapshot[f"{key}|v"] = v.detach().cpu()

    def _lookup_dense_kv(
        self,
        stage_name: str,
        layer_idx: int,
        attn_type: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        ctx = self.current_forward_context
        assert ctx is not None
        active_t = self._active_step_source_times.get(stage_name)
        if active_t is None or abs(active_t - float(ctx["logical_t"])) > 1e-6:
            return None, None
        active_snapshot = self._active_step_source_snapshots.get(stage_name)
        if active_snapshot is None:
            return None, None
        key = self._snapshot_key(ctx["pass_kind"], int(ctx["eval_idx"]), layer_idx, attn_type)
        cache_k = active_snapshot.get(f"{key}|k")
        cache_v = active_snapshot.get(f"{key}|v")
        if cache_k is None or cache_v is None:
            return None, None
        return cache_k.to(device=device, dtype=dtype), cache_v.to(device=device, dtype=dtype)

    def _dense_token_mask(
        self,
        stage_name: str,
        attn_type: str,
        device: torch.device,
        dtype: torch.dtype,
        seq_len: int,
    ) -> Optional[torch.Tensor]:
        token_mask = self.stage_masks.get(stage_name, {}).get(attn_type)
        if token_mask is None:
            return None
        key = (stage_name, attn_type, str(device), seq_len)
        if key not in self._dense_mask_cache:
            if int(token_mask.numel()) != int(seq_len):
                raise RuntimeError(
                    f"{stage_name}.{attn_type} KV mask length mismatch: "
                    f"expected {seq_len}, got {token_mask.numel()}"
                )
            self._dense_mask_cache[key] = token_mask.to(device=device, dtype=torch.float32).view(1, 1, seq_len, 1)
        return self._dense_mask_cache[key].to(dtype=dtype)

    def _sparse_self_mask(self, stage_name: str, device: torch.device) -> Optional[torch.Tensor | SparseLatentBlendMask]:
        sparse_mask = self.stage_masks.get(stage_name, {}).get("self")
        if sparse_mask is None:
            return None
        key = (stage_name, "self", str(device))
        if key not in self._sparse_mask_cache:
            if isinstance(sparse_mask, SparseLatentBlendMask):
                edit_weights = sparse_mask.edit_weights
                self._sparse_mask_cache[key] = SparseLatentBlendMask(
                    coords=sparse_mask.coords.to(device=device),
                    edit_weights=None if edit_weights is None else edit_weights.to(device=device),
                )
            else:
                self._sparse_mask_cache[key] = sparse_mask.to(device=device)
        return self._sparse_mask_cache[key]

    @staticmethod
    def _resolve_sparse_self_mask(
        sparse_mask: torch.Tensor | SparseLatentBlendMask | None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if sparse_mask is None:
            return None, None
        if isinstance(sparse_mask, SparseLatentBlendMask):
            return sparse_mask.coords, sparse_mask.edit_weights
        return sparse_mask, None

    @staticmethod
    def _prepare_sparse_edit_weights(
        edit_weights: torch.Tensor,
        valid: torch.Tensor,
        target_feats: torch.Tensor,
    ) -> torch.Tensor:
        weights = edit_weights.to(device=target_feats.device, dtype=target_feats.dtype)
        weights = weights.reshape(weights.shape[0], -1)
        if weights.shape[1] != 1:
            raise RuntimeError(
                "Sparse KV blend expects scalar soft-mask weights per preserve coordinate, "
                f"got shape {tuple(edit_weights.shape)}"
            )
        if weights.shape[0] != valid.shape[0]:
            raise RuntimeError(
                "Sparse KV blend edit_weights length mismatch: "
                f"expected {valid.shape[0]}, got {weights.shape[0]}"
            )
        weights = weights[valid]
        return weights.reshape(weights.shape[0], *([1] * (target_feats.ndim - 1)))

    @staticmethod
    def _blend_sparse_tensor(
        current,
        cached,
        preserve_coords: torch.Tensor,
        edit_weights: Optional[torch.Tensor],
    ):
        if preserve_coords.shape[0] == 0:
            return current

        match_current = (current.coords.unsqueeze(1) == preserve_coords.unsqueeze(0)).all(dim=-1)
        match_cached = (cached.coords.to(current.coords.device).unsqueeze(1) == preserve_coords.unsqueeze(0)).all(dim=-1)
        preserve_in_current = match_current.any(dim=0)
        preserve_in_cached = match_cached.any(dim=0)
        valid = preserve_in_current & preserve_in_cached
        if not valid.any():
            return current

        idx_current = match_current[:, valid].float().argmax(0)
        idx_cached = match_cached[:, valid].float().argmax(0)
        feats = current.feats.clone()
        cached_feats = cached.feats.to(device=feats.device, dtype=feats.dtype)[idx_cached]
        if edit_weights is None:
            feats[idx_current] = cached_feats
        else:
            current_feats = feats[idx_current].clone()
            weights = KVBlendHook._prepare_sparse_edit_weights(edit_weights, valid, current_feats)
            feats[idx_current] = current_feats * weights + cached_feats * (1.0 - weights)
        return current.replace(feats)

    def _apply_dense_kv(
        self,
        stage_name: str,
        layer_idx: int,
        attn_type: str,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._is_active(stage_name, attn_type):
            return k, v
        ctx = self.current_forward_context
        assert ctx is not None
        if ctx["phase"] == "source_capture":
            self._capture_dense_kv(stage_name, layer_idx, attn_type, k, v)
            return k, v
        if ctx["phase"] != "denoise":
            return k, v

        mask = self._dense_token_mask(stage_name, attn_type, k.device, k.dtype, k.shape[2])
        if mask is None:
            return k, v
        cache_k, cache_v = self._lookup_dense_kv(stage_name, layer_idx, attn_type, k.device, k.dtype)
        if cache_k is None or cache_v is None:
            return k, v
        k = k * mask + cache_k * (1.0 - mask)
        v = v * mask + cache_v * (1.0 - mask)
        return k, v

    def _patch_dense_self(self, stage_name: str, module, layer_idx: int) -> None:
        original_forward = module.forward
        self._save_original(module, "forward", original_forward)
        hook = self

        def wrapped_forward(attn_self, x, context=None, indices=None):
            if not hook._is_active(stage_name, "self"):
                return original_forward(x, context, indices)

            batch_size, num_queries, _ = x.shape
            qkv = attn_self.to_qkv(x).reshape(batch_size, num_queries, 3, attn_self.num_heads, -1)
            q, k, v = qkv.unbind(dim=2)
            if attn_self.use_rope:
                q, k = attn_self.rope(q, k, indices)
            if attn_self.qk_rms_norm:
                q = attn_self.q_rms_norm(q)
                k = attn_self.k_rms_norm(k)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            k, v = hook._apply_dense_kv(stage_name, layer_idx, "self", k, v)
            h = F.scaled_dot_product_attention(q, k, v)
            h = h.permute(0, 2, 1, 3).reshape(batch_size, num_queries, -1)
            return attn_self.to_out(h)

        module.forward = types.MethodType(wrapped_forward, module)

    def _patch_dense_cross(self, stage_name: str, module, layer_idx: int) -> None:
        original_forward = module.forward
        self._save_original(module, "forward", original_forward)
        hook = self

        def wrapped_forward(attn_self, x, context=None, indices=None):
            if not hook._is_active(stage_name, "cross") or context is None:
                return original_forward(x, context, indices)

            batch_size, num_queries, _ = x.shape
            q = attn_self.to_q(x).reshape(batch_size, num_queries, attn_self.num_heads, -1)
            kv = attn_self.to_kv(context).reshape(batch_size, context.shape[1], 2, attn_self.num_heads, -1)
            k, v = kv.unbind(dim=2)
            if attn_self.qk_rms_norm:
                q = attn_self.q_rms_norm(q)
                k = attn_self.k_rms_norm(k)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            k, v = hook._apply_dense_kv(stage_name, layer_idx, "cross", k, v)
            h = F.scaled_dot_product_attention(q, k, v)
            h = h.permute(0, 2, 1, 3).reshape(batch_size, num_queries, -1)
            return attn_self.to_out(h)

        module.forward = types.MethodType(wrapped_forward, module)

    def _patch_sparse_self(self, stage_name: str, module, layer_idx: int) -> None:
        original_forward = module.forward
        self._save_original(module, "forward", original_forward)
        hook = self

        def wrapped_forward(attn_self, x, context=None):
            if not hook._is_active(stage_name, "self"):
                return original_forward(x, context)

            qkv = attn_self._linear(attn_self.to_qkv, x)
            qkv = attn_self._fused_pre(qkv, num_fused=3)
            if attn_self.use_rope:
                qkv = attn_self._rope(qkv)
            q, k, v = qkv.unbind(dim=1)
            if attn_self.qk_rms_norm:
                q = attn_self.q_rms_norm(q)
                k = attn_self.k_rms_norm(k)

            ctx = hook.current_forward_context
            assert ctx is not None
            if ctx["phase"] == "source_capture":
                snapshot = hook._pending_step_source_snapshots.setdefault(stage_name, {})
                key = hook._snapshot_key(ctx["pass_kind"], int(ctx["eval_idx"]), layer_idx, "self")
                snapshot[f"{key}|k_sparse"] = k.cpu()
                snapshot[f"{key}|v_sparse"] = v.cpu()
            elif ctx["phase"] == "denoise":
                sparse_mask = hook._sparse_self_mask(stage_name, x.coords.device)
                cached_k = None
                cached_v = None
                active_t = hook._active_step_source_times.get(stage_name)
                active_snapshot = hook._active_step_source_snapshots.get(stage_name)
                if active_t is not None and abs(active_t - float(ctx["logical_t"])) <= 1e-6 and active_snapshot is not None:
                    cache_key = hook._snapshot_key(ctx["pass_kind"], int(ctx["eval_idx"]), layer_idx, "self")
                    cached_k = active_snapshot.get(f"{cache_key}|k_sparse")
                    cached_v = active_snapshot.get(f"{cache_key}|v_sparse")
                preserve_coords, edit_weights = hook._resolve_sparse_self_mask(sparse_mask)
                if preserve_coords is not None and cached_k is not None and cached_v is not None:
                    k = hook._blend_sparse_tensor(k, cached_k, preserve_coords, edit_weights)
                    v = hook._blend_sparse_tensor(v, cached_v, preserve_coords, edit_weights)

            h = q
            q_feats = q.feats.unsqueeze(0).permute(0, 2, 1, 3)
            k_feats = k.feats.unsqueeze(0).permute(0, 2, 1, 3)
            v_feats = v.feats.unsqueeze(0).permute(0, 2, 1, 3)
            out = F.scaled_dot_product_attention(q_feats, k_feats, v_feats)
            out = out.permute(0, 2, 1, 3)[0]
            h = h.replace(out)
            h = attn_self._reshape_chs(h, (-1,))
            return attn_self._linear(attn_self.to_out, h)

        module.forward = types.MethodType(wrapped_forward, module)

    def _patch_sparse_cross(self, stage_name: str, module, layer_idx: int) -> None:
        original_forward = module.forward
        self._save_original(module, "forward", original_forward)
        hook = self

        def wrapped_forward(attn_self, x, context=None):
            if not hook._is_active(stage_name, "cross") or context is None:
                return original_forward(x, context)

            q = attn_self._linear(attn_self.to_q, x)
            q = attn_self._reshape_chs(q, (attn_self.num_heads, -1))
            kv = attn_self._linear(attn_self.to_kv, context)
            kv = attn_self._fused_pre(kv, num_fused=2)
            k, v = kv.unbind(dim=2)
            h = q
            q_feats = q.feats.unsqueeze(0).permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            k, v = hook._apply_dense_kv(stage_name, layer_idx, "cross", k, v)
            out = F.scaled_dot_product_attention(q_feats, k, v)
            out = out.permute(0, 2, 1, 3)[0]
            h = h.replace(out)
            h = attn_self._reshape_chs(h, (-1,))
            return attn_self._linear(attn_self.to_out, h)

        module.forward = types.MethodType(wrapped_forward, module)

    def _infer_pass_kind(self, cond: torch.Tensor) -> str:
        sig = tensor_signature(cond)
        if signature_distance(sig, self.neg_cond_signature) <= 1e-6:
            return "neg"
        return "cond"
