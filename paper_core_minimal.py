"""
Minimal paper-core extraction for TRELLIS.

This file compresses the main paper implementation into one readable script.

Corresponds mainly to these original modules:
- trellis/pipelines/trellis_image_to_3d.py
- trellis/pipelines/trellis_text_to_3d.py
- trellis/pipelines/samplers/flow_euler.py
- trellis/models/sparse_structure_flow.py
- trellis/models/sparse_structure_vae.py
- trellis/models/structured_latent_flow.py
- trellis/models/structured_latent_vae/encoder.py
- trellis/models/structured_latent_vae/decoder_gs.py
- trellis/models/structured_latent_vae/decoder_rf.py
- trellis/models/structured_latent_vae/decoder_mesh.py
- trellis/modules/transformer/modulated.py
- trellis/modules/sparse/transformer/modulated.py

What is preserved:
- TRELLIS's core two-stage generation path:
  condition tokens -> sparse structure flow -> occupancy decode ->
  sparse structured latent (SLAT) flow -> decode to multiple 3D formats
- the dense and sparse rectified-flow backbones
- the main VAE-style latent modules used to define the sparse structure and SLAT spaces
- the Euler flow sampler and the flow-matching training target
- original class and method names whenever possible

What is omitted:
- pretrained loading, Hugging Face glue, config parsing, CLI, logging
- rembg / DINOv2 / CLIP implementation details
- backend-specific sparse kernels from spconv / torchsparse
- rendering, export, dataset loading, distributed training, EMA, checkpoints
- the real FlexiCubes mesh extraction implementation

Why this file may not run directly:
- sparse operators are intentionally reduced to readable placeholders
- image/text condition encoders accept precomputed tokens or TODO hooks
- representation builders keep the original data flow but not the full runtime stack
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def patchify_3d(x: Tensor, patch_size: int) -> Tensor:
    """Readable copy of trellis.modules.spatial.patchify for 3D tensors."""
    b, c, d, h, w = x.shape
    assert d % patch_size == 0 and h % patch_size == 0 and w % patch_size == 0
    x = x.reshape(
        b,
        c,
        d // patch_size,
        patch_size,
        h // patch_size,
        patch_size,
        w // patch_size,
        patch_size,
    )
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
    x = x.reshape(b, c * patch_size**3, d // patch_size, h // patch_size, w // patch_size)
    return x


def unpatchify_3d(x: Tensor, patch_size: int) -> Tensor:
    """Readable copy of trellis.modules.spatial.unpatchify for 3D tensors."""
    b, c, d, h, w = x.shape
    assert c % patch_size**3 == 0
    out_c = c // patch_size**3
    x = x.reshape(b, out_c, patch_size, patch_size, patch_size, d, h, w)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    x = x.reshape(b, out_c, d * patch_size, h * patch_size, w * patch_size)
    return x


def pixel_shuffle_3d(x: Tensor, scale_factor: int) -> Tensor:
    """Readable copy of trellis.modules.spatial.pixel_shuffle_3d."""
    b, c, d, h, w = x.shape
    out_c = c // scale_factor**3
    x = x.reshape(b, out_c, scale_factor, scale_factor, scale_factor, d, h, w)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    x = x.reshape(b, out_c, d * scale_factor, h * scale_factor, w * scale_factor)
    return x


def repeat_cond_if_needed(cond: Tensor, batch_size: int) -> Tensor:
    if cond.shape[0] == batch_size:
        return cond
    if cond.shape[0] == 1:
        return cond.repeat(batch_size, *([1] * (cond.ndim - 1)))
    raise ValueError(f"Cannot broadcast cond of shape {tuple(cond.shape)} to batch size {batch_size}")


def dense_positional_grid(size: int, device: torch.device) -> Tensor:
    axes = torch.arange(size, device=device, dtype=torch.float32)
    grid = torch.meshgrid(axes, axes, axes, indexing="ij")
    return torch.stack(grid, dim=-1).reshape(-1, 3)


def sample_logit_normal_t(batch_size: int, mean: float = 0.0, std: float = 1.0, device: Optional[torch.device] = None) -> Tensor:
    return torch.sigmoid(torch.randn(batch_size, device=device) * std + mean)


def mse_between(pred: Union[Tensor, "SparseTensor"], target: Union[Tensor, "SparseTensor"]) -> Tensor:
    if isinstance(pred, SparseTensor):
        return F.mse_loss(pred.feats, target.feats)
    return F.mse_loss(pred, target)


# -----------------------------------------------------------------------------
# Minimal sparse tensor abstraction
# -----------------------------------------------------------------------------


@dataclass
class SparseTensor:
    """
    Readable stand-in for trellis.modules.sparse.basic.SparseTensor.

    The real repo supports spconv / torchsparse backends. Here we only keep:
    - per-voxel features
    - integer coordinates [batch_idx, x, y, z]
    - enough helpers to make the paper data flow explicit
    """

    feats: Tensor
    coords: Tensor

    def __post_init__(self) -> None:
        if self.coords.ndim != 2 or self.coords.shape[1] != 4:
            raise ValueError("coords must have shape [N, 4] with [batch_idx, x, y, z]")
        if self.feats.shape[0] != self.coords.shape[0]:
            raise ValueError("feats and coords must share the first dimension")
        order = torch.argsort(self.coords[:, 0])
        self.feats = self.feats[order]
        self.coords = self.coords[order].int()

    @property
    def batch_size(self) -> int:
        return int(self.coords[:, 0].max().item()) + 1 if self.coords.numel() else 0

    @property
    def shape(self) -> torch.Size:
        return torch.Size([self.batch_size, self.feats.shape[-1]])

    @property
    def device(self) -> torch.device:
        return self.feats.device

    @property
    def dtype(self) -> torch.dtype:
        return self.feats.dtype

    @property
    def layout(self) -> List[slice]:
        if self.coords.numel() == 0:
            return []
        counts = torch.bincount(self.coords[:, 0], minlength=self.batch_size)
        offsets = torch.cumsum(counts, dim=0)
        starts = offsets - counts
        return [slice(int(starts[i]), int(offsets[i])) for i in range(self.batch_size)]

    def replace(self, feats: Tensor, coords: Optional[Tensor] = None) -> "SparseTensor":
        return SparseTensor(feats=feats, coords=self.coords if coords is None else coords)

    def type(self, dtype: torch.dtype) -> "SparseTensor":
        return self.replace(self.feats.type(dtype))

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> "SparseTensor":
        return SparseTensor(
            feats=self.feats.to(device=device, dtype=dtype),
            coords=self.coords.to(device=device),
        )

    def float(self) -> "SparseTensor":
        return self.type(torch.float32)

    def half(self) -> "SparseTensor":
        return self.type(torch.float16)

    def __getitem__(self, batch_index: int) -> "SparseTensor":
        sl = self.layout[batch_index]
        coords = self.coords[sl].clone()
        coords[:, 0] = 0
        return SparseTensor(self.feats[sl], coords)

    def _binary(self, other: Union["SparseTensor", Tensor, float, int], op) -> "SparseTensor":
        other_value = other.feats if isinstance(other, SparseTensor) else other
        return self.replace(op(self.feats, other_value))

    def __add__(self, other: Union["SparseTensor", Tensor, float, int]) -> "SparseTensor":
        return self._binary(other, torch.add)

    def __radd__(self, other: Union[Tensor, float, int]) -> "SparseTensor":
        return self.__add__(other)

    def __sub__(self, other: Union["SparseTensor", Tensor, float, int]) -> "SparseTensor":
        return self._binary(other, torch.sub)

    def __rsub__(self, other: Union[Tensor, float, int]) -> "SparseTensor":
        return self.replace(torch.sub(other, self.feats))

    def __mul__(self, other: Union["SparseTensor", Tensor, float, int]) -> "SparseTensor":
        return self._binary(other, torch.mul)

    def __rmul__(self, other: Union[Tensor, float, int]) -> "SparseTensor":
        return self.__mul__(other)

    def __truediv__(self, other: Union[Tensor, float, int]) -> "SparseTensor":
        return self.replace(torch.div(self.feats, other))


def sparse_batch_broadcast(x: SparseTensor, values: Tensor) -> Tensor:
    """
    Broadcast a [B, C] tensor to each occupied sparse voxel in the batch.
    """
    chunks: List[Tensor] = []
    for batch_index, sl in enumerate(x.layout):
        count = sl.stop - sl.start
        chunks.append(values[batch_index : batch_index + 1].expand(count, -1))
    return torch.cat(chunks, dim=0) if chunks else values.new_zeros((0, values.shape[-1]))


def coalesce_sparse_tensor(x: SparseTensor) -> SparseTensor:
    """
    Merge duplicate coords after naive downsampling.
    """
    if x.coords.numel() == 0:
        return x
    uniq, inverse = torch.unique(x.coords, dim=0, return_inverse=True, sorted=True)
    feats = torch.zeros(uniq.shape[0], x.feats.shape[-1], device=x.device, dtype=x.dtype)
    feats.index_add_(0, inverse, x.feats)
    counts = torch.bincount(inverse, minlength=uniq.shape[0]).clamp_min(1).to(feats.dtype)
    feats = feats / counts[:, None]
    return SparseTensor(feats=feats, coords=uniq)


# -----------------------------------------------------------------------------
# Readable sparse operator stand-ins
# -----------------------------------------------------------------------------


class SparseLinear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x: SparseTensor) -> SparseTensor:
        return x.replace(self.linear(x.feats))


class SparseConv3d(nn.Module):
    """
    Placeholder for backend-specific sparse 3D convolution.

    The real repo uses neighborhood-aware sparse kernels. For readability we
    keep only a per-voxel projection so that the main data flow stays visible.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.proj = nn.Linear(in_channels, out_channels)

    def forward(self, x: SparseTensor) -> SparseTensor:
        return x.replace(self.proj(x.feats))


class SparseDownsample(nn.Module):
    def __init__(self, factor: int):
        super().__init__()
        self.factor = factor

    def forward(self, x: SparseTensor) -> SparseTensor:
        coords = x.coords.clone()
        coords[:, 1:] = torch.div(coords[:, 1:], self.factor, rounding_mode="floor")
        return coalesce_sparse_tensor(SparseTensor(x.feats, coords))


class SparseUpsample(nn.Module):
    def __init__(self, factor: int):
        super().__init__()
        self.factor = factor

    def forward(self, x: SparseTensor) -> SparseTensor:
        coords = x.coords.clone()
        coords[:, 1:] = coords[:, 1:] * self.factor
        return SparseTensor(x.feats, coords)


class SparseSubdivide(nn.Module):
    """
    Placeholder for trellis.modules.sparse.SparseSubdivide.

    Real TRELLIS duplicates each occupied voxel into child voxels before mesh
    extraction. Here we keep the semantic operation explicit.
    """

    def forward(self, x: SparseTensor) -> SparseTensor:
        child_offsets = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ],
            device=x.device,
            dtype=x.coords.dtype,
        )
        repeats = child_offsets.shape[0]
        coords = x.coords.repeat_interleave(repeats, dim=0)
        coords[:, 1:] = coords[:, 1:] * 2 + child_offsets.repeat(x.coords.shape[0], 1)
        feats = x.feats.repeat_interleave(repeats, dim=0)
        return SparseTensor(feats, coords)


# -----------------------------------------------------------------------------
# Attention and transformer blocks
# -----------------------------------------------------------------------------


class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: Tensor) -> Tensor:
        return (F.normalize(x.float(), dim=-1) * self.gamma * self.scale).to(x.dtype)


class AbsolutePositionEmbedder(nn.Module):
    """
    Minimal readable copy of trellis.modules.transformer.AbsolutePositionEmbedder.
    """

    def __init__(self, channels: int, in_channels: int = 3):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.freq_dim = max(1, channels // in_channels // 2)
        freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.register_buffer("freqs", 1.0 / (10000**freqs), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        n, d = x.shape
        if d != self.in_channels:
            raise ValueError(f"Expected {self.in_channels}D coords, got {d}")
        out = torch.outer(x.reshape(-1), self.freqs.to(x.device))
        out = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
        out = out.reshape(n, -1)
        if out.shape[1] < self.channels:
            pad = out.new_zeros(n, self.channels - out.shape[1])
            out = torch.cat([out, pad], dim=-1)
        return out[:, : self.channels]


class TimestepEmbedder(nn.Module):
    """
    Directly adapted from trellis.models.sparse_structure_flow.TimestepEmbedder.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class FeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SparseFeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.net = nn.Sequential(
            SparseLinear(channels, hidden),
            nn.GELU(approximate="tanh"),
            SparseLinear(hidden, channels),
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        h: Union[SparseTensor, Tensor] = x
        for layer in self.net:
            if isinstance(h, SparseTensor):
                if isinstance(layer, nn.GELU):
                    h = h.replace(layer(h.feats))
                else:
                    h = layer(h)
            else:
                raise TypeError("SparseFeedForwardNet only accepts SparseTensor")
        return h


class MultiHeadAttention(nn.Module):
    """
    A readable approximation of TRELLIS attention.

    Omitted on purpose:
    - windowed attention
    - serialized sparse attention
    - rotary embeddings
    The paper-critical part is still visible: self-attention and cross-attention
    between 3D tokens and condition tokens.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: Literal["self", "cross"] = "self",
        qk_rms_norm: bool = False,
        **_: Any,
    ):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.ctx_channels = channels if ctx_channels is None else ctx_channels
        self.type = type
        self.qk_rms_norm = qk_rms_norm
        if type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3)
        else:
            self.to_q = nn.Linear(channels, channels)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2)
        if qk_rms_norm:
            self.q_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.to_out = nn.Linear(channels, channels)

    def _reshape_heads(self, x: Tensor) -> Tensor:
        b, n, c = x.shape
        return x.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        b, h, n, d = x.shape
        return x.transpose(1, 2).reshape(b, n, h * d)

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        if self.type == "self":
            q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        else:
            if context is None:
                raise ValueError("Cross-attention requires context")
            q = self.to_q(x)
            k, v = self.to_kv(context).chunk(2, dim=-1)
        q = self._reshape_heads(q)
        k = self._reshape_heads(k)
        v = self._reshape_heads(v)
        if self.qk_rms_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        h = F.scaled_dot_product_attention(q, k, v)
        h = self._merge_heads(h)
        return self.to_out(h)


class SparseMultiHeadAttention(nn.Module):
    """
    Readable sparse attention wrapper.

    The real implementation has specialized sparse kernels and windowing modes.
    This minimal version simply applies dense attention per batch over occupied
    sparse tokens, which is enough to expose the SLAT data flow.
    """

    def __init__(self, channels: int, num_heads: int, ctx_channels: Optional[int] = None, type: Literal["self", "cross"] = "self", qk_rms_norm: bool = False, **kwargs: Any):
        super().__init__()
        self.inner = MultiHeadAttention(
            channels=channels,
            num_heads=num_heads,
            ctx_channels=ctx_channels,
            type=type,
            qk_rms_norm=qk_rms_norm,
            **kwargs,
        )
        self.type = type

    def forward(self, x: SparseTensor, context: Optional[Tensor] = None) -> SparseTensor:
        outputs: List[Tensor] = []
        if self.type == "self":
            for sl in x.layout:
                outputs.append(self.inner(x.feats[sl][None])[0])
        else:
            if context is None:
                raise ValueError("Cross-attention requires context")
            context = repeat_cond_if_needed(context, x.batch_size)
            for batch_index, sl in enumerate(x.layout):
                outputs.append(self.inner(x.feats[sl][None], context[batch_index : batch_index + 1])[0])
        if not outputs:
            return x.replace(x.feats.new_zeros((0, x.feats.shape[-1])))
        return x.replace(torch.cat(outputs, dim=0))


class SparseTransformerBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int, mlp_ratio: float = 4.0, qk_rms_norm: bool = False, **kwargs: Any):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(channels, elementwise_affine=False)
        self.attn = SparseMultiHeadAttention(channels, num_heads, type="self", qk_rms_norm=qk_rms_norm, **kwargs)
        self.mlp = SparseFeedForwardNet(channels, mlp_ratio=mlp_ratio)

    def forward(self, x: SparseTensor) -> SparseTensor:
        h = x.replace(self.norm1(x.feats))
        x = x + self.attn(h)
        h = x.replace(self.norm2(x.feats))
        x = x + self.mlp(h)
        return x


class ModulatedTransformerCrossBlock(nn.Module):
    """
    Dense rectified-flow block used by SparseStructureFlowModel.

    Paper role:
    - self-attend over 3D dense tokens
    - cross-attend to image/text condition tokens
    - modulate normalization with timestep embedding via AdaLN
    """

    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        self.share_mod = share_mod
        self.norm1 = nn.LayerNorm(channels, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(channels, elementwise_affine=True)
        self.norm3 = nn.LayerNorm(channels, elementwise_affine=False)
        self.self_attn = MultiHeadAttention(channels, num_heads, type="self", qk_rms_norm=qk_rms_norm, **kwargs)
        self.cross_attn = MultiHeadAttention(channels, num_heads, ctx_channels=ctx_channels, type="cross", qk_rms_norm=qk_rms_norm_cross)
        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio)
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 6 * channels))

    def forward(self, x: Tensor, mod: Tensor, context: Tensor) -> Tensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale_msa[:, None]) + shift_msa[:, None]
        x = x + self.self_attn(h) * gate_msa[:, None]
        h = self.norm2(x)
        x = x + self.cross_attn(h, context)
        h = self.norm3(x)
        h = h * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        x = x + self.mlp(h) * gate_mlp[:, None]
        return x


class ModulatedSparseTransformerCrossBlock(nn.Module):
    """
    Sparse rectified-flow block used by SLatFlowModel.

    Paper role:
    - self-attend only over occupied sparse voxels
    - cross-attend to the same condition tokens as stage 1
    - inject timestep through AdaLN modulation
    """

    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        self.share_mod = share_mod
        self.norm1 = nn.LayerNorm(channels, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(channels, elementwise_affine=True)
        self.norm3 = nn.LayerNorm(channels, elementwise_affine=False)
        self.self_attn = SparseMultiHeadAttention(channels, num_heads, type="self", qk_rms_norm=qk_rms_norm, **kwargs)
        self.cross_attn = SparseMultiHeadAttention(channels, num_heads, ctx_channels=ctx_channels, type="cross", qk_rms_norm=qk_rms_norm_cross)
        self.mlp = SparseFeedForwardNet(channels, mlp_ratio=mlp_ratio)
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 6 * channels))

    def forward(self, x: SparseTensor, mod: Tensor, context: Tensor) -> SparseTensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        shift_msa = sparse_batch_broadcast(x, shift_msa)
        scale_msa = sparse_batch_broadcast(x, scale_msa)
        gate_msa = sparse_batch_broadcast(x, gate_msa)
        shift_mlp = sparse_batch_broadcast(x, shift_mlp)
        scale_mlp = sparse_batch_broadcast(x, scale_mlp)
        gate_mlp = sparse_batch_broadcast(x, gate_mlp)
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        x = x + self.self_attn(h) * gate_msa
        h = x.replace(self.norm2(x.feats))
        x = x + self.cross_attn(h, context)
        h = x.replace(self.norm3(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        x = x + self.mlp(h) * gate_mlp
        return x


# -----------------------------------------------------------------------------
# Stage 1: sparse structure VAE and dense flow
# -----------------------------------------------------------------------------


class ResBlock3d(nn.Module):
    def __init__(self, channels: int, out_channels: Optional[int] = None):
        super().__init__()
        out_channels = channels if out_channels is None else out_channels
        self.norm1 = nn.GroupNorm(32, channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv1 = nn.Conv3d(channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Identity() if channels == out_channels else nn.Conv3d(channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class DownsampleBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class UpsampleBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels * 8, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return pixel_shuffle_3d(self.conv(x), scale_factor=2)


class SparseStructureEncoder(nn.Module):
    """
    VAE encoder for the occupancy latent z_s.

    It appears in training rather than inference, but it is part of the paper's
    core factorization: structure is learned in a compact dense 3D latent.
    """

    def __init__(self, in_channels: int, latent_channels: int, channels: List[int], num_res_blocks: int = 2):
        super().__init__()
        self.input_layer = nn.Conv3d(in_channels, channels[0], kernel_size=3, padding=1)
        blocks: List[nn.Module] = []
        for i, ch in enumerate(channels):
            for _ in range(num_res_blocks):
                blocks.append(ResBlock3d(ch, ch))
            if i < len(channels) - 1:
                blocks.append(DownsampleBlock3d(ch, channels[i + 1]))
        self.blocks = nn.ModuleList(blocks)
        self.middle = nn.Sequential(ResBlock3d(channels[-1]), ResBlock3d(channels[-1]))
        self.out_layer = nn.Sequential(
            nn.GroupNorm(32, channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], latent_channels * 2, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor, sample_posterior: bool = False, return_raw: bool = False):
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        h = self.middle(h)
        mean, logvar = self.out_layer(h).chunk(2, dim=1)
        z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean) if sample_posterior else mean
        return (z, mean, logvar) if return_raw else z


class SparseStructureDecoder(nn.Module):
    """
    Decoder D_S from the paper: dense latent z_s -> occupancy logits.
    """

    def __init__(self, out_channels: int, latent_channels: int, channels: List[int], num_res_blocks: int = 2):
        super().__init__()
        self.input_layer = nn.Conv3d(latent_channels, channels[0], kernel_size=3, padding=1)
        self.middle = nn.Sequential(ResBlock3d(channels[0]), ResBlock3d(channels[0]))
        blocks: List[nn.Module] = []
        for i, ch in enumerate(channels):
            for _ in range(num_res_blocks):
                blocks.append(ResBlock3d(ch, ch))
            if i < len(channels) - 1:
                blocks.append(UpsampleBlock3d(ch, channels[i + 1]))
        self.blocks = nn.ModuleList(blocks)
        self.out_layer = nn.Sequential(
            nn.GroupNorm(32, channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.middle(self.input_layer(x))
        for block in self.blocks:
            h = block(h)
        return self.out_layer(h)


class SparseStructureFlowModel(nn.Module):
    """
    Dense rectified-flow transformer for stage 1.

    Input:
    - dense Gaussian noise in the occupancy latent space z_s
    - timestep t
    - image/text condition tokens

    Output:
    - velocity field v_theta(x_t, t, cond) in the same latent space
    """

    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        patch_size: int = 2,
        num_head_channels: int = 64,
        mlp_ratio: float = 4.0,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.num_heads = model_channels // num_head_channels
        self.t_embedder = TimestepEmbedder(model_channels)
        self.pos_embedder = AbsolutePositionEmbedder(model_channels)
        self.input_layer = nn.Linear(in_channels * patch_size**3, model_channels)
        self.blocks = nn.ModuleList(
            [
                ModulatedTransformerCrossBlock(
                    channels=model_channels,
                    ctx_channels=cond_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                    share_mod=share_mod,
                    qk_rms_norm=qk_rms_norm,
                    qk_rms_norm_cross=qk_rms_norm_cross,
                )
                for _ in range(num_blocks)
            ]
        )
        self.out_layer = nn.Linear(model_channels, out_channels * patch_size**3)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 6 * model_channels))
        self.share_mod = share_mod

    def forward(self, x: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        b = x.shape[0]
        cond = repeat_cond_if_needed(cond, b)
        h = patchify_3d(x, self.patch_size)
        h = h.view(b, h.shape[1], -1).permute(0, 2, 1).contiguous()
        h = self.input_layer(h)
        pos = dense_positional_grid(self.resolution // self.patch_size, device=x.device)
        h = h + self.pos_embedder(pos)[None]
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        for block in self.blocks:
            h = block(h, t_emb, cond)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)
        h = h.permute(0, 2, 1).reshape(
            b,
            self.out_channels * self.patch_size**3,
            self.resolution // self.patch_size,
            self.resolution // self.patch_size,
            self.resolution // self.patch_size,
        )
        return unpatchify_3d(h, self.patch_size)


# -----------------------------------------------------------------------------
# Stage 2: structured latent (SLAT) VAE and sparse flow
# -----------------------------------------------------------------------------


class SparseResBlock3d(nn.Module):
    """
    Sparse IO block from trellis.models.structured_latent_flow.

    In the real repo this mixes sparse 3D convolution with timestep modulation.
    Here we keep the original call pattern and tensor flow.
    """

    def __init__(
        self,
        channels: int,
        emb_channels: int,
        out_channels: Optional[int] = None,
        downsample: bool = False,
        upsample: bool = False,
    ):
        super().__init__()
        out_channels = channels if out_channels is None else out_channels
        self.norm1 = nn.LayerNorm(channels, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(out_channels, elementwise_affine=False)
        self.conv1 = SparseConv3d(channels, out_channels, kernel_size=3)
        self.conv2 = SparseConv3d(out_channels, out_channels, kernel_size=3)
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, 2 * out_channels))
        self.skip = nn.Identity() if channels == out_channels else SparseLinear(channels, out_channels)
        self.updown: Optional[nn.Module] = None
        if downsample:
            self.updown = SparseDownsample(2)
        if upsample:
            self.updown = SparseUpsample(2)

    def forward(self, x: SparseTensor, emb: Tensor) -> SparseTensor:
        x = self.updown(x) if self.updown is not None else x
        scale, shift = self.emb_layers(emb).chunk(2, dim=1)
        scale = sparse_batch_broadcast(x, scale)
        shift = sparse_batch_broadcast(x, shift)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv1(h)
        h = h.replace(self.norm2(h.feats) * (1 + scale) + shift)
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        return h + self.skip(x)


class SparseTransformerBase(nn.Module):
    """
    Shared sparse transformer torso used by SLatEncoder and the decoders.
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        num_blocks: int,
        num_head_channels: int = 64,
        mlp_ratio: float = 4.0,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        self.model_channels = model_channels
        self.num_heads = model_channels // num_head_channels
        self.pos_embedder = AbsolutePositionEmbedder(model_channels)
        self.input_layer = SparseLinear(in_channels, model_channels)
        self.blocks = nn.ModuleList(
            [
                SparseTransformerBlock(
                    channels=model_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                    qk_rms_norm=qk_rms_norm,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        h = self.input_layer(x)
        h = h + self.pos_embedder(h.coords[:, 1:].float())
        for block in self.blocks:
            h = block(h)
        return h


class SLatEncoder(SparseTransformerBase):
    """
    Sparse VAE encoder for the structured latent z_l (SLAT).
    """

    def __init__(self, in_channels: int, model_channels: int, latent_channels: int, num_blocks: int, **kwargs: Any):
        super().__init__(in_channels=in_channels, model_channels=model_channels, num_blocks=num_blocks, **kwargs)
        self.out_layer = SparseLinear(model_channels, 2 * latent_channels)

    def forward(self, x: SparseTensor, sample_posterior: bool = True, return_raw: bool = False):
        h = super().forward(x)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        mean, logvar = h.feats.chunk(2, dim=-1)
        z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean) if sample_posterior else mean
        z = h.replace(z)
        return (z, mean, logvar) if return_raw else z


@dataclass
class Gaussian:
    _xyz: Tensor
    _features_dc: Tensor
    _scaling: Tensor
    _rotation: Tensor
    _opacity: Tensor


@dataclass
class Strivec:
    position: Tensor
    trivec: Tensor
    density: Tensor
    features_dc: Tensor


@dataclass
class MeshExtractResult:
    voxel_coords: Tensor
    voxel_features: Tensor
    notes: str


class SparseFeatures2Mesh(nn.Module):
    """
    Stand-in for trellis.representations.mesh.SparseFeatures2Mesh.

    Real TRELLIS uses a FlexiCubes-based extractor to turn sparse voxel-aligned
    features into an explicit mesh. Here we keep only the semantic contract.
    """

    def __init__(self, res: int, use_color: bool = False):
        super().__init__()
        self.res = res
        self.use_color = use_color
        self.feats_channels = 32 if use_color else 16

    def forward(self, x: SparseTensor, training: bool = False) -> MeshExtractResult:
        return MeshExtractResult(
            voxel_coords=x.coords[:, 1:].float() / float(self.res),
            voxel_features=x.feats,
            notes="Placeholder for FlexiCubes mesh extraction. See trellis/representations/mesh/.",
        )


class SLatGaussianDecoder(SparseTransformerBase):
    """
    Decode each occupied sparse voxel into a small set of local Gaussians.
    """

    def __init__(self, resolution: int, model_channels: int, latent_channels: int, num_blocks: int, representation_config: Optional[Dict[str, Any]] = None, **kwargs: Any):
        super().__init__(in_channels=latent_channels, model_channels=model_channels, num_blocks=num_blocks, **kwargs)
        self.resolution = resolution
        self.rep_config = representation_config or {
            "num_gaussians": 4,
            "voxel_size": 1.0,
        }
        self._calc_layout()
        self.out_layer = SparseLinear(model_channels, self.out_channels)

    def _calc_layout(self) -> None:
        num_gaussians = self.rep_config["num_gaussians"]
        self.layout = {
            "_xyz": {"shape": (num_gaussians, 3), "size": num_gaussians * 3},
            "_features_dc": {"shape": (num_gaussians, 1, 3), "size": num_gaussians * 3},
            "_scaling": {"shape": (num_gaussians, 3), "size": num_gaussians * 3},
            "_rotation": {"shape": (num_gaussians, 4), "size": num_gaussians * 4},
            "_opacity": {"shape": (num_gaussians, 1), "size": num_gaussians},
        }
        start = 0
        for value in self.layout.values():
            value["range"] = (start, start + value["size"])
            start += value["size"]
        self.out_channels = start

    def to_representation(self, x: SparseTensor) -> List[Gaussian]:
        outputs: List[Gaussian] = []
        for sl in x.layout:
            coords = x.coords[sl][:, 1:].float()
            base_xyz = (coords + 0.5) / self.resolution
            feats = x.feats[sl]
            pieces: Dict[str, Tensor] = {}
            for name, spec in self.layout.items():
                start, end = spec["range"]
                pieces[name] = feats[:, start:end].reshape(-1, *spec["shape"])
            outputs.append(
                Gaussian(
                    _xyz=base_xyz[:, None, :] + torch.tanh(pieces["_xyz"]) / self.resolution,
                    _features_dc=pieces["_features_dc"],
                    _scaling=pieces["_scaling"],
                    _rotation=pieces["_rotation"],
                    _opacity=pieces["_opacity"],
                )
            )
        return outputs

    def forward(self, x: SparseTensor) -> List[Gaussian]:
        h = super().forward(x)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        return self.to_representation(h)


class SLatRadianceFieldDecoder(SparseTransformerBase):
    """
    Decode each occupied sparse voxel into a local tri-vector radiance field.
    """

    def __init__(self, resolution: int, model_channels: int, latent_channels: int, num_blocks: int, representation_config: Optional[Dict[str, Any]] = None, **kwargs: Any):
        super().__init__(in_channels=latent_channels, model_channels=model_channels, num_blocks=num_blocks, **kwargs)
        self.resolution = resolution
        self.rep_config = representation_config or {
            "rank": 8,
            "dim": 8,
        }
        self._calc_layout()
        self.out_layer = SparseLinear(model_channels, self.out_channels)

    def _calc_layout(self) -> None:
        rank = self.rep_config["rank"]
        dim = self.rep_config["dim"]
        self.layout = {
            "trivec": {"shape": (rank, 3, dim), "size": rank * 3 * dim},
            "density": {"shape": (rank,), "size": rank},
            "features_dc": {"shape": (rank, 1, 3), "size": rank * 3},
        }
        start = 0
        for value in self.layout.values():
            value["range"] = (start, start + value["size"])
            start += value["size"]
        self.out_channels = start

    def to_representation(self, x: SparseTensor) -> List[Strivec]:
        outputs: List[Strivec] = []
        for sl in x.layout:
            coords = (x.coords[sl][:, 1:].float() + 0.5) / self.resolution
            feats = x.feats[sl]
            fields: Dict[str, Tensor] = {}
            for name, spec in self.layout.items():
                start, end = spec["range"]
                fields[name] = feats[:, start:end].reshape(-1, *spec["shape"])
            outputs.append(
                Strivec(
                    position=coords,
                    trivec=fields["trivec"],
                    density=fields["density"],
                    features_dc=fields["features_dc"],
                )
            )
        return outputs

    def forward(self, x: SparseTensor) -> List[Strivec]:
        h = super().forward(x)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        return self.to_representation(h)


class SparseSubdivideBlock3d(nn.Module):
    def __init__(self, channels: int, out_channels: Optional[int] = None):
        super().__init__()
        out_channels = channels if out_channels is None else out_channels
        self.norm = nn.LayerNorm(channels, elementwise_affine=True)
        self.subdivide = SparseSubdivide()
        self.conv1 = SparseConv3d(channels, out_channels, kernel_size=3)
        self.conv2 = SparseConv3d(out_channels, out_channels, kernel_size=3)
        self.skip = nn.Identity() if channels == out_channels else SparseConv3d(channels, out_channels, kernel_size=1)

    def forward(self, x: SparseTensor) -> SparseTensor:
        h = x.replace(F.silu(self.norm(x.feats)))
        h = self.subdivide(h)
        x = self.subdivide(x)
        h = self.conv2(self.conv1(h))
        return h + self.skip(x)


class SLatMeshDecoder(SparseTransformerBase):
    """
    Decode sparse SLAT features into mesh-ready sparse features, then extract a mesh.
    """

    def __init__(self, resolution: int, model_channels: int, latent_channels: int, num_blocks: int, representation_config: Optional[Dict[str, Any]] = None, **kwargs: Any):
        super().__init__(in_channels=latent_channels, model_channels=model_channels, num_blocks=num_blocks, **kwargs)
        self.resolution = resolution
        self.rep_config = representation_config or {"use_color": False}
        self.mesh_extractor = SparseFeatures2Mesh(res=resolution * 4, use_color=self.rep_config.get("use_color", False))
        self.upsample = nn.ModuleList(
            [
                SparseSubdivideBlock3d(model_channels, model_channels // 4),
                SparseSubdivideBlock3d(model_channels // 4, model_channels // 8),
            ]
        )
        self.out_layer = SparseLinear(model_channels // 8, self.mesh_extractor.feats_channels)

    def forward(self, x: SparseTensor) -> List[MeshExtractResult]:
        h = super().forward(x)
        for block in self.upsample:
            h = block(h)
        h = self.out_layer(h)
        return [self.mesh_extractor(h[i], training=self.training) for i in range(h.batch_size)]


class SLatFlowModel(nn.Module):
    """
    Sparse rectified-flow transformer for stage 2.

    Paper role:
    - reuse the occupied sparse coordinates found in stage 1
    - generate a feature vector for each occupied voxel instead of a dense grid
    - this sparse feature field is the paper's structured latent (SLAT)
    """

    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_head_channels: int = 64,
        mlp_ratio: float = 4.0,
        num_io_res_blocks: int = 2,
        io_block_channels: Optional[List[int]] = None,
        use_skip_connection: bool = True,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_heads = model_channels // num_head_channels
        self.use_skip_connection = use_skip_connection
        self.share_mod = share_mod
        self.t_embedder = TimestepEmbedder(model_channels)
        self.pos_embedder = AbsolutePositionEmbedder(model_channels)
        self.input_layer = SparseLinear(in_channels, model_channels if io_block_channels is None else io_block_channels[0])
        self.input_blocks = nn.ModuleList()
        self.out_blocks = nn.ModuleList()
        if io_block_channels is not None:
            for channels, next_channels in zip(io_block_channels, io_block_channels[1:] + [model_channels]):
                for _ in range(num_io_res_blocks - 1):
                    self.input_blocks.append(SparseResBlock3d(channels, model_channels, out_channels=channels))
                self.input_blocks.append(SparseResBlock3d(channels, model_channels, out_channels=next_channels, downsample=True))
            reversed_channels = list(reversed(io_block_channels))
            reversed_prev = [model_channels] + list(reversed(io_block_channels[1:]))
            for channels, prev_channels in zip(reversed_channels, reversed_prev):
                in_ch = prev_channels * 2 if use_skip_connection else prev_channels
                self.out_blocks.append(SparseResBlock3d(in_ch, model_channels, out_channels=channels, upsample=True))
                for _ in range(num_io_res_blocks - 1):
                    in_ch = channels * 2 if use_skip_connection else channels
                    self.out_blocks.append(SparseResBlock3d(in_ch, model_channels, out_channels=channels))
        self.blocks = nn.ModuleList(
            [
                ModulatedSparseTransformerCrossBlock(
                    channels=model_channels,
                    ctx_channels=cond_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                    share_mod=share_mod,
                    qk_rms_norm=qk_rms_norm,
                    qk_rms_norm_cross=qk_rms_norm_cross,
                )
                for _ in range(num_blocks)
            ]
        )
        if share_mod:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 6 * model_channels))
        self.out_layer = SparseLinear(model_channels if io_block_channels is None else io_block_channels[0], out_channels)

    def forward(self, x: SparseTensor, t: Tensor, cond: Tensor) -> SparseTensor:
        cond = repeat_cond_if_needed(cond, x.batch_size)
        h = self.input_layer(x)
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        skips: List[Tensor] = []
        for block in self.input_blocks:
            h = block(h, t_emb)
            skips.append(h.feats)
        h = h + self.pos_embedder(h.coords[:, 1:].float())
        for block in self.blocks:
            h = block(h, t_emb, cond)
        for block, skip in zip(self.out_blocks, reversed(skips)):
            h = h.replace(torch.cat([h.feats, skip], dim=1)) if self.use_skip_connection else h
            h = block(h, t_emb)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        return self.out_layer(h)


# -----------------------------------------------------------------------------
# Flow matching: training objective and Euler sampler
# -----------------------------------------------------------------------------


def randn_like(x: Union[Tensor, SparseTensor]) -> Union[Tensor, SparseTensor]:
    if isinstance(x, SparseTensor):
        return x.replace(torch.randn_like(x.feats))
    return torch.randn_like(x)


def expand_t_like(x: Union[Tensor, SparseTensor], t: Union[Tensor, float]) -> Union[Tensor, float]:
    if isinstance(t, float):
        if isinstance(x, SparseTensor):
            return torch.full((x.feats.shape[0], 1), t, device=x.device, dtype=x.dtype)
        dims = [1] * (x.ndim - 1)
        return torch.full((x.shape[0], *dims), t, device=x.device, dtype=x.dtype)
    if isinstance(x, SparseTensor):
        return sparse_batch_broadcast(x, t[:, None]).to(x.dtype)
    return t.view(-1, *([1] * (x.ndim - 1))).to(x.dtype)


def diffuse(x_0: Union[Tensor, SparseTensor], t: Tensor, noise: Union[Tensor, SparseTensor], sigma_min: float = 1e-5):
    t_full = expand_t_like(x_0, t)
    if isinstance(x_0, SparseTensor):
        return x_0 * (1 - t_full) + noise * (sigma_min + (1 - sigma_min) * t_full)
    return (1 - t_full) * x_0 + (sigma_min + (1 - sigma_min) * t_full) * noise


def get_v_target(x_0: Union[Tensor, SparseTensor], noise: Union[Tensor, SparseTensor], sigma_min: float = 1e-5):
    return (1 - sigma_min) * noise - x_0


def flow_matching_loss(denoiser: nn.Module, x_0: Union[Tensor, SparseTensor], cond: Tensor, sigma_min: float = 1e-5, t: Optional[Tensor] = None) -> Tensor:
    """
    Minimal readable training objective used by TRELLIS flow models.
    """

    batch_size = x_0.shape[0] if isinstance(x_0, Tensor) else x_0.batch_size
    if t is None:
        t = sample_logit_normal_t(batch_size, device=x_0.device if isinstance(x_0, SparseTensor) else x_0.device)
    noise = randn_like(x_0)
    x_t = diffuse(x_0, t, noise, sigma_min=sigma_min)
    target = get_v_target(x_0, noise, sigma_min=sigma_min)
    pred = denoiser(x_t, t * 1000, cond)
    return mse_between(pred, target)


class FlowEulerSampler:
    """
    Minimal copy of trellis.pipelines.samplers.flow_euler.FlowEulerSampler.

    It integrates the velocity field predicted by the flow model from t=1 to t=0.
    """

    def __init__(self, sigma_min: float):
        self.sigma_min = sigma_min

    def _v_to_xstart_eps(self, x_t: Union[Tensor, SparseTensor], t: float, v: Union[Tensor, SparseTensor]):
        t_full = expand_t_like(x_t, float(t))
        if isinstance(x_t, SparseTensor):
            eps = v * (1 - t_full) + x_t
            x_0 = x_t * (1 - self.sigma_min) - v * (self.sigma_min + (1 - self.sigma_min) * t_full)
        else:
            eps = (1 - t_full) * v + x_t
            x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t_full) * v
        return x_0, eps

    def _inference_model(self, model: nn.Module, x_t: Union[Tensor, SparseTensor], t: float, cond: Optional[Tensor] = None, **kwargs: Any):
        if cond is not None:
            batch_size = x_t.shape[0] if isinstance(x_t, Tensor) else x_t.batch_size
            cond = repeat_cond_if_needed(cond, batch_size)
        model_t = torch.full(
            (x_t.shape[0],) if isinstance(x_t, Tensor) else (x_t.batch_size,),
            1000 * t,
            device=x_t.device if isinstance(x_t, SparseTensor) else x_t.device,
            dtype=torch.float32,
        )
        return model(x_t, model_t, cond, **kwargs)

    def _get_model_prediction(self, model: nn.Module, x_t: Union[Tensor, SparseTensor], t: float, cond: Optional[Tensor] = None, **kwargs: Any):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t, t, pred_v)
        return pred_x_0, pred_eps, pred_v

    @torch.no_grad()
    def sample_once(self, model: nn.Module, x_t: Union[Tensor, SparseTensor], t: float, t_prev: float, cond: Optional[Tensor] = None, **kwargs: Any) -> Dict[str, Union[Tensor, SparseTensor]]:
        pred_x_0, _pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return {"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0}

    @torch.no_grad()
    def sample(self, model: nn.Module, noise: Union[Tensor, SparseTensor], cond: Optional[Tensor] = None, steps: int = 50, rescale_t: float = 1.0, **kwargs: Any) -> Dict[str, Any]:
        sample = noise
        t_seq = torch.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        pred_x_t = []
        pred_x_0 = []
        for i in range(steps):
            t = float(t_seq[i])
            t_prev = float(t_seq[i + 1])
            out = self.sample_once(model, sample, t, t_prev, cond=cond, **kwargs)
            sample = out["pred_x_prev"]
            pred_x_t.append(sample)
            pred_x_0.append(out["pred_x_0"])
        return {"samples": sample, "pred_x_t": pred_x_t, "pred_x_0": pred_x_0}


class FlowEulerGuidanceIntervalSampler(FlowEulerSampler):
    """
    Same sampler plus classifier-free guidance in a timestep interval.
    """

    def _inference_model(
        self,
        model: nn.Module,
        x_t: Union[Tensor, SparseTensor],
        t: float,
        cond: Optional[Tensor] = None,
        neg_cond: Optional[Tensor] = None,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        **kwargs: Any,
    ):
        if neg_cond is None or cond is None or not (cfg_interval[0] <= t <= cfg_interval[1]):
            return super()._inference_model(model, x_t, t, cond=cond, **kwargs)
        pred = super()._inference_model(model, x_t, t, cond=cond, **kwargs)
        neg_pred = super()._inference_model(model, x_t, t, cond=neg_cond, **kwargs)
        return (1 + cfg_strength) * pred - cfg_strength * neg_pred


# -----------------------------------------------------------------------------
# Readable condition encoders
# -----------------------------------------------------------------------------


class ImageConditionEncoder(nn.Module):
    """
    Minimal stand-in for TRELLIS image conditioning.

    Real repo path:
    preprocess_image -> DINOv2(x_prenorm) -> layer_norm -> patch tokens
    """

    def preprocess_image(self, image: Any) -> Any:
        # Real code does background removal, alpha crop, resize to 518x518,
        # then premultiplies the alpha channel.
        return image

    def encode_image(self, image: Union[Tensor, Any]) -> Tensor:
        if isinstance(image, torch.Tensor):
            return image
        raise NotImplementedError("Pass precomputed DINO patch tokens, or plug in the real DINOv2 encoder.")

    def get_cond(self, image: Union[Tensor, Any]) -> Dict[str, Tensor]:
        cond = self.encode_image(image)
        return {"cond": cond, "neg_cond": torch.zeros_like(cond)}


class TextConditionEncoder(nn.Module):
    """
    Minimal stand-in for TRELLIS text conditioning.

    Real repo path:
    tokenizer -> CLIPTextModel.last_hidden_state -> cond tokens
    """

    def encode_text(self, prompt: Union[Tensor, List[str], str]) -> Tensor:
        if isinstance(prompt, torch.Tensor):
            return prompt
        raise NotImplementedError("Pass precomputed CLIP text tokens, or plug in the real tokenizer + encoder.")

    def get_cond(self, prompt: Union[Tensor, List[str], str]) -> Dict[str, Tensor]:
        cond = self.encode_text(prompt)
        return {"cond": cond, "neg_cond": torch.zeros_like(cond)}


# -----------------------------------------------------------------------------
# Minimal end-to-end pipelines
# -----------------------------------------------------------------------------


class TrellisCorePipeline(nn.Module):
    """
    Shared core pipeline:
    cond -> stage-1 sparse structure -> stage-2 SLAT -> decode to 3D outputs
    """

    def __init__(
        self,
        sparse_structure_flow_model: SparseStructureFlowModel,
        sparse_structure_decoder: SparseStructureDecoder,
        slat_flow_model: SLatFlowModel,
        slat_decoder_gs: Optional[SLatGaussianDecoder] = None,
        slat_decoder_rf: Optional[SLatRadianceFieldDecoder] = None,
        slat_decoder_mesh: Optional[SLatMeshDecoder] = None,
        sparse_structure_sampler: Optional[FlowEulerSampler] = None,
        slat_sampler: Optional[FlowEulerSampler] = None,
        slat_normalization: Optional[Dict[str, List[float]]] = None,
    ):
        super().__init__()
        self.sparse_structure_flow_model = sparse_structure_flow_model
        self.sparse_structure_decoder = sparse_structure_decoder
        self.slat_flow_model = slat_flow_model
        self.slat_decoder_gs = slat_decoder_gs
        self.slat_decoder_rf = slat_decoder_rf
        self.slat_decoder_mesh = slat_decoder_mesh
        self.sparse_structure_sampler = sparse_structure_sampler or FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
        self.slat_sampler = slat_sampler or FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
        latent_dim = slat_flow_model.out_channels
        self.slat_normalization = slat_normalization or {
            "mean": [0.0] * latent_dim,
            "std": [1.0] * latent_dim,
        }

    def sample_sparse_structure(self, cond: Dict[str, Tensor], num_samples: int = 1, sampler_params: Optional[Dict[str, Any]] = None) -> Tensor:
        """
        Stage 1:
        dense noise -> sparse structure flow -> z_s -> occupancy logits -> occupied coords
        """

        sampler_params = sampler_params or {}
        flow_model = self.sparse_structure_flow_model
        noise = torch.randn(
            num_samples,
            flow_model.in_channels,
            flow_model.resolution,
            flow_model.resolution,
            flow_model.resolution,
            device=cond["cond"].device,
        )
        z_s = self.sparse_structure_sampler.sample(flow_model, noise, **cond, **sampler_params)["samples"]
        occupancy_logits = self.sparse_structure_decoder(z_s)
        coords = torch.argwhere(occupancy_logits > 0)[:, [0, 2, 3, 4]].int()
        return coords

    def sample_slat(self, cond: Dict[str, Tensor], coords: Tensor, sampler_params: Optional[Dict[str, Any]] = None) -> SparseTensor:
        """
        Stage 2:
        sparse noise on occupied coords -> sparse SLAT flow -> normalized SLAT -> denormalized SLAT
        """

        sampler_params = sampler_params or {}
        flow_model = self.slat_flow_model
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels, device=coords.device),
            coords=coords,
        )
        slat = self.slat_sampler.sample(flow_model, noise, **cond, **sampler_params)["samples"]
        mean = torch.tensor(self.slat_normalization["mean"], device=slat.device, dtype=slat.dtype)[None]
        std = torch.tensor(self.slat_normalization["std"], device=slat.device, dtype=slat.dtype)[None]
        return slat * std + mean

    def decode_slat(self, slat: SparseTensor, formats: List[str]) -> Dict[str, Any]:
        outputs: Dict[str, Any] = {}
        if "mesh" in formats and self.slat_decoder_mesh is not None:
            outputs["mesh"] = self.slat_decoder_mesh(slat)
        if "gaussian" in formats and self.slat_decoder_gs is not None:
            outputs["gaussian"] = self.slat_decoder_gs(slat)
        if "radiance_field" in formats and self.slat_decoder_rf is not None:
            outputs["radiance_field"] = self.slat_decoder_rf(slat)
        return outputs


class TrellisImageTo3DPipeline(TrellisCorePipeline):
    """
    Readable mirror of trellis.pipelines.TrellisImageTo3DPipeline.
    """

    def __init__(self, image_conditioner: ImageConditionEncoder, **kwargs: Any):
        super().__init__(**kwargs)
        self.image_conditioner = image_conditioner

    @torch.no_grad()
    def run(
        self,
        image: Union[Tensor, Any],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: Optional[Dict[str, Any]] = None,
        slat_sampler_params: Optional[Dict[str, Any]] = None,
        formats: Optional[List[str]] = None,
        preprocess_image: bool = True,
    ) -> Dict[str, Any]:
        if preprocess_image:
            image = self.image_conditioner.preprocess_image(image)
        cond = self.image_conditioner.get_cond(image)
        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats or ["mesh", "gaussian", "radiance_field"])


class TrellisTextTo3DPipeline(TrellisCorePipeline):
    """
    Readable mirror of trellis.pipelines.TrellisTextTo3DPipeline.
    """

    def __init__(self, text_conditioner: TextConditionEncoder, **kwargs: Any):
        super().__init__(**kwargs)
        self.text_conditioner = text_conditioner

    @torch.no_grad()
    def run(
        self,
        prompt: Union[Tensor, List[str], str],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: Optional[Dict[str, Any]] = None,
        slat_sampler_params: Optional[Dict[str, Any]] = None,
        formats: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        cond = self.text_conditioner.get_cond(prompt)
        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats or ["mesh", "gaussian", "radiance_field"])


# -----------------------------------------------------------------------------
# Minimal reading-oriented example
# -----------------------------------------------------------------------------


def build_demo_trellis_core() -> TrellisImageTo3DPipeline:
    """
    Instantiate a tiny reading-oriented TRELLIS stack.

    The dimensions here are intentionally small. This function is not meant to
    reproduce released checkpoints; it only wires the paper modules together in
    one place so the call graph is easy to follow.
    """

    sparse_structure_flow_model = SparseStructureFlowModel(
        resolution=16,
        in_channels=8,
        model_channels=256,
        cond_channels=1024,
        out_channels=8,
        num_blocks=4,
        patch_size=2,
    )
    sparse_structure_decoder = SparseStructureDecoder(
        out_channels=1,
        latent_channels=8,
        channels=[64, 128, 256],
    )
    slat_flow_model = SLatFlowModel(
        resolution=64,
        in_channels=8,
        model_channels=256,
        cond_channels=1024,
        out_channels=8,
        num_blocks=4,
        io_block_channels=[64, 128],
    )
    slat_decoder_gs = SLatGaussianDecoder(
        resolution=64,
        model_channels=256,
        latent_channels=8,
        num_blocks=4,
    )
    slat_decoder_rf = SLatRadianceFieldDecoder(
        resolution=64,
        model_channels=256,
        latent_channels=8,
        num_blocks=4,
    )
    slat_decoder_mesh = SLatMeshDecoder(
        resolution=64,
        model_channels=256,
        latent_channels=8,
        num_blocks=4,
    )
    return TrellisImageTo3DPipeline(
        image_conditioner=ImageConditionEncoder(),
        sparse_structure_flow_model=sparse_structure_flow_model,
        sparse_structure_decoder=sparse_structure_decoder,
        slat_flow_model=slat_flow_model,
        slat_decoder_gs=slat_decoder_gs,
        slat_decoder_rf=slat_decoder_rf,
        slat_decoder_mesh=slat_decoder_mesh,
        sparse_structure_sampler=FlowEulerGuidanceIntervalSampler(sigma_min=1e-5),
        slat_sampler=FlowEulerGuidanceIntervalSampler(sigma_min=1e-5),
        slat_normalization={"mean": [0.0] * 8, "std": [1.0] * 8},
    )


if __name__ == "__main__":
    print("paper_core_minimal.py is a reading-oriented extraction, not a full runnable TRELLIS clone.")
    print("Start from build_demo_trellis_core(), then trace:")
    print("  condition -> sample_sparse_structure -> sample_slat -> decode_slat")
