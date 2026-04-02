#!/usr/bin/env python3
from __future__ import annotations
"""
TRELLIS image UniEdit editing with RF inversion initialization.

This script keeps TRELLIS core code untouched and adds a two-stage UniEdit
workflow on top of the existing RF inversion initialization:

1. Stage 1 (`sparse_structure`)
   - Invert the source asset to terminal noise.
   - Run UniEdit-style source/target velocity fusion on sparse structure.
   - Decode the edited occupancy and only keep voxel changes inside `--mask_glb`.
     Voxels outside the 3D mask are restored from the source structure.

2. Stage 2 (`slat`)
   - Fix the Stage-1 coordinates and project source terminal noise onto them.
   - Run two ablations:
     - `preserve_uniedit`: overlapping/source-kept voxels keep using UniEdit,
       while newly added voxels are left to the target branch freely.
     - `free_target`: same coordinates and initialization, but Stage 2 uses the
       target branch only.

The source image, edit image, and optional 2D mask are still used to build
aligned image conditions. The local 3D edit region comes from `--mask_glb`.
"""

import argparse
import gc
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import trimesh
from PIL import Image
from tqdm import tqdm

from output_layout import build_output_layout
import example_image_prompt_to_prompt as image_p2p
import example_image_prompt_to_prompt_rf_inversion as rf_utils


_attn_backend = image_p2p._peek_arg("--attn-backend", "")
if _attn_backend:
    os.environ["ATTN_BACKEND"] = _attn_backend

os.environ.setdefault("SPCONV_ALGO", "native")


EDIT_METHOD_NAME = "image_uniedit_rf_inversion"
STAGE2_VARIANTS = ("preserve_uniedit", "free_target")


def _coords3d(coords: torch.Tensor) -> torch.Tensor:
    if coords.ndim != 2:
        raise RuntimeError(f"Expected 2D coords tensor, got shape {tuple(coords.shape)}")
    if coords.shape[1] == 3:
        return coords.int()
    if coords.shape[1] == 4:
        return coords[:, 1:].int()
    raise RuntimeError(f"Unsupported coordinate shape: {tuple(coords.shape)}")


def coords3d_to_batched(coords: torch.Tensor, batch_idx: int = 0) -> torch.Tensor:
    coords3 = _coords3d(coords)
    batch = torch.full(
        (coords3.shape[0], 1),
        fill_value=int(batch_idx),
        dtype=torch.int32,
        device=coords3.device,
    )
    return torch.cat([batch, coords3], dim=1)


def load_mask_glb_coords(mask_glb: str, device: torch.device, resolution: int = 64) -> Tuple[Optional[torch.Tensor], dict]:
    mask_glb = mask_glb.strip()
    if not mask_glb:
        return None, {"mask_glb": None, "enabled": False}

    mask_path = rf_utils.ensure_path_exists(Path(mask_glb).expanduser().resolve(), "mask_glb")
    loaded = trimesh.load(str(mask_path), process=False)
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.dump(concatenate=True)
        if not isinstance(mesh, trimesh.Trimesh) or mesh.vertices.shape[0] == 0:
            raise RuntimeError(f"mask_glb contains no usable triangle mesh after scene concatenation: {mask_path}")
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded.copy()
    else:
        raise RuntimeError(f"Unsupported mask_glb payload: {type(loaded)}")

    raw_vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if raw_vertices.ndim != 2 or raw_vertices.shape[1] != 3 or raw_vertices.shape[0] == 0:
        raise RuntimeError(f"mask_glb must contain valid Nx3 vertices, got shape {raw_vertices.shape}")

    raw_min = raw_vertices.min(axis=0)
    raw_max = raw_vertices.max(axis=0)
    auto_normalized = False
    if raw_min.min() < -0.55 or raw_max.max() > 0.55:
        center = (raw_min + raw_max) / 2.0
        scale = float(np.max(raw_max - raw_min))
        if scale <= 0.0:
            raise RuntimeError(f"mask_glb has degenerate bounds and cannot be normalized: {mask_path}")
        mesh.vertices = (raw_vertices - center) / scale
        auto_normalized = True

    voxel = mesh.voxelized(pitch=1.0 / float(resolution))
    try:
        voxel = voxel.fill()
    except Exception:
        pass
    points = np.asarray(voxel.points, dtype=np.float32)
    if points.size == 0:
        raise RuntimeError(f"mask_glb voxelization produced no occupied voxels: {mask_path}")

    coords = np.floor((points + 0.5) * float(resolution)).astype(np.int32)
    coords = np.clip(coords, 0, resolution - 1)
    coords = np.unique(coords, axis=0)
    if coords.shape[0] == 0:
        raise RuntimeError(f"mask_glb voxelization became empty after clipping: {mask_path}")

    meta = {
        "mask_glb": str(mask_path),
        "enabled": True,
        "resolution": int(resolution),
        "auto_normalized_to_unit_cube": bool(auto_normalized),
        "raw_aabb_min": raw_min.tolist(),
        "raw_aabb_max": raw_max.tolist(),
        "voxel_count": int(coords.shape[0]),
    }
    return torch.from_numpy(coords).int().to(device=device), meta


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def move_models(pipeline, names: Iterable[str], device: torch.device) -> None:
    for name in names:
        pipeline.models[name].to(device)
    release_cuda_memory()


def encode_cond_on_device(pipeline, images, device: torch.device) -> dict:
    model = pipeline.models["image_cond_model"]
    move_models(pipeline, ["image_cond_model"], device)
    if isinstance(images, list):
        image = [i.resize((518, 518), Image.LANCZOS) for i in images]
        image = [np.array(i.convert("RGB")).astype(np.float32) / 255 for i in image]
        image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
        image = torch.stack(image).to(device)
    elif isinstance(images, torch.Tensor):
        image = images.to(device=device)
    else:
        raise RuntimeError(f"Unsupported image input type: {type(images)}")

    image = pipeline.image_cond_model_transform(image).to(device)
    with torch.no_grad():
        features = model(image, is_training=True)["x_prenorm"]
        patchtokens = torch.nn.functional.layer_norm(features, features.shape[-1:])
    return {"cond": patchtokens, "neg_cond": torch.zeros_like(patchtokens)}


def feats_to_slat_on_device(pipeline, feats_path: Path, device: torch.device):
    feats = np.load(feats_path)
    if "patchtokens" not in feats or "indices" not in feats:
        raise RuntimeError(
            f"features.npz must contain 'patchtokens' and 'indices', got keys: {list(feats.keys())}"
        )
    move_models(pipeline, ["slat_encoder"], device)
    sparse_tensor = image_p2p.SparseTensor(
        feats=torch.from_numpy(feats["patchtokens"]).float().to(device),
        coords=torch.cat(
            [
                torch.zeros(feats["patchtokens"].shape[0], 1, dtype=torch.int32),
                torch.from_numpy(feats["indices"]).int(),
            ],
            dim=1,
        ).to(device),
    )
    feats_encoder = pipeline.models["slat_encoder"]
    with torch.no_grad():
        return feats_encoder(sparse_tensor, sample_posterior=False)


def _dense_scalar_field(x: torch.Tensor) -> torch.Tensor:
    return x.abs().mean(dim=1, keepdim=True)


def _sparse_scalar_field(x) -> torch.Tensor:
    return x.feats.abs().mean(dim=1, keepdim=True)


def _normalize_dense_map(scores: torch.Tensor, selector: Optional[torch.Tensor]) -> torch.Tensor:
    bsz = scores.shape[0]
    scores_flat = scores.reshape(bsz, -1)
    out_flat = torch.zeros_like(scores_flat)
    if selector is None:
        selector_flat = torch.ones_like(scores_flat, dtype=torch.bool)
    else:
        selector_flat = selector.reshape(bsz, -1) > 0.5

    for batch_idx in range(bsz):
        active = selector_flat[batch_idx]
        if not torch.any(active):
            continue
        vals = scores_flat[batch_idx, active]
        min_val = vals.min()
        max_val = vals.max()
        denom = torch.clamp(max_val - min_val, min=1e-6)
        out_flat[batch_idx, active] = (scores_flat[batch_idx, active] - min_val) / denom
    return out_flat.reshape_as(scores)


def _normalize_sparse_map(scores: torch.Tensor, coords: torch.Tensor, selector: Optional[torch.Tensor]) -> torch.Tensor:
    out = torch.zeros_like(scores)
    batch_ids = coords[:, 0].long()
    active_selector = None if selector is None else (selector.reshape(-1) > 0.5)
    batch_size = int(batch_ids.max().item()) + 1 if batch_ids.numel() > 0 else 0

    for batch_idx in range(batch_size):
        batch_mask = batch_ids == batch_idx
        if active_selector is None:
            active = batch_mask
        else:
            active = batch_mask & active_selector
        if not torch.any(active):
            continue
        vals = scores[active]
        min_val = vals.min()
        max_val = vals.max()
        denom = torch.clamp(max_val - min_val, min=1e-6)
        out[active] = (scores[active] - min_val) / denom
    return out


def compute_uniedit_map(guidance, selector: Optional[torch.Tensor] = None) -> torch.Tensor:
    if isinstance(guidance, image_p2p.SparseTensor):
        scores = _sparse_scalar_field(guidance)
        return _normalize_sparse_map(scores=scores, coords=guidance.coords, selector=selector)
    if torch.is_tensor(guidance):
        scores = _dense_scalar_field(guidance)
        return _normalize_dense_map(scores=scores, selector=selector)
    raise RuntimeError(f"Unsupported guidance type: {type(guidance)}")


class UniEditRFSampler(rf_utils.SecondOrderRFSampler):
    def _guided_prediction_for_cond(
        self,
        model,
        sample,
        t_value: float,
        cond: torch.Tensor,
        neg_cond: torch.Tensor,
        cfg_strength: float,
        cfg_interval: Tuple[float, float],
    ):
        if cfg_interval[0] <= t_value <= cfg_interval[1] and cfg_strength > 0.0:
            pred = self._run_model(model, sample, t_value, cond)
            neg_pred = self._run_model(model, sample, t_value, neg_cond)
            return (1.0 + cfg_strength) * pred - cfg_strength * neg_pred
        return self._run_model(model, sample, t_value, cond)

    def _merged_prediction(
        self,
        model,
        sample,
        t_value: float,
        source_cond: torch.Tensor,
        target_cond: torch.Tensor,
        neg_cond: torch.Tensor,
        cfg_strength: float,
        cfg_interval: Tuple[float, float],
        omega: float,
        selector: Optional[torch.Tensor],
        mode: str,
    ):
        pred_tgt = self._guided_prediction_for_cond(
            model=model,
            sample=sample,
            t_value=t_value,
            cond=target_cond,
            neg_cond=neg_cond,
            cfg_strength=cfg_strength,
            cfg_interval=cfg_interval,
        )
        if mode == "target_only":
            return pred_tgt

        pred_src = self._guided_prediction_for_cond(
            model=model,
            sample=sample,
            t_value=t_value,
            cond=source_cond,
            neg_cond=neg_cond,
            cfg_strength=cfg_strength,
            cfg_interval=cfg_interval,
        )
        guidance = pred_tgt - pred_src
        save_map = compute_uniedit_map(guidance, selector=selector if mode == "preserve_overlap" else None)
        fused = pred_tgt * save_map + pred_src * (1.0 - save_map)
        pred = fused + guidance * ((1.0 + save_map) * float(omega))
        if mode == "preserve_overlap" and selector is not None:
            pred = pred * selector + pred_tgt * (1.0 - selector)
        return pred

    def sample_once(
        self,
        model,
        sample,
        t_curr: float,
        t_next: float,
        source_cond: torch.Tensor,
        target_cond: torch.Tensor,
        neg_cond: torch.Tensor,
        cfg_strength: float,
        cfg_interval: Tuple[float, float],
        omega: float,
        selector: Optional[torch.Tensor],
        mode: str,
    ):
        pred = self._merged_prediction(
            model=model,
            sample=sample,
            t_value=t_curr,
            source_cond=source_cond,
            target_cond=target_cond,
            neg_cond=neg_cond,
            cfg_strength=cfg_strength,
            cfg_interval=cfg_interval,
            omega=omega,
            selector=selector,
            mode=mode,
        )
        dt = t_next - t_curr
        sample_mid = sample + 0.5 * dt * pred
        t_mid = t_curr + 0.5 * dt
        pred_mid = self._merged_prediction(
            model=model,
            sample=sample_mid,
            t_value=t_mid,
            source_cond=source_cond,
            target_cond=target_cond,
            neg_cond=neg_cond,
            cfg_strength=cfg_strength,
            cfg_interval=cfg_interval,
            omega=omega,
            selector=selector,
            mode=mode,
        )
        first_order = (pred_mid - pred) / (0.5 * dt)
        return sample + dt * pred - 0.5 * (dt ** 2) * first_order

    def sample(
        self,
        model,
        sample,
        source_cond: torch.Tensor,
        target_cond: torch.Tensor,
        neg_cond: torch.Tensor,
        steps: int,
        rescale_t: float,
        cfg_strength: float,
        cfg_interval: Tuple[float, float],
        omega: float,
        selector: Optional[torch.Tensor] = None,
        mode: str = "full_uniedit",
        verbose: bool = True,
    ):
        t_seq = np.linspace(1.0, 0.0, int(steps) + 1)
        t_seq = rescale_t * t_seq / (1.0 + (rescale_t - 1.0) * t_seq)
        t_pairs = list((float(t_seq[i]), float(t_seq[i + 1])) for i in range(len(t_seq) - 1))

        desc_map = {
            "full_uniedit": "UniEdit denoise",
            "preserve_overlap": "UniEdit denoise (preserve overlap)",
            "target_only": "Target-only denoise",
        }
        for t_curr, t_next in tqdm(t_pairs, desc=desc_map.get(mode, "UniEdit denoise"), disable=not verbose):
            sample = self.sample_once(
                model=model,
                sample=sample,
                t_curr=t_curr,
                t_next=t_next,
                source_cond=source_cond,
                target_cond=target_cond,
                neg_cond=neg_cond,
                cfg_strength=cfg_strength,
                cfg_interval=cfg_interval,
                omega=omega,
                selector=selector,
                mode=mode,
            )
        return sample


def build_stage2_selector(coords_target: torch.Tensor, coords_source: torch.Tensor) -> torch.Tensor:
    tgt_codes = rf_utils.coords_to_flat_indices(_coords3d(coords_target))
    src_codes = rf_utils.coords_to_flat_indices(_coords3d(coords_source))
    overlap = torch.isin(tgt_codes, src_codes)
    return overlap.to(device=coords_target.device, dtype=torch.float32).unsqueeze(1)


def compose_stage1_coords(
    coords_source: torch.Tensor,
    coords_stage1_raw: torch.Tensor,
    mask_coords: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, dict]:
    coords_source = _coords3d(coords_source)
    coords_stage1_raw = _coords3d(coords_stage1_raw)
    raw_batched = coords3d_to_batched(coords_stage1_raw, batch_idx=0)

    if mask_coords is None:
        return raw_batched, {"mask_enabled": False, "masked_voxel_count": None}

    mask_coords = _coords3d(mask_coords)
    src_codes = rf_utils.coords_to_flat_indices(coords_source)
    raw_codes = rf_utils.coords_to_flat_indices(coords_stage1_raw)
    mask_codes = rf_utils.coords_to_flat_indices(mask_coords)

    src_outside_mask = coords_source[~torch.isin(src_codes, mask_codes)]
    raw_inside_mask = coords_stage1_raw[torch.isin(raw_codes, mask_codes)]
    composed = torch.cat([src_outside_mask, raw_inside_mask], dim=0)
    composed = torch.unique(composed, dim=0).int()
    composed_batched = coords3d_to_batched(composed, batch_idx=0)

    meta = {
        "mask_enabled": True,
        "masked_voxel_count": int(mask_coords.shape[0]),
        "source_outside_mask_count": int(src_outside_mask.shape[0]),
        "edited_inside_mask_count": int(raw_inside_mask.shape[0]),
        "composed_voxel_count": int(composed.shape[0]),
    }
    return composed_batched, meta


def summarize_coord_transition(
    coords_source: torch.Tensor,
    coords_stage1_raw: torch.Tensor,
    coords_stage1_masked: torch.Tensor,
    mask_coords: Optional[torch.Tensor],
) -> dict:
    src = _coords3d(coords_source)
    raw = _coords3d(coords_stage1_raw)
    masked = _coords3d(coords_stage1_masked)
    src_codes = rf_utils.coords_to_flat_indices(src)
    raw_codes = rf_utils.coords_to_flat_indices(raw)
    masked_codes = rf_utils.coords_to_flat_indices(masked)

    raw_overlap = int(torch.isin(raw_codes, src_codes).sum().item())
    masked_overlap = int(torch.isin(masked_codes, src_codes).sum().item())

    stats = {
        "source_voxel_count": int(src.shape[0]),
        "stage1_raw_voxel_count": int(raw.shape[0]),
        "stage1_masked_voxel_count": int(masked.shape[0]),
        "stage1_raw_overlap_with_source": raw_overlap,
        "stage1_masked_overlap_with_source": masked_overlap,
        "stage1_raw_added_count": int((~torch.isin(raw_codes, src_codes)).sum().item()),
        "stage1_raw_removed_count": int((~torch.isin(src_codes, raw_codes)).sum().item()),
        "stage1_masked_added_count": int((~torch.isin(masked_codes, src_codes)).sum().item()),
        "stage1_masked_removed_count": int((~torch.isin(src_codes, masked_codes)).sum().item()),
    }

    if mask_coords is not None:
        mask_codes = rf_utils.coords_to_flat_indices(_coords3d(mask_coords))
        stats.update(
            {
                "mask_voxel_count": int(mask_coords.shape[0]),
                "stage1_raw_voxels_in_mask": int(torch.isin(raw_codes, mask_codes).sum().item()),
                "stage1_masked_voxels_in_mask": int(torch.isin(masked_codes, mask_codes).sum().item()),
                "source_voxels_in_mask": int(torch.isin(src_codes, mask_codes).sum().item()),
            }
        )
    else:
        stats["mask_voxel_count"] = None
    return stats


def denoise_sparse_structure_uniedit(
    pipeline,
    source_cond: dict,
    target_cond: dict,
    terminal_noise: torch.Tensor,
    params: dict,
    cfg_interval: Tuple[float, float],
    omega: float,
    verbose: bool,
) -> torch.Tensor:
    flow_model = pipeline.models["sparse_structure_flow_model"]
    decoder = pipeline.models["sparse_structure_decoder"]
    sampler = UniEditRFSampler()
    z_tgt = sampler.sample(
        model=flow_model,
        sample=terminal_noise,
        source_cond=source_cond["cond"],
        target_cond=target_cond["cond"],
        neg_cond=target_cond["neg_cond"],
        steps=params["steps"],
        rescale_t=params["rescale_t"],
        cfg_strength=params["cfg_strength"],
        cfg_interval=cfg_interval,
        omega=float(omega),
        selector=None,
        mode="full_uniedit",
        verbose=verbose,
    )
    voxel = decoder(z_tgt)
    coords = torch.argwhere(voxel > 0)[:, [0, 2, 3, 4]].int()
    if coords.shape[0] == 0:
        raise RuntimeError("Stage 1 UniEdit sparse-structure denoising produced an empty target structure.")
    return coords


def denoise_slat_variant(
    pipeline,
    source_cond: dict,
    target_cond: dict,
    terminal_noise,
    params: dict,
    cfg_interval: Tuple[float, float],
    omega: float,
    selector: Optional[torch.Tensor],
    mode: str,
    verbose: bool,
):
    flow_model = pipeline.models["slat_flow_model"]
    sampler = UniEditRFSampler()
    slat_normalized = sampler.sample(
        model=flow_model,
        sample=terminal_noise,
        source_cond=source_cond["cond"],
        target_cond=target_cond["cond"],
        neg_cond=target_cond["neg_cond"],
        steps=params["steps"],
        rescale_t=params["rescale_t"],
        cfg_strength=params["cfg_strength"],
        cfg_interval=cfg_interval,
        omega=float(omega),
        selector=selector,
        mode=mode,
        verbose=verbose,
    )
    mean, std = rf_utils.get_slat_norm_tensors(
        pipeline,
        slat_normalized.device,
        slat_normalized.feats.dtype,
    )
    return slat_normalized * std + mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TRELLIS image UniEdit editing with RF inversion initialization."
    )
    parser.add_argument("--model", default="microsoft/TRELLIS-image-large", help="Pipeline checkpoint or HF repo.")
    parser.add_argument("--input_model", default="", help="Compatibility arg matching VoxHammer inference.py.")
    parser.add_argument("--mask_glb", default="", help="Optional local 3D edit region GLB/GLTF.")
    parser.add_argument("--render_dir", default="", help="Render directory containing voxels.ply and features.npz.")
    parser.add_argument(
        "--output_path",
        default="",
        help="Optional primary GLB path. The preserve_uniedit result is copied here; the free ablation gets a sibling filename.",
    )
    parser.add_argument(
        "--image_dir",
        default="",
        help="Directory containing 2d_render.png, 2d_edit.png, and optionally 2d_mask.png.",
    )
    parser.add_argument(
        "--source-model",
        default="",
        help="Source asset directory, or any file inside it. The directory must contain voxels.ply and features.npz.",
    )
    parser.add_argument("--edit-image", default="", help="Edited target image path.")
    parser.add_argument(
        "--source-image",
        default="",
        help="Optional aligned source render image path. If omitted, the script tries common filenames inside source-model.",
    )
    parser.add_argument(
        "--mask-image",
        default="",
        help="Optional binary/grayscale 2D edit mask path. If omitted, a mask is auto-generated from source/edit differences.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--ss-steps", type=int, default=None, help="Override sparse-structure steps for inversion and denoise.")
    parser.add_argument("--slat-steps", type=int, default=None, help="Override SLat steps for inversion and denoise.")
    parser.add_argument("--ss-cfg", type=float, default=None, help="Override sparse-structure forward CFG strength.")
    parser.add_argument("--slat-cfg", type=float, default=None, help="Override SLat forward CFG strength.")
    parser.add_argument(
        "--ss-inverse-cfg",
        type=float,
        default=None,
        help="Optional sparse-structure inverse CFG strength. Defaults to --ss-cfg or the pipeline default.",
    )
    parser.add_argument(
        "--slat-inverse-cfg",
        type=float,
        default=None,
        help="Optional SLat inverse CFG strength. Defaults to --slat-cfg or the pipeline default.",
    )
    parser.add_argument("--ss-omega", type=float, default=1.0, help="Stage-1 UniEdit omega.")
    parser.add_argument("--slat-omega", type=float, default=1.0, help="Stage-2 UniEdit omega.")
    parser.add_argument(
        "--cfg-interval-start",
        type=float,
        default=0.5,
        help="Late-time CFG interval start used by the RF sampler.",
    )
    parser.add_argument(
        "--cfg-interval-end",
        type=float,
        default=1.0,
        help="Late-time CFG interval end used by the RF sampler.",
    )
    parser.add_argument("--mask-threshold", type=int, default=127, help="2D mask pixel threshold in [0, 255].")
    parser.add_argument(
        "--auto-mask-threshold",
        type=int,
        default=24,
        help="Pixel difference threshold used when auto-generating the 2D mask.",
    )
    parser.add_argument(
        "--auto-mask-max-filter",
        type=int,
        default=7,
        help="Odd max-filter size used to thicken the auto-generated 2D mask. Use 1 to disable.",
    )
    parser.set_defaults(preprocess=True)
    parser.add_argument(
        "--preprocess",
        dest="preprocess",
        action="store_true",
        help="Apply a shared crop/resize based on source foreground and edit foreground.",
    )
    parser.add_argument(
        "--no-preprocess",
        dest="preprocess",
        action="store_false",
        help="Assume source/edit/mask are already aligned and only resize them to the DINO input size.",
    )
    parser.add_argument("--case-name", default="", help="Optional output case name.")
    parser.add_argument("--skip-render", action="store_true", help="Skip rendering mp4 previews.")
    parser.add_argument("--skip-glb", action="store_true", help="Skip exporting GLB.")
    parser.add_argument("--skip-ply", action="store_true", help="Skip exporting gaussian PLY.")
    parser.add_argument(
        "--attn-backend",
        default=_attn_backend or "",
        help="Attention backend override: flash_attn or xformers.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce sampler progress output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backend = image_p2p.configure_attention_backend(args.attn_backend)

    from trellis.modules.sparse.basic import SparseTensor as _SparseTensor
    from trellis.pipelines import TrellisImageTo3DPipeline as _TrellisImageTo3DPipeline

    image_p2p.SparseTensor = _SparseTensor
    rf_utils.image_p2p.SparseTensor = _SparseTensor

    args.source_model = args.source_model or args.input_model
    cfg_interval = (float(args.cfg_interval_start), float(args.cfg_interval_end))
    if cfg_interval[0] > cfg_interval[1]:
        raise RuntimeError(
            f"cfg-interval-start must be <= cfg-interval-end, got {cfg_interval[0]} > {cfg_interval[1]}"
        )

    resolved_paths = rf_utils.resolve_pipeline_paths(args)
    asset_dir = resolved_paths["asset_dir"]
    voxels_path = resolved_paths["voxels_path"]
    features_path = resolved_paths["features_path"]
    source_image_path = resolved_paths["source_image_path"]
    edit_image_path = resolved_paths["edit_image_path"]
    mask_image_path = resolved_paths["mask_image_path"]
    output_path = resolved_paths["output_path"]

    pipeline = _TrellisImageTo3DPipeline.from_pretrained(args.model)
    cuda_device = torch.device("cuda")

    source_image = Image.open(source_image_path)
    edit_image = Image.open(edit_image_path)
    mask_image = Image.open(mask_image_path) if mask_image_path is not None else None

    prepared = rf_utils.prepare_inputs_with_optional_auto_mask(
        pipeline=pipeline,
        source_image=source_image,
        edit_image=edit_image,
        mask_image=mask_image,
        preprocess=bool(args.preprocess),
        mask_threshold=int(args.mask_threshold),
        auto_mask_threshold=int(args.auto_mask_threshold),
        auto_mask_max_filter=int(args.auto_mask_max_filter),
    )

    source_cond = encode_cond_on_device(pipeline, [prepared.source], cuda_device)
    edit_cond = encode_cond_on_device(pipeline, [prepared.edit], cuda_device)
    move_models(pipeline, ["image_cond_model"], torch.device("cpu"))

    ss_inverse_params, ss_forward_params = rf_utils.resolve_stage_sampling_params(
        pipeline.sparse_structure_sampler_params,
        args.ss_steps,
        args.ss_cfg,
        args.ss_inverse_cfg,
    )
    slat_inverse_params, slat_forward_params = rf_utils.resolve_stage_sampling_params(
        pipeline.slat_sampler_params,
        args.slat_steps,
        args.slat_cfg,
        args.slat_inverse_cfg,
    )

    case_name = args.case_name.strip()
    if not case_name:
        if output_path is not None:
            case_name = image_p2p.slugify(output_path.stem)
        else:
            case_name = image_p2p.slugify(f"{asset_dir.name}_to_{edit_image_path.stem}")
    output_layout = build_output_layout(EDIT_METHOD_NAME, case_name)
    out_dir = image_p2p.ensure_dir(output_layout.edit_dir)

    mask_coords, mask_meta = load_mask_glb_coords(args.mask_glb, device=cuda_device)

    image_p2p.save_json(
        out_dir / "config.json",
        {
            "method_name": EDIT_METHOD_NAME,
            "case_name": case_name,
            "model": args.model,
            "attn_backend": backend,
            "output_root_dir": str(output_layout.root_dir),
            "output_case_dir": str(output_layout.case_dir),
            "output_edit_dir": str(out_dir),
            "source_model": str(asset_dir),
            "input_model": args.input_model or None,
            "render_dir": str(resolved_paths["render_dir"]) if resolved_paths["render_dir"] is not None else None,
            "image_dir": str(resolved_paths["image_dir"]) if resolved_paths["image_dir"] is not None else None,
            "output_path": str(output_path) if output_path is not None else None,
            "voxels_ply": str(voxels_path),
            "features_npz": str(features_path),
            "source_image": str(source_image_path),
            "edit_image": str(edit_image_path),
            "mask_image": str(mask_image_path) if mask_image_path is not None else None,
            "seed": args.seed,
            "preprocess": args.preprocess,
            "cfg_interval": list(cfg_interval),
            "ss_omega": float(args.ss_omega),
            "slat_omega": float(args.slat_omega),
            "sparse_structure_inverse_params": ss_inverse_params,
            "sparse_structure_forward_params": ss_forward_params,
            "slat_inverse_params": slat_inverse_params,
            "slat_forward_params": slat_forward_params,
            "stage2_variants": list(STAGE2_VARIANTS),
            "mask_glb_meta": mask_meta,
        },
    )
    image_p2p.save_json(out_dir / "input_preprocess.json", prepared.meta)
    prepared.source.save(out_dir / "source_preprocessed.png")
    prepared.edit.save(out_dir / "edit_preprocessed.png")
    prepared.mask.save(out_dir / "mask_preprocessed.png")
    if mask_coords is not None:
        np.save(out_dir / "mask_coords.npy", mask_coords.detach().cpu().numpy())

    torch.manual_seed(args.seed)
    verbose = not bool(args.quiet)

    with torch.no_grad():
        move_models(
            pipeline,
            ["sparse_structure_encoder", "sparse_structure_flow_model"],
            cuda_device,
        )
        coords_src = rf_utils.ply_to_coords(voxels_path, cuda_device)
        voxel_src = rf_utils.coords_to_voxel(coords_src, cuda_device)

        ss_terminal_noise = rf_utils.invert_sparse_structure(
            pipeline=pipeline,
            cond_src=source_cond,
            voxel_src=voxel_src,
            params=ss_inverse_params,
            cfg_interval=cfg_interval,
            verbose=verbose,
        )
        move_models(
            pipeline,
            ["sparse_structure_encoder"],
            torch.device("cpu"),
        )

        slat_src = feats_to_slat_on_device(pipeline, features_path, cuda_device)
        move_models(
            pipeline,
            ["slat_flow_model"],
            cuda_device,
        )
        slat_terminal_noise = rf_utils.invert_slat(
            pipeline=pipeline,
            cond_src=source_cond,
            slat_src=slat_src,
            params=slat_inverse_params,
            cfg_interval=cfg_interval,
            verbose=verbose,
        )

        move_models(
            pipeline,
            ["sparse_structure_decoder"],
            cuda_device,
        )

        coords_stage1_raw = denoise_sparse_structure_uniedit(
            pipeline=pipeline,
            source_cond=source_cond,
            target_cond=edit_cond,
            terminal_noise=ss_terminal_noise,
            params=ss_forward_params,
            cfg_interval=cfg_interval,
            omega=float(args.ss_omega),
            verbose=verbose,
        )
        coords_stage1_masked, stage1_meta = compose_stage1_coords(
            coords_source=coords_src,
            coords_stage1_raw=coords_stage1_raw,
            mask_coords=mask_coords,
        )
        projected_slat_noise = rf_utils.project_sparse_terminal_noise(
            source_noise=slat_terminal_noise,
            target_coords=coords_stage1_masked.to(device=cuda_device),
            device=cuda_device,
        )
        stage2_selector = build_stage2_selector(coords_stage1_masked, coords_src)
        move_models(
            pipeline,
            ["sparse_structure_flow_model", "sparse_structure_decoder", "slat_encoder"],
            torch.device("cpu"),
        )

        stage2_summary = {
            "overlap_voxel_count": int(stage2_selector.sum().item()),
            "new_voxel_count": int(stage2_selector.shape[0] - stage2_selector.sum().item()),
        }

        primary_glb = None
        free_glb = None
        for variant in STAGE2_VARIANTS:
            variant_dir = image_p2p.ensure_dir(out_dir / variant)
            if variant == "preserve_uniedit":
                selector = stage2_selector
                mode = "preserve_overlap"
            else:
                selector = None
                mode = "target_only"

            slat_tgt = denoise_slat_variant(
                pipeline=pipeline,
                source_cond=source_cond,
                target_cond=edit_cond,
                terminal_noise=projected_slat_noise,
                params=slat_forward_params,
                cfg_interval=cfg_interval,
                omega=float(args.slat_omega),
                selector=selector,
                mode=mode,
                verbose=verbose,
            )
            move_models(
                pipeline,
                ["slat_decoder_mesh", "slat_decoder_gs", "slat_decoder_rf"],
                cuda_device,
            )
            outputs = pipeline.decode_slat(slat_tgt, ["mesh", "gaussian", "radiance_field"])
            image_p2p.save_outputs(
                outputs=outputs,
                out_dir=variant_dir,
                skip_render=args.skip_render,
                skip_glb=args.skip_glb,
                skip_ply=args.skip_ply,
            )

            np.save(variant_dir / "coords_stage1.npy", coords_stage1_masked.detach().cpu().numpy())
            image_p2p.save_json(
                variant_dir / "stage2_variant.json",
                {
                    "variant": variant,
                    "mode": mode,
                    "selector_enabled": selector is not None,
                    "selector_overlap_voxel_count": int(stage2_selector.sum().item()),
                    "selector_new_voxel_count": int(stage2_selector.shape[0] - stage2_selector.sum().item()),
                    "slat_omega": float(args.slat_omega),
                    "slat_forward_params": slat_forward_params,
                },
            )

            generated_glb = variant_dir / "sample_00.glb"
            if variant == "preserve_uniedit":
                primary_glb = generated_glb if generated_glb.is_file() else None
            elif variant == "free_target":
                free_glb = generated_glb if generated_glb.is_file() else None

            move_models(
                pipeline,
                ["slat_decoder_mesh", "slat_decoder_gs", "slat_decoder_rf"],
                torch.device("cpu"),
            )
            del outputs
            del slat_tgt
            release_cuda_memory()

    stage1_stats = summarize_coord_transition(
        coords_source=coords_src,
        coords_stage1_raw=coords_stage1_raw,
        coords_stage1_masked=coords_stage1_masked,
        mask_coords=mask_coords,
    )
    image_p2p.save_json(
        out_dir / "stage1_structure_stats.json",
        {
            **stage1_stats,
            **stage1_meta,
            **stage2_summary,
        },
    )
    np.save(out_dir / "coords_source.npy", coords_src.detach().cpu().numpy())
    np.save(out_dir / "coords_stage1_raw.npy", coords_stage1_raw.detach().cpu().numpy())
    np.save(out_dir / "coords_stage1_masked.npy", coords_stage1_masked.detach().cpu().numpy())

    del voxel_src
    del slat_src
    del ss_terminal_noise
    del slat_terminal_noise
    del projected_slat_noise
    del source_cond
    del edit_cond
    del pipeline
    release_cuda_memory()

    copied_paths = []
    if output_path is not None and not args.skip_glb:
        if primary_glb is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(primary_glb, output_path)
            copied_paths.append(str(output_path))
        if free_glb is not None:
            free_output_path = output_path.with_name(f"{output_path.stem}_free{output_path.suffix}")
            shutil.copy2(free_glb, free_output_path)
            copied_paths.append(str(free_output_path))

    print(f"Saved UniEdit RF-initialized edit results to: {out_dir}")
    for variant in STAGE2_VARIANTS:
        print(f"  - {variant}: {out_dir / variant}")
    if copied_paths:
        print("Copied GLBs:")
        for path in copied_paths:
            print(f"  - {path}")
    print(f"Source voxels: {coords_src.shape[0]}")
    print(f"Stage 1 raw voxels: {coords_stage1_raw.shape[0]}")
    print(f"Stage 1 masked voxels: {coords_stage1_masked.shape[0]}")
    print(f"Stage 2 overlap voxels: {int(stage2_selector.sum().item())}")
    print(f"Stage 2 new voxels: {int(stage2_selector.shape[0] - stage2_selector.sum().item())}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
