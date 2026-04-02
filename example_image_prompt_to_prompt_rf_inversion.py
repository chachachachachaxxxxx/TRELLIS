#!/usr/bin/env python3
from __future__ import annotations
"""
TRELLIS image Prompt-to-Prompt editing with RF inversion initialization.

This script keeps TRELLIS core code untouched and combines:
1. RF-style second-order inversion from a source 3D asset to terminal noise.
2. TRELLIS image Prompt-to-Prompt cross-attention injection during denoising.

It is intentionally narrower than VoxHammer:
- No latent replacement.
- No KV replacement.
- No local 3D edit masks inside the denoising loop.
- Prompt-to-Prompt cross-attention injection is enabled only for forward denoising.

Required source asset files:
- voxels.ply
- features.npz
- a source render image such as 2d_render.png (or pass --source-image explicitly)

Usage examples:
  python example_image_prompt_to_prompt_rf_inversion.py \
    --render_dir outputs/example/render \
    --image_dir assets/example/images \
    --output_path outputs/example/output.glb

  python example_image_prompt_to_prompt_rf_inversion.py \
    --source-model assets/example_edit/source_asset \
    --edit-image assets/example_edit/2d_edit.png \
    --source-image assets/example_edit/2d_render.png

python example_image_prompt_to_prompt_rf_inversion.py \
  --input_model assets/example/model.glb \
  --mask_glb assets/example/mask.glb \
  --render_dir outputs/rf_p2p/render \
  --image_dir outputs/rf_p2p/images \
  --output_path outputs/rf_p2p/output.glb

"""

import argparse
import gc
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import trimesh
from PIL import Image, ImageFilter
from tqdm import tqdm

from output_layout import build_output_layout
import example_image_prompt_to_prompt as image_p2p


_attn_backend = image_p2p._peek_arg("--attn-backend", "")
if _attn_backend:
    os.environ["ATTN_BACKEND"] = _attn_backend

os.environ.setdefault("SPCONV_ALGO", "native")


SOURCE_RENDER_CANDIDATES = (
    "2d_render.png",
    "render.png",
    "source_render.png",
    "source.png",
    "input.png",
)

EDIT_METHOD_NAME = "image_prompt_to_prompt_rf_inversion"


def ensure_path_exists(path: Path, label: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{label} does not exist: {path}")
    return path


def resolve_asset_dir(source_model: str) -> Path:
    source_path = ensure_path_exists(Path(source_model).expanduser().resolve(), "source-model")
    if source_path.is_dir():
        return source_path
    return source_path.parent


def resolve_source_image_path(asset_dir: Path, explicit_path: str) -> Path:
    if explicit_path:
        return ensure_path_exists(Path(explicit_path).expanduser().resolve(), "source-image")

    for name in SOURCE_RENDER_CANDIDATES:
        candidate = asset_dir / name
        if candidate.is_file():
            return candidate

    for name in SOURCE_RENDER_CANDIDATES:
        matches = sorted(asset_dir.rglob(name))
        if matches:
            return matches[0]

    raise RuntimeError(
        "Could not find a source render image automatically. "
        f"Expected one of {list(SOURCE_RENDER_CANDIDATES)} inside {asset_dir}. "
        "Please pass --source-image explicitly."
    )


def validate_required_asset_files(asset_dir: Path) -> Tuple[Path, Path]:
    voxels_path = asset_dir / "voxels.ply"
    features_path = asset_dir / "features.npz"
    missing = [str(path.name) for path in (voxels_path, features_path) if not path.is_file()]
    if missing:
        raise RuntimeError(
            "source-model must point to an asset directory (or a file inside it) containing "
            f"{missing}. A plain TRELLIS export like sample.ply/sample.glb is not enough for RF inversion."
        )
    return voxels_path, features_path


def resolve_image_dir(image_dir: str) -> Path:
    return ensure_path_exists(Path(image_dir).expanduser().resolve(), "image_dir")


def candidate_file(path: Path) -> Optional[Path]:
    return path if path.is_file() else None


def resolve_pipeline_paths(args: argparse.Namespace) -> dict:
    render_dir = None
    image_dir = None
    if args.render_dir:
        render_dir = ensure_path_exists(Path(args.render_dir).expanduser().resolve(), "render_dir")
    if args.image_dir:
        image_dir = resolve_image_dir(args.image_dir)

    if render_dir is not None:
        asset_dir = render_dir
    elif args.source_model:
        asset_dir = resolve_asset_dir(args.source_model)
    else:
        raise RuntimeError("Please provide either --render_dir or --source-model.")

    voxels_path, features_path = validate_required_asset_files(asset_dir)

    if args.source_image:
        source_image_path = ensure_path_exists(Path(args.source_image).expanduser().resolve(), "source-image")
    elif image_dir is not None and candidate_file(image_dir / "2d_render.png") is not None:
        source_image_path = image_dir / "2d_render.png"
    else:
        source_image_path = resolve_source_image_path(asset_dir, "")

    if args.edit_image:
        edit_image_path = ensure_path_exists(Path(args.edit_image).expanduser().resolve(), "edit-image")
    elif image_dir is not None and candidate_file(image_dir / "2d_edit.png") is not None:
        edit_image_path = image_dir / "2d_edit.png"
    else:
        raise RuntimeError("Please provide --edit-image, or pass --image_dir containing 2d_edit.png.")

    if args.mask_image:
        mask_image_path = ensure_path_exists(Path(args.mask_image).expanduser().resolve(), "mask-image")
    elif image_dir is not None:
        mask_image_path = candidate_file(image_dir / "2d_mask.png")
    else:
        mask_image_path = None

    output_path = Path(args.output_path).expanduser().resolve() if args.output_path else None
    if output_path is not None and output_path.suffix.lower() != ".glb":
        raise RuntimeError(f"output_path must end with .glb, got: {output_path}")
    return {
        "asset_dir": asset_dir,
        "render_dir": render_dir,
        "image_dir": image_dir,
        "voxels_path": voxels_path,
        "features_path": features_path,
        "source_image_path": source_image_path,
        "edit_image_path": edit_image_path,
        "mask_image_path": mask_image_path,
        "output_path": output_path,
    }


def build_blank_mask(size: Tuple[int, int]) -> Image.Image:
    return Image.new("L", size, color=0)


def build_auto_mask(
    source_image: Image.Image,
    edit_image: Image.Image,
    threshold: int,
    max_filter: int,
) -> Image.Image:
    src = np.asarray(source_image.convert("RGB"), dtype=np.int16)
    tgt = np.asarray(edit_image.convert("RGB"), dtype=np.int16)
    diff = np.max(np.abs(src - tgt), axis=-1)
    mask = (diff >= int(threshold)).astype(np.uint8) * 255
    mask_image = Image.fromarray(mask, mode="L")
    filter_size = max(1, int(max_filter))
    if filter_size % 2 == 0:
        filter_size += 1
    if filter_size > 1:
        mask_image = mask_image.filter(ImageFilter.MaxFilter(size=filter_size))
    return mask_image


def load_ply_positions(ply_path: Path) -> np.ndarray:
    try:
        import utils3d  # type: ignore

        position = utils3d.io.read_ply(str(ply_path))[0]
        return np.asarray(position, dtype=np.float32)
    except Exception:
        mesh = trimesh.load(str(ply_path), process=False)
        if hasattr(mesh, "vertices"):
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
        elif isinstance(mesh, trimesh.points.PointCloud):
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
        else:
            raise RuntimeError(f"Unsupported PLY payload in {ply_path}")
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise RuntimeError(f"Expected Nx3 vertices in {ply_path}, got shape {vertices.shape}")
        return vertices


def ply_to_coords(ply_path: Path, device: torch.device) -> torch.Tensor:
    position = load_ply_positions(ply_path)
    coords = ((torch.from_numpy(position) + 0.5) * 64).int()
    coords = torch.clamp(coords, min=0, max=63)
    coords = torch.unique(coords, dim=0).contiguous()
    return coords.to(device=device)


def coords_to_voxel(coords: torch.Tensor, device: torch.device) -> torch.Tensor:
    voxel = torch.zeros(1, 1, 64, 64, 64, dtype=torch.float32, device=device)
    voxel[:, 0, coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
    return voxel


def feats_to_slat(pipeline, feats_path: Path):
    feats = np.load(feats_path)
    if "patchtokens" not in feats or "indices" not in feats:
        raise RuntimeError(
            f"features.npz must contain 'patchtokens' and 'indices', got keys: {list(feats.keys())}"
        )
    sparse_tensor = image_p2p.SparseTensor(
        feats=torch.from_numpy(feats["patchtokens"]).float().to(pipeline.device),
        coords=torch.cat(
            [
                torch.zeros(feats["patchtokens"].shape[0], 1, dtype=torch.int32),
                torch.from_numpy(feats["indices"]).int(),
            ],
            dim=1,
        ).to(pipeline.device),
    )
    feats_encoder = pipeline.models["slat_encoder"]
    return feats_encoder(sparse_tensor, sample_posterior=False)


def coords_to_flat_indices(coords: torch.Tensor, resolution: int = 64) -> torch.Tensor:
    coords = coords.long()
    if coords.shape[1] == 3:
        return coords[:, 0] * resolution * resolution + coords[:, 1] * resolution + coords[:, 2]
    if coords.shape[1] == 4:
        return (
            coords[:, 0] * resolution * resolution * resolution
            + coords[:, 1] * resolution * resolution
            + coords[:, 2] * resolution
            + coords[:, 3]
        )
    raise RuntimeError(f"Unsupported coordinate shape: {tuple(coords.shape)}")


def sparse_batch_slice(sparse_tensor, batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    slc = sparse_tensor.layout[batch_idx]
    return sparse_tensor.coords[slc], sparse_tensor.feats[slc]


def project_sparse_terminal_noise(source_noise, target_coords: torch.Tensor, device: torch.device):
    if target_coords.numel() == 0:
        raise RuntimeError("Sparse-structure denoising produced no target voxels, so SLAT projection cannot continue.")
    batch_size = int(target_coords[:, 0].max().item()) + 1 if target_coords.numel() > 0 else 1
    feature_dim = source_noise.feats.shape[1]
    all_coords = []
    all_feats = []
    src_batch_count = source_noise.shape[0]

    for batch_idx in range(batch_size):
        tgt_mask = target_coords[:, 0] == batch_idx
        tgt_coords_batch = target_coords[tgt_mask]
        if tgt_coords_batch.shape[0] == 0:
            continue

        src_coords_full, src_feats_full = sparse_batch_slice(
            source_noise, batch_idx if src_batch_count > 1 else 0
        )
        src_coords = src_coords_full[:, 1:]
        tgt_coords = tgt_coords_batch[:, 1:]

        src_codes = coords_to_flat_indices(src_coords)
        tgt_codes = coords_to_flat_indices(tgt_coords)
        randn_feats = torch.randn(
            tgt_coords.shape[0],
            feature_dim,
            device=device,
            dtype=source_noise.feats.dtype,
        )

        if src_codes.numel() > 0:
            src_codes_sorted, order = torch.sort(src_codes)
            insert_pos = torch.searchsorted(src_codes_sorted, tgt_codes)
            valid = insert_pos < src_codes_sorted.shape[0]
            matched = valid.clone()
            matched[valid] = src_codes_sorted[insert_pos[valid]] == tgt_codes[valid]
            if matched.any():
                matched_order = order[insert_pos[matched]]
                randn_feats[matched] = src_feats_full[matched_order]

        all_coords.append(tgt_coords_batch)
        all_feats.append(randn_feats)

    coords = torch.cat(all_coords, dim=0).to(device=device)
    feats = torch.cat(all_feats, dim=0).to(device=device)
    return image_p2p.SparseTensor(feats=feats, coords=coords)


def get_slat_norm_tensors(pipeline, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    std = torch.tensor(pipeline.slat_normalization["std"], device=device, dtype=dtype)[None]
    mean = torch.tensor(pipeline.slat_normalization["mean"], device=device, dtype=dtype)[None]
    return mean, std


class SecondOrderRFSampler:
    """
    Taylor-improved second-order flow sampler inspired by VoxHammer's RF-Solver.

    This version intentionally keeps the original TRELLIS model forward signature
    and only adds:
    - inverse sampling (data -> terminal noise)
    - second-order midpoint correction
    - late-time CFG on t in [cfg_interval[0], cfg_interval[1]]
    """

    def _run_model(self, model, sample, t_value: float, cond: Optional[torch.Tensor]):
        t = torch.tensor([1000.0 * t_value] * sample.shape[0], device=sample.device, dtype=torch.float32)
        if cond is not None and cond.shape[0] == 1 and sample.shape[0] > 1:
            cond = cond.repeat(sample.shape[0], *([1] * (cond.ndim - 1)))
        return model(sample, t, cond)

    def _guided_prediction(
        self,
        model,
        sample,
        t_value: float,
        cond_dict: dict,
        cfg_strength: float,
        cfg_interval: Tuple[float, float],
    ):
        if cfg_interval[0] <= t_value <= cfg_interval[1] and cfg_strength > 0.0:
            pred = self._run_model(model, sample, t_value, cond_dict["cond"])
            neg_pred = self._run_model(model, sample, t_value, cond_dict["neg_cond"])
            return (1.0 + cfg_strength) * pred - cfg_strength * neg_pred
        return self._run_model(model, sample, t_value, cond_dict["cond"])

    def sample_once(
        self,
        model,
        sample,
        t_curr: float,
        t_next: float,
        cond_dict: dict,
        cfg_strength: float,
        cfg_interval: Tuple[float, float],
    ):
        pred = self._guided_prediction(model, sample, t_curr, cond_dict, cfg_strength, cfg_interval)
        dt = t_next - t_curr
        sample_mid = sample + 0.5 * dt * pred
        t_mid = t_curr + 0.5 * dt
        pred_mid = self._guided_prediction(model, sample_mid, t_mid, cond_dict, cfg_strength, cfg_interval)
        first_order = (pred_mid - pred) / (0.5 * dt)
        return sample + dt * pred - 0.5 * (dt ** 2) * first_order

    def sample(
        self,
        model,
        sample,
        cond_dict: dict,
        steps: int,
        rescale_t: float,
        cfg_strength: float,
        cfg_interval: Tuple[float, float],
        inverse: bool,
        verbose: bool = True,
    ):
        t_seq = np.linspace(1.0, 0.0, int(steps) + 1)
        t_seq = rescale_t * t_seq / (1.0 + (rescale_t - 1.0) * t_seq)
        if inverse:
            t_seq = t_seq[::-1]
            desc = "RF inversion"
        else:
            desc = "RF denoise"
        t_pairs = list((float(t_seq[i]), float(t_seq[i + 1])) for i in range(len(t_seq) - 1))
        for t_curr, t_next in tqdm(t_pairs, desc=desc, disable=not verbose):
            sample = self.sample_once(model, sample, t_curr, t_next, cond_dict, cfg_strength, cfg_interval)
        return sample


def resolve_stage_sampling_params(
    base_params: dict,
    steps_override: Optional[int],
    forward_cfg_override: Optional[float],
    inverse_cfg_override: Optional[float],
) -> Tuple[dict, dict]:
    steps = int(steps_override if steps_override is not None else base_params.get("steps", 50))
    rescale_t = float(base_params.get("rescale_t", 1.0))
    base_cfg = float(base_params.get("cfg_strength", 3.0))
    forward_cfg = float(forward_cfg_override) if forward_cfg_override is not None else base_cfg
    if inverse_cfg_override is not None:
        inverse_cfg = float(inverse_cfg_override)
    elif forward_cfg_override is not None:
        inverse_cfg = float(forward_cfg_override)
    else:
        inverse_cfg = base_cfg
    inverse_params = {"steps": steps, "rescale_t": rescale_t, "cfg_strength": inverse_cfg}
    forward_params = {"steps": steps, "rescale_t": rescale_t, "cfg_strength": forward_cfg}
    return inverse_params, forward_params


def invert_sparse_structure(
    pipeline,
    cond_src: dict,
    voxel_src: torch.Tensor,
    params: dict,
    cfg_interval: Tuple[float, float],
    verbose: bool,
) -> torch.Tensor:
    encoder = pipeline.models["sparse_structure_encoder"]
    flow_model = pipeline.models["sparse_structure_flow_model"]
    z_src = encoder(voxel_src)
    sampler = SecondOrderRFSampler()
    return sampler.sample(
        model=flow_model,
        sample=z_src,
        cond_dict=cond_src,
        steps=params["steps"],
        rescale_t=params["rescale_t"],
        cfg_strength=params["cfg_strength"],
        cfg_interval=cfg_interval,
        inverse=True,
        verbose=verbose,
    )


def denoise_sparse_structure(
    pipeline,
    cond_edit: dict,
    terminal_noise: torch.Tensor,
    params: dict,
    cfg_interval: Tuple[float, float],
    verbose: bool,
) -> torch.Tensor:
    flow_model = pipeline.models["sparse_structure_flow_model"]
    decoder = pipeline.models["sparse_structure_decoder"]
    sampler = SecondOrderRFSampler()
    z_tgt = sampler.sample(
        model=flow_model,
        sample=terminal_noise,
        cond_dict=cond_edit,
        steps=params["steps"],
        rescale_t=params["rescale_t"],
        cfg_strength=params["cfg_strength"],
        cfg_interval=cfg_interval,
        inverse=False,
        verbose=verbose,
    )
    voxel = decoder(z_tgt)
    coords = torch.argwhere(voxel > 0)[:, [0, 2, 3, 4]].int()
    if coords.shape[0] == 0:
        raise RuntimeError("Sparse-structure denoising produced an empty target structure.")
    return coords


def invert_slat(
    pipeline,
    cond_src: dict,
    slat_src,
    params: dict,
    cfg_interval: Tuple[float, float],
    verbose: bool,
):
    flow_model = pipeline.models["slat_flow_model"]
    mean, std = get_slat_norm_tensors(pipeline, slat_src.device, slat_src.feats.dtype)
    slat_normalized = (slat_src - mean) / std
    sampler = SecondOrderRFSampler()
    return sampler.sample(
        model=flow_model,
        sample=slat_normalized,
        cond_dict=cond_src,
        steps=params["steps"],
        rescale_t=params["rescale_t"],
        cfg_strength=params["cfg_strength"],
        cfg_interval=cfg_interval,
        inverse=True,
        verbose=verbose,
    )


def denoise_slat(
    pipeline,
    cond_edit: dict,
    terminal_noise,
    params: dict,
    cfg_interval: Tuple[float, float],
    verbose: bool,
):
    flow_model = pipeline.models["slat_flow_model"]
    sampler = SecondOrderRFSampler()
    slat_normalized = sampler.sample(
        model=flow_model,
        sample=terminal_noise,
        cond_dict=cond_edit,
        steps=params["steps"],
        rescale_t=params["rescale_t"],
        cfg_strength=params["cfg_strength"],
        cfg_interval=cfg_interval,
        inverse=False,
        verbose=verbose,
    )
    mean, std = get_slat_norm_tensors(pipeline, slat_normalized.device, slat_normalized.feats.dtype)
    return slat_normalized * std + mean


def stage_configs_from_args(args: argparse.Namespace, inject_stages: Sequence[str]) -> Dict[str, image_p2p.StageConfig]:
    return {
        "sparse_structure": image_p2p.StageConfig(
            name="sparse_structure",
            enabled="sparse_structure" in inject_stages,
            t_start=float(args.ss_t_start),
            t_end=float(args.ss_t_end),
            strength=float(args.ss_strength),
        ),
        "slat": image_p2p.StageConfig(
            name="slat",
            enabled="slat" in inject_stages,
            t_start=float(args.slat_t_start),
            t_end=float(args.slat_t_end),
            strength=float(args.slat_strength),
        ),
    }


def prepare_inputs_with_optional_auto_mask(
    pipeline,
    source_image: Image.Image,
    edit_image: Image.Image,
    mask_image: Optional[Image.Image],
    preprocess: bool,
    mask_threshold: int,
    auto_mask_threshold: int,
    auto_mask_max_filter: int,
):
    raw_mask = mask_image if mask_image is not None else build_blank_mask(source_image.size)
    prepared = image_p2p.prepare_aligned_inputs(
        source_image=source_image,
        edit_image=edit_image,
        mask_image=raw_mask,
        pipeline=pipeline,
        preprocess=preprocess,
        mask_threshold=mask_threshold,
    )

    if mask_image is not None:
        meta = dict(prepared.meta)
        meta["mask_source"] = "provided"
        return image_p2p.PreparedInputs(source=prepared.source, edit=prepared.edit, mask=prepared.mask, meta=meta)

    auto_mask = build_auto_mask(
        source_image=prepared.source,
        edit_image=prepared.edit,
        threshold=auto_mask_threshold,
        max_filter=auto_mask_max_filter,
    )
    meta = dict(prepared.meta)
    meta["mask_source"] = "auto_diff"
    meta["auto_mask_threshold"] = int(auto_mask_threshold)
    meta["auto_mask_max_filter"] = int(auto_mask_max_filter)
    meta["auto_mask_rule"] = "max_rgb_abs_diff_on_preprocessed_source_and_edit"
    return image_p2p.PreparedInputs(source=prepared.source, edit=prepared.edit, mask=auto_mask, meta=meta)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TRELLIS image Prompt-to-Prompt editing with RF inversion initialization."
    )
    parser.add_argument("--model", default="microsoft/TRELLIS-image-large", help="Pipeline checkpoint or HF repo.")
    parser.add_argument("--input_model", default="", help="Compatibility arg matching VoxHammer inference.py.")
    parser.add_argument("--mask_glb", default="", help="Compatibility arg matching VoxHammer inference.py.")
    parser.add_argument("--render_dir", default="", help="Render directory containing voxels.ply and features.npz.")
    parser.add_argument(
        "--output_path",
        default="",
        help="Optional final GLB path, matching VoxHammer inference.py. The script still writes its normal output directory.",
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
        help="Optional binary/grayscale edit mask path. If omitted, a mask is auto-generated from source/edit differences.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument(
        "--inject-stages",
        default="st,slat",
        help="Comma-separated stages to inject: st/ss/sparse_structure, slat, or none.",
    )
    parser.add_argument("--ss-t-start", type=float, default=1.0, help="Sparse-structure injection start t in [0, 1].")
    parser.add_argument("--ss-t-end", type=float, default=0.5, help="Sparse-structure injection end t in [0, 1].")
    parser.add_argument("--slat-t-start", type=float, default=1.0, help="SLat injection start t in [0, 1].")
    parser.add_argument("--slat-t-end", type=float, default=0.5, help="SLat injection end t in [0, 1].")
    parser.add_argument("--ss-strength", type=float, default=1.0, help="Sparse-structure P2P replacement strength.")
    parser.add_argument("--slat-strength", type=float, default=1.0, help="SLat P2P replacement strength.")
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
    parser.add_argument("--query-chunk", type=int, default=1024, help="Query chunk size for patched attention.")
    parser.add_argument("--mask-threshold", type=int, default=127, help="Mask pixel threshold in [0, 255].")
    parser.add_argument(
        "--auto-mask-threshold",
        type=int,
        default=24,
        help="Pixel difference threshold used when auto-generating the mask.",
    )
    parser.add_argument(
        "--auto-mask-max-filter",
        type=int,
        default=7,
        help="Odd max-filter size used to thicken the auto-generated mask. Use 1 to disable.",
    )
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

    from trellis.modules.attention.modules import MultiHeadAttention as _MultiHeadAttention
    from trellis.modules.sparse.attention.modules import SparseMultiHeadAttention as _SparseMultiHeadAttention
    from trellis.modules.sparse.basic import SparseTensor as _SparseTensor
    from trellis.pipelines import TrellisImageTo3DPipeline as _TrellisImageTo3DPipeline

    image_p2p.MultiHeadAttention = _MultiHeadAttention
    image_p2p.SparseMultiHeadAttention = _SparseMultiHeadAttention
    image_p2p.SparseTensor = _SparseTensor
    image_p2p.TrellisImageTo3DPipeline = _TrellisImageTo3DPipeline

    inject_stages = image_p2p.parse_stage_list(args.inject_stages)
    stage_configs = stage_configs_from_args(args, inject_stages)
    cfg_interval = (float(args.cfg_interval_start), float(args.cfg_interval_end))
    if cfg_interval[0] > cfg_interval[1]:
        raise RuntimeError(
            f"cfg-interval-start must be <= cfg-interval-end, got {cfg_interval[0]} > {cfg_interval[1]}"
        )

    resolved_paths = resolve_pipeline_paths(args)
    asset_dir = resolved_paths["asset_dir"]
    voxels_path = resolved_paths["voxels_path"]
    features_path = resolved_paths["features_path"]
    source_image_path = resolved_paths["source_image_path"]
    edit_image_path = resolved_paths["edit_image_path"]
    mask_image_path = resolved_paths["mask_image_path"]
    output_path = resolved_paths["output_path"]

    pipeline = _TrellisImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()

    source_image = Image.open(source_image_path)
    edit_image = Image.open(edit_image_path)
    mask_image = Image.open(mask_image_path) if mask_image_path is not None else None

    prepared = prepare_inputs_with_optional_auto_mask(
        pipeline=pipeline,
        source_image=source_image,
        edit_image=edit_image,
        mask_image=mask_image,
        preprocess=bool(args.preprocess),
        mask_threshold=int(args.mask_threshold),
        auto_mask_threshold=int(args.auto_mask_threshold),
        auto_mask_max_filter=int(args.auto_mask_max_filter),
    )

    source_cond = pipeline.get_cond([prepared.source])
    edit_cond = pipeline.get_cond([prepared.edit])

    patch_size = image_p2p.resolve_patch_size(pipeline.models["image_cond_model"].patch_size)
    token_meta = image_p2p.build_image_token_metadata(
        cond=edit_cond["cond"],
        mask=prepared.mask,
        patch_size=patch_size,
        patch_coverage_threshold=float(args.patch_coverage_threshold),
    )
    if not token_meta["edited_token_indices"]:
        print("Warning: the mask selected no visual patch tokens; cross-attention injection will be effectively disabled.")
    if not token_meta["source_keep_indices"]:
        print("Warning: the mask covers all visual patch tokens; injection becomes equivalent to using the edit prompt only.")

    editor = image_p2p.ImagePromptToPromptEditor(
        source_cond=source_cond["cond"],
        edit_cond=edit_cond["cond"],
        neg_cond=edit_cond["neg_cond"],
        token_meta=token_meta,
        stage_configs=stage_configs,
        query_chunk=args.query_chunk,
    )

    ss_inverse_params, ss_forward_params = resolve_stage_sampling_params(
        pipeline.sparse_structure_sampler_params,
        args.ss_steps,
        args.ss_cfg,
        args.ss_inverse_cfg,
    )
    slat_inverse_params, slat_forward_params = resolve_stage_sampling_params(
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
            "mask_glb": args.mask_glb or None,
            "render_dir": str(resolved_paths["render_dir"]) if resolved_paths["render_dir"] is not None else None,
            "image_dir": str(resolved_paths["image_dir"]) if resolved_paths["image_dir"] is not None else None,
            "output_path": str(output_path) if output_path is not None else None,
            "voxels_ply": str(voxels_path),
            "features_npz": str(features_path),
            "source_image": str(source_image_path),
            "edit_image": str(edit_image_path),
            "mask_image": str(mask_image_path) if mask_image_path is not None else None,
            "seed": args.seed,
            "inject_stages": inject_stages,
            "query_chunk": args.query_chunk,
            "mask_threshold": args.mask_threshold,
            "auto_mask_threshold": args.auto_mask_threshold,
            "auto_mask_max_filter": args.auto_mask_max_filter,
            "patch_coverage_threshold": args.patch_coverage_threshold,
            "preprocess": args.preprocess,
            "cfg_interval": list(cfg_interval),
            "stage_configs": {
                name: {
                    "enabled": config.enabled,
                    "t_start": config.t_start,
                    "t_end": config.t_end,
                    "strength": config.strength,
                }
                for name, config in stage_configs.items()
            },
            "sparse_structure_inverse_params": ss_inverse_params,
            "sparse_structure_forward_params": ss_forward_params,
            "slat_inverse_params": slat_inverse_params,
            "slat_forward_params": slat_forward_params,
        },
    )
    image_p2p.save_json(out_dir / "input_preprocess.json", prepared.meta)
    image_p2p.save_json(out_dir / "token_metadata.json", token_meta)

    prepared.source.save(out_dir / "source_preprocessed.png")
    prepared.edit.save(out_dir / "edit_preprocessed.png")
    prepared.mask.save(out_dir / "mask_preprocessed.png")
    image_p2p.save_mask_overlay_preview(
        edit_image=prepared.edit,
        mask_image=prepared.mask,
        token_meta=token_meta,
        path=out_dir / "mask_token_overlay.png",
    )
    image_p2p.save_patch_grid_preview(token_meta, out_dir / "edited_patch_grid.png")

    torch.manual_seed(args.seed)
    verbose = not bool(args.quiet)

    with torch.no_grad():
        coords_src = ply_to_coords(voxels_path, pipeline.device)
        voxel_src = coords_to_voxel(coords_src, pipeline.device)
        slat_src = feats_to_slat(pipeline, features_path)

        ss_terminal_noise = invert_sparse_structure(
            pipeline=pipeline,
            cond_src=source_cond,
            voxel_src=voxel_src,
            params=ss_inverse_params,
            cfg_interval=cfg_interval,
            verbose=verbose,
        )
        slat_terminal_noise = invert_slat(
            pipeline=pipeline,
            cond_src=source_cond,
            slat_src=slat_src,
            params=slat_inverse_params,
            cfg_interval=cfg_interval,
            verbose=verbose,
        )

        torch.cuda.empty_cache()

        try:
            editor.prepare_stage("sparse_structure", pipeline.models["sparse_structure_flow_model"])
            editor.prepare_stage("slat", pipeline.models["slat_flow_model"])

            coords_tgt = denoise_sparse_structure(
                pipeline=pipeline,
                cond_edit=edit_cond,
                terminal_noise=ss_terminal_noise,
                params=ss_forward_params,
                cfg_interval=cfg_interval,
                verbose=verbose,
            )
            projected_slat_noise = project_sparse_terminal_noise(
                source_noise=slat_terminal_noise,
                target_coords=coords_tgt.to(device=pipeline.device),
                device=pipeline.device,
            )
            slat_tgt = denoise_slat(
                pipeline=pipeline,
                cond_edit=edit_cond,
                terminal_noise=projected_slat_noise,
                params=slat_forward_params,
                cfg_interval=cfg_interval,
                verbose=verbose,
            )
            outputs = pipeline.decode_slat(slat_tgt, ["mesh", "gaussian", "radiance_field"])
        finally:
            editor.restore()

    source_codes = coords_to_flat_indices(torch.cat([torch.zeros(coords_src.shape[0], 1, device=coords_src.device, dtype=torch.int32), coords_src], dim=1))
    target_codes = coords_to_flat_indices(coords_tgt)
    overlap_count = int(torch.isin(target_codes, source_codes).sum().item())
    image_p2p.save_json(
        out_dir / "rf_inversion_stats.json",
        {
            "source_voxel_count": int(coords_src.shape[0]),
            "target_voxel_count": int(coords_tgt.shape[0]),
            "target_source_overlap_count": overlap_count,
            "edited_patch_token_count": int(len(token_meta["edited_token_indices"])),
            "kept_patch_token_count": int(len(token_meta["source_keep_indices"])),
        },
    )
    np.save(out_dir / "coords_target.npy", coords_tgt.detach().cpu().numpy())

    # Free the heavy generation pipeline before render/GLB export, which has its own GPU workload.
    del voxel_src
    del slat_src
    del ss_terminal_noise
    del slat_terminal_noise
    del projected_slat_noise
    del slat_tgt
    del source_cond
    del edit_cond
    del editor
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    image_p2p.save_outputs(
        outputs=outputs,
        out_dir=out_dir,
        skip_render=args.skip_render,
        skip_glb=args.skip_glb,
        skip_ply=args.skip_ply,
    )

    if output_path is not None and not args.skip_glb:
        generated_glb = out_dir / "sample_00.glb"
        if generated_glb.is_file():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(generated_glb, output_path)
        else:
            print(f"Warning: expected generated GLB not found at {generated_glb}, so --output_path was not populated.")

    print(f"Saved RF-initialized image Prompt-to-Prompt edit results to: {out_dir}")
    if output_path is not None and not args.skip_glb:
        print(f"Copied primary GLB to: {output_path}")
    print(f"Source voxels: {coords_src.shape[0]}")
    print(f"Target voxels: {coords_tgt.shape[0]}")
    print(f"Projected SLAT overlap voxels: {overlap_count}")
    print(f"Edited patch tokens: {len(token_meta['edited_token_indices'])}")
    print(f"Injected stages: {inject_stages if inject_stages else ['none']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
