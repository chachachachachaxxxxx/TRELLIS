#!/usr/bin/env python3
from __future__ import annotations
"""
TRELLIS image editing by post-hoc SLAT block fusion.

Pipeline:
1. Encode a reference 3D asset from its saved `features.npz` into `source_slat`.
2. Generate a fresh target structure and target SLAT directly from a new image.
3. Compose the final SLAT on the target coordinates:
   - overlapping voxels reuse the source SLAT feature block
   - target-only voxels keep the newly generated target SLAT feature block
   - source-only voxels are dropped so the final structure follows the new image

This keeps TRELLIS core code untouched and implements the behavior as a
standalone experimental example.
"""

import argparse
import gc
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image

from output_layout import build_output_layout
import example_image_prompt_to_prompt as image_p2p
import example_image_prompt_to_prompt_rf_inversion as rf_utils


_attn_backend = image_p2p._peek_arg("--attn-backend", "")
if _attn_backend:
    os.environ["ATTN_BACKEND"] = _attn_backend

os.environ.setdefault("SPCONV_ALGO", "native")


EDIT_METHOD_NAME = "image_slat_xor_fusion"
DEFAULT_EDIT_IMAGE_NAME = "2d_edit.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose a new TRELLIS SLAT by reusing source overlap blocks and target-only edit blocks."
    )
    parser.add_argument("--model", default="microsoft/TRELLIS-image-large", help="Pipeline checkpoint or HF repo.")
    parser.add_argument(
        "--render_dir",
        default="",
        help="Source asset directory containing at least features.npz. This matches existing TRELLIS render/export folders.",
    )
    parser.add_argument(
        "--source-model",
        default="",
        help="Source asset directory, or any file inside it. The directory must contain features.npz.",
    )
    parser.add_argument("--edit-image", default="", help="Edited target image path.")
    parser.add_argument(
        "--image_dir",
        default="",
        help=f"Optional directory containing {DEFAULT_EDIT_IMAGE_NAME}.",
    )
    parser.add_argument(
        "--output_path",
        default="",
        help="Optional primary GLB path. The script still writes its normal output directory.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of target samples to generate.")
    parser.add_argument("--ss-steps", type=int, default=None, help="Override sparse-structure sampling steps.")
    parser.add_argument("--slat-steps", type=int, default=None, help="Override SLAT sampling steps.")
    parser.add_argument("--ss-cfg", type=float, default=None, help="Override sparse-structure CFG strength.")
    parser.add_argument("--slat-cfg", type=float, default=None, help="Override SLAT CFG strength.")
    parser.set_defaults(preprocess=True)
    parser.add_argument(
        "--preprocess",
        dest="preprocess",
        action="store_true",
        help="Apply TRELLIS foreground crop/remove-background preprocessing to the edit image before generation.",
    )
    parser.add_argument(
        "--no-preprocess",
        dest="preprocess",
        action="store_false",
        help="Skip TRELLIS preprocessing and only use the raw edit image as conditioning input.",
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


def resolve_paths(args: argparse.Namespace) -> Dict[str, Path | None]:
    render_dir = None
    image_dir = None
    if args.render_dir:
        render_dir = rf_utils.ensure_path_exists(Path(args.render_dir).expanduser().resolve(), "render_dir")
    if args.image_dir:
        image_dir = rf_utils.resolve_image_dir(args.image_dir)

    if render_dir is not None:
        asset_dir = render_dir
    elif args.source_model:
        asset_dir = rf_utils.resolve_asset_dir(args.source_model)
    else:
        raise RuntimeError("Please provide either --render_dir or --source-model.")

    features_path = asset_dir / "features.npz"
    if not features_path.is_file():
        raise RuntimeError(
            "source-model/render_dir must contain features.npz so the reference asset can be encoded back into SLAT."
        )

    if args.edit_image:
        edit_image_path = rf_utils.ensure_path_exists(Path(args.edit_image).expanduser().resolve(), "edit-image")
    elif image_dir is not None and rf_utils.candidate_file(image_dir / DEFAULT_EDIT_IMAGE_NAME) is not None:
        edit_image_path = image_dir / DEFAULT_EDIT_IMAGE_NAME
    else:
        raise RuntimeError(
            f"Please provide --edit-image, or pass --image_dir containing {DEFAULT_EDIT_IMAGE_NAME}."
        )

    output_path = Path(args.output_path).expanduser().resolve() if args.output_path else None
    if output_path is not None and output_path.suffix.lower() != ".glb":
        raise RuntimeError(f"output_path must end with .glb, got: {output_path}")

    return {
        "asset_dir": asset_dir,
        "render_dir": render_dir,
        "image_dir": image_dir,
        "features_path": features_path,
        "edit_image_path": edit_image_path,
        "output_path": output_path,
    }


def maybe_preprocess_edit_image(pipeline, image: Image.Image, preprocess: bool) -> Image.Image:
    if preprocess:
        return pipeline.preprocess_image(image)
    return image


def coords3d(coords: torch.Tensor) -> torch.Tensor:
    if coords.ndim != 2:
        raise RuntimeError(f"Expected 2D coords tensor, got shape {tuple(coords.shape)}")
    if coords.shape[1] == 4:
        return coords[:, 1:]
    if coords.shape[1] == 3:
        return coords
    raise RuntimeError(f"Unsupported coordinate shape: {tuple(coords.shape)}")


def compose_slat_keep_source_overlap(source_slat, target_slat) -> Tuple[object, dict, np.ndarray]:
    """
    Compose a final SLAT on the target coordinates.

    If a target voxel also exists in the source SLAT, reuse the source feature.
    Otherwise keep the target feature. Source-only voxels are omitted.
    """
    src_batch_count = source_slat.shape[0]
    tgt_batch_count = target_slat.shape[0]
    fused_feats = target_slat.feats.clone()
    reuse_masks = []
    batch_stats = []

    for batch_idx in range(tgt_batch_count):
        tgt_coords_full, tgt_feats = rf_utils.sparse_batch_slice(target_slat, batch_idx)
        src_coords_full, src_feats = rf_utils.sparse_batch_slice(
            source_slat,
            batch_idx if src_batch_count > 1 else 0,
        )

        tgt_codes = rf_utils.coords_to_flat_indices(coords3d(tgt_coords_full))
        src_codes = rf_utils.coords_to_flat_indices(coords3d(src_coords_full))
        batch_reuse = torch.zeros(tgt_codes.shape[0], dtype=torch.bool, device=tgt_codes.device)
        batch_fused = tgt_feats.clone()

        if src_codes.numel() > 0 and tgt_codes.numel() > 0:
            src_codes_sorted, order = torch.sort(src_codes)
            insert_pos = torch.searchsorted(src_codes_sorted, tgt_codes)
            valid = insert_pos < src_codes_sorted.shape[0]
            matched = valid.clone()
            matched[valid] = src_codes_sorted[insert_pos[valid]] == tgt_codes[valid]
            if matched.any():
                matched_src_order = order[insert_pos[matched]]
                batch_fused[matched] = src_feats[matched_src_order]
                batch_reuse[matched] = True

        fused_feats[target_slat.layout[batch_idx]] = batch_fused
        reuse_masks.append(batch_reuse.detach().cpu())

        source_only = int((~torch.isin(src_codes, tgt_codes)).sum().item())
        overlap = int(batch_reuse.sum().item())
        target_only = int((~batch_reuse).sum().item())
        batch_stats.append(
            {
                "batch_idx": batch_idx,
                "source_voxel_count": int(src_codes.shape[0]),
                "target_voxel_count": int(tgt_codes.shape[0]),
                "overlap_reused_from_source_count": overlap,
                "target_only_generated_count": target_only,
                "source_only_dropped_count": source_only,
                "symmetric_difference_count": source_only + target_only,
            }
        )

    fused_slat = target_slat.replace(fused_feats)
    reuse_mask = torch.cat(reuse_masks, dim=0).numpy().astype(np.uint8)
    stats = {
        "composition_rule": "target_coords_only__reuse_source_on_overlap__keep_target_on_target_only",
        "source_batch_count": int(src_batch_count),
        "target_batch_count": int(tgt_batch_count),
        "source_batch_broadcasted": bool(src_batch_count == 1 and tgt_batch_count > 1),
        "fused_total_voxel_count": int(fused_slat.coords.shape[0]),
        "reused_source_voxel_count_total": int(reuse_mask.sum().item()),
        "generated_target_voxel_count_total": int(reuse_mask.shape[0] - reuse_mask.sum().item()),
        "batch_stats": batch_stats,
    }
    return fused_slat, stats, reuse_mask


def build_sampler_params(base_params: dict, steps_override: int | None, cfg_override: float | None) -> dict:
    params = dict(base_params)
    if steps_override is not None:
        params["steps"] = int(steps_override)
    if cfg_override is not None:
        params["cfg_strength"] = float(cfg_override)
    return params


def main() -> int:
    args = parse_args()
    backend = image_p2p.configure_attention_backend(args.attn_backend)
    resolved_paths = resolve_paths(args)

    from trellis.modules.sparse.basic import SparseTensor as _SparseTensor
    from trellis.pipelines import TrellisImageTo3DPipeline as _TrellisImageTo3DPipeline

    image_p2p.SparseTensor = _SparseTensor
    image_p2p.TrellisImageTo3DPipeline = _TrellisImageTo3DPipeline

    asset_dir = resolved_paths["asset_dir"]
    features_path = resolved_paths["features_path"]
    edit_image_path = resolved_paths["edit_image_path"]
    output_path = resolved_paths["output_path"]

    pipeline = _TrellisImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()

    edit_image_raw = Image.open(edit_image_path)
    edit_image = maybe_preprocess_edit_image(pipeline, edit_image_raw, bool(args.preprocess))
    edit_cond = pipeline.get_cond([edit_image])

    ss_params = build_sampler_params(
        pipeline.sparse_structure_sampler_params,
        args.ss_steps,
        args.ss_cfg,
    )
    slat_params = build_sampler_params(
        pipeline.slat_sampler_params,
        args.slat_steps,
        args.slat_cfg,
    )

    case_name = args.case_name.strip()
    if not case_name:
        if output_path is not None:
            case_name = image_p2p.slugify(output_path.stem)
        else:
            case_name = image_p2p.slugify(f"{asset_dir.name}_xor_{edit_image_path.stem}")
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
            "render_dir": str(resolved_paths["render_dir"]) if resolved_paths["render_dir"] is not None else None,
            "image_dir": str(resolved_paths["image_dir"]) if resolved_paths["image_dir"] is not None else None,
            "features_npz": str(features_path),
            "edit_image": str(edit_image_path),
            "output_path": str(output_path) if output_path is not None else None,
            "seed": int(args.seed),
            "num_samples": int(args.num_samples),
            "preprocess": bool(args.preprocess),
            "sparse_structure_sampler_params": ss_params,
            "slat_sampler_params": slat_params,
            "composition_rule": "reuse source SLAT on overlap, keep target SLAT on target-only voxels, drop source-only voxels",
        },
    )
    edit_image.save(out_dir / "edit_condition_input.png")

    torch.manual_seed(args.seed)

    with torch.no_grad():
        source_slat = rf_utils.feats_to_slat(pipeline, features_path)
        coords_target = pipeline.sample_sparse_structure(
            edit_cond,
            num_samples=int(args.num_samples),
            sampler_params=ss_params,
        )
        target_slat = pipeline.sample_slat(
            edit_cond,
            coords_target,
            sampler_params=slat_params,
        )
        fused_slat, fusion_stats, reuse_mask = compose_slat_keep_source_overlap(source_slat, target_slat)
        outputs = pipeline.decode_slat(fused_slat, ["mesh", "gaussian", "radiance_field"])

    np.save(
        out_dir / "source_slat_coords.npy",
        source_slat.coords.detach().cpu().numpy().astype(np.int16),
    )
    np.save(
        out_dir / "target_slat_coords.npy",
        target_slat.coords.detach().cpu().numpy().astype(np.int16),
    )
    np.save(
        out_dir / "fused_slat_coords.npy",
        fused_slat.coords.detach().cpu().numpy().astype(np.int16),
    )
    np.save(
        out_dir / "fused_reuse_source_mask.npy",
        reuse_mask,
    )
    image_p2p.save_json(out_dir / "fusion_stats.json", fusion_stats)

    del source_slat
    del target_slat
    del fused_slat
    del coords_target
    del edit_cond
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

    print(f"Saved SLAT XOR-style fusion results to: {out_dir}")
    print(f"Source asset: {asset_dir}")
    print(f"Edit image: {edit_image_path}")
    print(f"Fused voxels: {fusion_stats['fused_total_voxel_count']}")
    print(f"Reused source voxels: {fusion_stats['reused_source_voxel_count_total']}")
    print(f"Generated target voxels: {fusion_stats['generated_target_voxel_count_total']}")
    if output_path is not None and not args.skip_glb:
        print(f"Copied primary GLB to: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
