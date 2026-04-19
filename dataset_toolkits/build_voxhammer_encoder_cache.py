#!/usr/bin/env python3
"""
Build VoxHammer/TRELLIS encoder cache for the pseudo-source micro datasets.

This reproduces the object-level shared cache layout used by Edit3D-Bench:

    renders/<dataset>/<object>/
      000.png ... 149.png
      transforms.json
      mesh.ply
      voxels.ply
      features.npz
      prompt_x/
        transforms.json -> ../transforms.json
        mesh.ply -> ../mesh.ply
        voxels.ply -> ../voxels.ply
        features.npz -> ../features.npz
        mesh_delete.ply
        voxels_delete.ply

The same shared cache is linked into both the single-view and multiview roots.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
VOXHAMMER_ROOT = WORKSPACE_ROOT / "VoxHammer"

os.environ.setdefault("ATTN_BACKEND", "flash-attn")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

if str(VOXHAMMER_ROOT) not in sys.path:
    sys.path.insert(0, str(VOXHAMMER_ROOT))


@dataclass(frozen=True)
class PromptCase:
    dataset: str
    object_name: str
    prompt_name: str

    @property
    def object_key(self) -> str:
        return f"{self.dataset}/{self.object_name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build shared VoxHammer encoder cache for pseudo-source datasets.")
    parser.add_argument(
        "--sv_root",
        type=Path,
        default=Path("/cache/wangxinxing/data/trellis_edit_benchmark/edit3d_sv_pseudosource_micro10"),
        help="Single-view dataset root.",
    )
    parser.add_argument(
        "--mv_root",
        type=Path,
        default=Path("/cache/wangxinxing/data/trellis_edit_benchmark/edit3d_mv_pseudosource_micro10"),
        help="Multiview dataset root. Used for linking the shared renders root.",
    )
    parser.add_argument(
        "--render_root",
        type=Path,
        default=Path("/cache/wangxinxing/data/trellis_edit_benchmark/edit3d_pseudosource_micro10_renders"),
        help="Output shared renders root.",
    )
    parser.add_argument(
        "--feature_model",
        type=str,
        default="dinov2_vitl14_reg",
        help="DINOv2 model name for feature extraction.",
    )
    parser.add_argument(
        "--feature_batch_size",
        type=int,
        default=10,
        help="Batch size for DINOv2 feature extraction.",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=150,
        help="Number of shared render views per source object.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Shared render resolution.",
    )
    parser.add_argument(
        "--shard_idx",
        type=int,
        default=0,
        help="Current shard index.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of shards.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip existing shared features and prompt delete voxels.",
    )
    parser.add_argument(
        "--no_skip_existing",
        dest="skip_existing",
        action="store_false",
        help="Rebuild even when outputs already exist.",
    )
    return parser.parse_args()


def load_prompt_cases(sv_root: Path) -> list[PromptCase]:
    metadata_path = sv_root / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as fp:
        metadata = json.load(fp)

    cases: list[PromptCase] = []
    for entry in metadata:
        dataset = entry["dataset"]
        object_name = entry["source_model"]
        for key in sorted(entry.keys()):
            if key.startswith("prompt_") and entry[key]:
                cases.append(PromptCase(dataset=dataset, object_name=object_name, prompt_name=key))
    return cases


def unique_object_cases(prompt_cases: Iterable[PromptCase]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    ordered: list[tuple[str, str]] = []
    for case in prompt_cases:
        key = (case.dataset, case.object_name)
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def shard_filter(items: list[Any], shard_idx: int, num_shards: int) -> list[Any]:
    return [item for idx, item in enumerate(items) if idx % num_shards == shard_idx]


def ensure_symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src)


def setup_prompt_render_dir(shared_render_dir: Path, prompt_render_dir: Path) -> None:
    prompt_render_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("transforms.json", "mesh.ply", "voxels.ply", "features.npz"):
        src = shared_render_dir / fname
        dst = prompt_render_dir / fname
        if src.exists():
            ensure_symlink(src, dst)


def write_summary(render_root: Path, built_objects: list[str], built_prompts: list[str], shard_idx: int, num_shards: int) -> None:
    summary = {
        "render_root": str(render_root),
        "built_object_count": len(built_objects),
        "built_prompt_count": len(built_prompts),
        "built_objects": built_objects,
        "built_prompts": built_prompts,
        "shard_idx": shard_idx,
        "num_shards": num_shards,
    }
    summary_path = render_root / f"summary.shard{shard_idx}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.render_root.mkdir(parents=True, exist_ok=True)

    os.chdir(VOXHAMMER_ROOT)

    from inference import run_3d_rendering, run_feature_extraction, run_voxel_masking

    prompt_cases = load_prompt_cases(args.sv_root)
    objects = unique_object_cases(prompt_cases)

    shard_objects = shard_filter(objects, args.shard_idx, args.num_shards)
    shard_object_keys = {f"{dataset}/{object_name}" for dataset, object_name in shard_objects}
    shard_prompts = [case for case in prompt_cases if case.object_key in shard_object_keys]

    built_objects: list[str] = []
    built_prompts: list[str] = []

    for dataset, object_name in shard_objects:
        object_dir = args.sv_root / dataset / object_name
        input_model = object_dir / "source_model" / "model.glb"
        shared_render_dir = args.render_root / dataset / object_name
        shared_render_dir.mkdir(parents=True, exist_ok=True)

        features_path = shared_render_dir / "features.npz"
        if not (args.skip_existing and features_path.exists()):
            run_3d_rendering(
                str(input_model),
                str(shared_render_dir),
                num_views=args.num_views,
                resolution=args.resolution,
            )
            run_feature_extraction(
                str(shared_render_dir),
                model=args.feature_model,
                batch_size=args.feature_batch_size,
            )
        built_objects.append(f"{dataset}/{object_name}")

    for case in shard_prompts:
        prompt_dir = args.sv_root / case.dataset / case.object_name / case.prompt_name
        mask_glb = prompt_dir / "3d_edit_region.glb"
        prompt_render_dir = args.render_root / case.dataset / case.object_name / case.prompt_name

        setup_prompt_render_dir(args.render_root / case.dataset / case.object_name, prompt_render_dir)

        voxels_delete = prompt_render_dir / "voxels_delete.ply"
        if not (args.skip_existing and voxels_delete.exists()):
            run_voxel_masking(str(mask_glb), str(prompt_render_dir))
        built_prompts.append(f"{case.dataset}/{case.object_name}/{case.prompt_name}")

    ensure_symlink(args.render_root, args.sv_root / "renders")
    ensure_symlink(args.render_root, args.mv_root / "renders")

    write_summary(args.render_root, built_objects, built_prompts, args.shard_idx, args.num_shards)


if __name__ == "__main__":
    main()
