#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from trellis_edit.common import (
    build_backend_config,
    ensure_dir,
    save_outputs,
    write_json,
)
from trellis_edit.common.save_utils import release_cuda_memory
from run_batch_edit_and_eval import (
    DEFAULT_METRICS,
    load_edit3d_metadata,
    render_all_results,
    run_evaluation,
    save_results,
)


DEFAULT_GT_ROOT = Path("/cache/wangxinxing/data/trellis_edit_benchmark/edit3d_data/data")
DEFAULT_ASSETS_ROOT = Path("/cache/wangxinxing/data/trellis_edit_benchmark/edit3d_data/renders")
DEFAULT_OUTPUT_ROOT = Path("/cache/wangxinxing/data/trellis_edit_benchmark/pred/baseline00_reconstruct")
DEFAULT_MODEL = "microsoft/TRELLIS-image-large"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct baseline using source features -> slat_encoder -> decode_slat.",
    )
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--benchmark-root", type=Path, default=None)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--render-gpus", type=str, default="", help="Comma-separated GPU ids for benchmark rendering.")
    parser.add_argument("--eval-device", type=str, default="", help="Evaluation device, defaults to --device.")
    parser.add_argument("--attn-backend", type=str, default="")
    parser.add_argument("--sparse-attn-backend", type=str, default="")
    parser.add_argument("--spconv-algo", type=str, default="native")
    parser.add_argument("--limit", type=int, default=12, help="Number of prompt-level cases to run.")
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--config-name", type=str, default="baseline00_reconstruct")
    parser.add_argument("--run-group", type=str, default="baseline")
    parser.add_argument("--skip-ply", action="store_true", help="Skip writing gaussian PLY outputs.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing object/prompt outputs.")
    parser.add_argument(
        "--max-new-objects",
        type=int,
        default=0,
        help="Stop after generating this many new objects. 0 means no limit.",
    )
    parser.add_argument(
        "--reconstruct-only",
        action="store_true",
        help="Only reconstruct object GLBs and prompt edit.glb files. Skip render/eval.",
    )
    parser.add_argument(
        "--object-shard-count",
        type=int,
        default=1,
        help="Split object reconstruction into this many disjoint shards.",
    )
    parser.add_argument(
        "--object-shard-index",
        type=int,
        default=0,
        help="Zero-based object shard index to run.",
    )
    return parser


def _resolved_load_device(requested_device: str) -> str:
    import os

    if "CUDA_VISIBLE_DEVICES" in os.environ and requested_device.startswith("cuda:"):
        return "cuda:0"
    return requested_device


def _default_render_gpu_ids(device: str) -> list[str]:
    import os

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_devices:
        return [item.strip() for item in visible_devices.split(",") if item.strip()]
    if device.startswith("cuda:"):
        return [device.split(":", 1)[1]]
    return []


def _parse_render_gpu_ids(raw_value: str, *, device: str) -> list[str]:
    if not raw_value.strip():
        return _default_render_gpu_ids(device)
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _select_cases(metadata: list[dict[str, Any]], limit: int) -> list[tuple[str, str, int]]:
    cases: list[tuple[str, str, int]] = []
    for row in metadata:
        dataset = str(row["dataset"])
        object_name = str(row["source_model"])
        for prompt_id in (1, 2, 3):
            cases.append((dataset, object_name, prompt_id))
            if len(cases) >= limit:
                return cases
    return cases


def _group_cases_by_object(cases: list[tuple[str, str, int]]) -> dict[tuple[str, str], list[int]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for dataset, object_name, prompt_id in cases:
        grouped[(dataset, object_name)].append(prompt_id)
    return dict(grouped)


def _select_object_shard(
    grouped_cases: dict[tuple[str, str], list[int]],
    *,
    shard_count: int,
    shard_index: int,
) -> dict[tuple[str, str], list[int]]:
    if shard_count <= 1:
        return grouped_cases
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            f"Invalid shard index {shard_index} for shard count {shard_count}."
        )
    shard_items = [
        item
        for object_idx, item in enumerate(grouped_cases.items())
        if object_idx % shard_count == shard_index
    ]
    return dict(shard_items)


def _load_pipeline(*, model: str, device: str, attn_backend: str, sparse_attn_backend: str, spconv_algo: str):
    import os
    from trellis.pipelines import TrellisImageTo3DPipeline

    backend = build_backend_config(
        attn_backend=attn_backend,
        sparse_attn_backend=sparse_attn_backend,
        spconv_algo=spconv_algo,
    )
    for key, value in backend.env.items():
        os.environ[key] = value

    pipeline = TrellisImageTo3DPipeline.from_pretrained(model)
    return pipeline


def _resolved_torch_device(device: str):
    import torch

    return torch.device(_resolved_load_device(device))


def _set_active_models(pipeline, *, active_names: set[str], device: str) -> None:
    target_device = _resolved_torch_device(device)
    cpu_device = _resolved_torch_device("cpu")
    for name, model in pipeline.models.items():
        if hasattr(model, "to"):
            model.to(target_device if name in active_names else cpu_device)
    release_cuda_memory()


def _load_features_as_slat(pipeline, *, features_path: Path, device: str, SparseTensor):
    import torch

    payload = np.load(features_path)
    target_device = _resolved_torch_device(device)

    if "patchtokens" in payload and "indices" in payload:
        sparse_tensor = SparseTensor(
            feats=torch.from_numpy(payload["patchtokens"]).float().to(target_device),
            coords=torch.cat(
                [
                    torch.zeros(payload["patchtokens"].shape[0], 1, dtype=torch.int32),
                    torch.from_numpy(payload["indices"]).int(),
                ],
                dim=1,
            ).to(target_device),
        )
        return pipeline.models["slat_encoder"](sparse_tensor, sample_posterior=False)

    if "feats" in payload and "coords" in payload:
        return SparseTensor(
            feats=torch.from_numpy(payload["feats"]).float().to(target_device),
            coords=torch.from_numpy(payload["coords"]).int().to(target_device),
        )

    raise RuntimeError(f"Unsupported features format: {features_path}")


def _move_mesh_outputs_to_cpu(mesh_outputs) -> None:
    for mesh in mesh_outputs:
        if hasattr(mesh, "vertices"):
            mesh.vertices = mesh.vertices.cpu()
        if hasattr(mesh, "faces"):
            mesh.faces = mesh.faces.cpu()
    release_cuda_memory()


def _reconstruct_object(
    *,
    pipeline,
    pipeline_device: str,
    features_path: Path,
    object_output_dir: Path,
    prompt_output_dirs: list[Path],
    skip_ply: bool,
) -> dict[str, Any]:
    import torch
    from trellis.modules.sparse.basic import SparseTensor

    glb_path = object_output_dir / "sample_00.glb"
    prompt_glb_paths: list[str] = []
    all_prompt_glbs_exist = all((prompt_output_dir / "edit.glb").is_file() for prompt_output_dir in prompt_output_dirs)

    if glb_path.is_file() and all_prompt_glbs_exist:
        for prompt_output_dir in prompt_output_dirs:
            prompt_glb_paths.append(str(prompt_output_dir / "edit.glb"))
        return {
            "features_path": str(features_path),
            "object_output_dir": str(object_output_dir),
            "glb_path": str(glb_path),
            "prompt_glb_paths": prompt_glb_paths,
            "resumed": True,
        }

    ensure_dir(object_output_dir)
    with torch.no_grad():
        _set_active_models(pipeline, active_names={"slat_encoder"}, device=pipeline_device)
        slat = _load_features_as_slat(
            pipeline,
            features_path=features_path,
            device=pipeline_device,
            SparseTensor=SparseTensor,
        )

        _set_active_models(pipeline, active_names={"slat_decoder_mesh"}, device=pipeline_device)
        mesh_outputs = pipeline.models["slat_decoder_mesh"](slat)
        _move_mesh_outputs_to_cpu(mesh_outputs)
        _set_active_models(pipeline, active_names={"slat_decoder_gs"}, device=pipeline_device)
        gaussian_outputs = pipeline.models["slat_decoder_gs"](slat)
    outputs = {"mesh": mesh_outputs, "gaussian": gaussian_outputs}
    _set_active_models(pipeline, active_names=set(), device=pipeline_device)
    try:
        save_outputs(
            outputs=outputs,
            out_dir=object_output_dir,
            skip_render=True,
            skip_glb=False,
            skip_ply=skip_ply,
        )
    finally:
        del outputs
        del mesh_outputs
        del gaussian_outputs
        del slat
        release_cuda_memory()
    if not glb_path.is_file():
        raise RuntimeError(f"Missing reconstructed GLB: {glb_path}")

    for prompt_output_dir in prompt_output_dirs:
        ensure_dir(prompt_output_dir)
        prompt_glb = prompt_output_dir / "edit.glb"
        shutil.copy2(glb_path, prompt_glb)
        prompt_glb_paths.append(str(prompt_glb))

    return {
        "features_path": str(features_path),
        "object_output_dir": str(object_output_dir),
        "glb_path": str(glb_path),
        "prompt_glb_paths": prompt_glb_paths,
        "resumed": False,
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.object_shard_count < 1:
        raise ValueError("--object-shard-count must be >= 1.")
    if args.object_shard_count > 1 and not args.reconstruct_only:
        raise RuntimeError(
            "Sharded reconstruction must use --reconstruct-only. "
            "Run a final non-sharded --resume pass for render/eval."
        )
    gt_root = args.gt_root.expanduser().resolve()
    assets_root = args.assets_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    benchmark_root = (
        args.benchmark_root.expanduser().resolve()
        if args.benchmark_root is not None
        else (output_root.parent.parent if output_root.parent.name == "pred" else output_root.parent)
    )
    eval_device = args.eval_device or args.device
    render_gpu_ids = _parse_render_gpu_ids(args.render_gpus, device=args.device)
    if not render_gpu_ids:
        raise RuntimeError("No render GPUs resolved. Pass --render-gpus or use a cuda device.")

    metadata = load_edit3d_metadata(gt_root)
    cases = _select_cases(metadata, args.limit)
    if not cases:
        raise RuntimeError("No cases selected from metadata.")
    grouped_cases = _group_cases_by_object(cases)
    total_objects_all = len(grouped_cases)
    grouped_cases = _select_object_shard(
        grouped_cases,
        shard_count=args.object_shard_count,
        shard_index=args.object_shard_index,
    )
    if not grouped_cases:
        raise RuntimeError(
            f"No objects selected for shard {args.object_shard_index}/{args.object_shard_count}."
        )

    if output_root.exists() and not args.resume:
        shutil.rmtree(output_root)
    ensure_dir(output_root)

    start_time = time.time()
    pipeline = _load_pipeline(
        model=args.model,
        device=args.device,
        attn_backend=args.attn_backend,
        sparse_attn_backend=args.sparse_attn_backend,
        spconv_algo=args.spconv_algo,
    )

    manifest_objects: dict[str, Any] = {}
    total_objects = len(grouped_cases)
    print(
        f"[Shard] object shard {args.object_shard_index}/{args.object_shard_count} "
        f"selected {total_objects}/{total_objects_all} objects"
    )
    new_object_count = 0
    for object_index, ((dataset, object_name), prompt_ids) in enumerate(grouped_cases.items(), start=1):
        features_path = assets_root / dataset / object_name / "features.npz"
        if not features_path.is_file():
            raise FileNotFoundError(f"Missing source features: {features_path}")

        print(
            f"[Reconstruct] {object_index}/{total_objects} {dataset}/{object_name} "
            f"-> prompts {','.join(str(item) for item in prompt_ids)}"
        )
        object_output_dir = output_root / "_object_recon" / dataset / object_name
        prompt_output_dirs = [
            output_root / dataset / object_name / f"prompt_{prompt_id}"
            for prompt_id in prompt_ids
        ]
        object_result = _reconstruct_object(
            pipeline=pipeline,
            pipeline_device=args.device,
            features_path=features_path,
            object_output_dir=object_output_dir,
            prompt_output_dirs=prompt_output_dirs,
            skip_ply=args.skip_ply,
        )
        manifest_objects[f"{dataset}/{object_name}"] = object_result
        if not object_result.get("resumed", False):
            new_object_count += 1
        write_json(
            output_root / "reconstruct_manifest.json",
            {
                "config_name": args.config_name,
                "run_group": args.run_group,
                "model": args.model,
                "device": args.device,
                "render_gpu_ids": render_gpu_ids,
                "eval_device": eval_device,
                "metrics": list(args.metrics),
                "object_shard_count": args.object_shard_count,
                "object_shard_index": args.object_shard_index,
                "cases": [
                    {
                        "dataset": case_dataset,
                        "object_name": case_object_name,
                        "prompt_id": case_prompt_id,
                    }
                    for case_dataset, case_object_name, case_prompt_id in cases
                ],
                "objects": manifest_objects,
            },
        )
        if args.max_new_objects > 0 and new_object_count >= args.max_new_objects:
            print(
                f"[Stop] Reached max new objects for this run: "
                f"{new_object_count}/{args.max_new_objects}"
            )
            break

    if args.reconstruct_only or len(manifest_objects) < total_objects:
        print(
            f"[Done] Reconstruct phase saved {len(manifest_objects)}/{total_objects} objects "
            f"(new this run: {new_object_count})"
        )
        print(f"[Done] Pred root: {output_root}")
        return

    print("[Render] Rendering benchmark views...")
    if not render_all_results(output_root, gpu_ids=render_gpu_ids, metrics=list(args.metrics)):
        raise RuntimeError("Benchmark rendering failed.")

    print("[Eval] Running benchmark evaluation...")
    eval_output_dir = ensure_dir(output_root / "evaluation_output")
    ok, summary = run_evaluation(
        gt_root=gt_root,
        pred_root=output_root,
        metrics=list(args.metrics),
        output_dir=eval_output_dir,
        device=eval_device,
    )
    if not ok or summary is None:
        raise RuntimeError("Benchmark evaluation failed.")

    total_time = time.time() - start_time
    save_results(
        output_root=output_root,
        entrypoint_name="baseline",
        config_name=args.config_name,
        run_group=args.run_group,
        gt_root=gt_root,
        cases=cases,
        requested_metrics=list(args.metrics),
        benchmark_root=benchmark_root,
        skip_benchmark_render=False,
        results=summary,
        total_time=total_time,
    )
    print(f"[Done] baseline00 reconstruct finished in {total_time:.1f}s")
    print(f"[Done] Pred root: {output_root}")


if __name__ == "__main__":
    main()
