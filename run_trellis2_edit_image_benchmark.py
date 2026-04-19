#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from PIL import Image

from run_batch_edit_and_eval import (
    DEFAULT_METRICS,
    render_all_results,
    run_evaluation,
    save_results,
)
from trellis_edit.common import ensure_dir, rewrite_glb_materials_to_matte_nonmetal, write_json


DEFAULT_GT_ROOT = Path("/cache/wangxinxing/data/trellis_edit_benchmark/edit3d_data/data")
DEFAULT_OUTPUT_ROOT = Path("/cache/wangxinxing/data/trellis_edit_benchmark/pred/trellis2_direct_edit_image")
DEFAULT_MODEL = "microsoft/TRELLIS.2-4B"
DEFAULT_QUALITY = "balanced"
TRELLIS2_REPO = Path("/home/wangxinxing/3dlocaledit/TRELLIS.2")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark TRELLIS.2 direct image-to-3D generation on edit images.",
    )
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--benchmark-root", type=Path, default=None)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--render-gpus", type=str, default="", help="Comma-separated GPU ids for benchmark rendering.")
    parser.add_argument("--eval-device", type=str, default="", help="Evaluation device, defaults to --device.")
    parser.add_argument("--limit", type=int, default=300, help="Number of prompt-level cases to include.")
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--config-name", type=str, default="baseline_trellis2_edit_image_direct")
    parser.add_argument("--run-group", type=str, default="baseline")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument(
        "--quality",
        type=str,
        default=DEFAULT_QUALITY,
        choices=(
            "high",
            "balanced",
            "balanced_remesh",
            "lowpoly_200k",
            "lowpoly_200k_remesh",
            "lowpoly_100k_remesh",
            "lowpoly_100k",
            "lowpoly_25k",
            "fast",
        ),
        help="Direct pipeline quality preset. balanced uses the non-cascade 1024 pipeline and lighter export.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from existing prompt outputs.")
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate prompt edit.glb files. Skip render/eval.",
    )
    parser.add_argument(
        "--max-new-cases",
        type=int,
        default=0,
        help="Stop after generating this many new prompt cases. 0 means no limit.",
    )
    parser.add_argument(
        "--case-shard-count",
        type=int,
        default=1,
        help="Split prompt-level generation into this many disjoint shards.",
    )
    parser.add_argument(
        "--case-shard-index",
        type=int,
        default=0,
        help="Zero-based case shard index to run.",
    )
    parser.add_argument(
        "--continue-on-case-error",
        action="store_true",
        help="Log prompt-level failures and continue instead of aborting the whole shard.",
    )
    parser.add_argument(
        "--drop-normal",
        action="store_true",
        help="Also remove normalTexture when rewriting final edit.glb materials.",
    )
    return parser


def _set_trellis2_env() -> None:
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _ensure_trellis2_import_path() -> None:
    trellis2_path = str(TRELLIS2_REPO.resolve())
    if trellis2_path not in sys.path:
        sys.path.insert(0, trellis2_path)


def _default_render_gpu_ids(device: str) -> list[str]:
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


def _load_benchmark_metadata(gt_root: Path) -> list[dict[str, Any]]:
    metadata_path = gt_root / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _select_cases(metadata: list[dict[str, Any]], limit: int, *, gt_root: Path) -> list[tuple[str, str, int]]:
    cases: list[tuple[str, str, int]] = []
    for row in metadata:
        dataset = str(row["dataset"])
        object_name = str(row["source_model"])
        for prompt_id in (1, 2, 3):
            if row.get(f"prompt_{prompt_id}") in (None, ""):
                continue
            prompt_dir = gt_root / dataset / object_name / f"prompt_{prompt_id}"
            if not prompt_dir.is_dir():
                continue
            cases.append((dataset, object_name, prompt_id))
            if len(cases) >= limit:
                return cases
    return cases


def _select_case_shard(
    cases: list[tuple[str, str, int]],
    *,
    shard_count: int,
    shard_index: int,
) -> list[tuple[str, str, int]]:
    if shard_count <= 1:
        return cases
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"Invalid shard index {shard_index} for shard count {shard_count}.")
    return [case for idx, case in enumerate(cases) if idx % shard_count == shard_index]


def _resolved_load_device(requested_device: str) -> str:
    if "CUDA_VISIBLE_DEVICES" in os.environ and requested_device.startswith("cuda:"):
        return "cuda:0"
    return requested_device


def _load_pipeline(*, model_name: str, device: str):
    import torch
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_name)
    pipeline.to(torch.device(_resolved_load_device(device)))
    return pipeline


def _pipeline_type_for_quality(quality: str) -> str:
    mapping = {
        "high": "1024_cascade",
        "balanced": "1024",
        "balanced_remesh": "1024",
        "lowpoly_200k": "1024",
        "lowpoly_200k_remesh": "1024",
        "lowpoly_100k_remesh": "1024",
        "lowpoly_100k": "1024",
        "lowpoly_25k": "1024",
        "fast": "512",
    }
    try:
        return mapping[quality]
    except KeyError as exc:
        raise ValueError(f"Unknown quality preset: {quality}") from exc


def _export_attempts_for_quality(quality: str) -> list[dict[str, Any]]:
    if quality == "high":
        return [
            {
                "name": "default",
                "decimation_target": 1000000,
                "texture_size": 4096,
                "remesh": True,
                "remesh_band": 1,
                "remesh_project": 0,
            },
            {
                "name": "fallback_no_remesh_500k",
                "decimation_target": 500000,
                "texture_size": 2048,
                "remesh": False,
                "remesh_band": 1,
                "remesh_project": 0,
            },
            {
                "name": "fallback_no_remesh_300k",
                "decimation_target": 300000,
                "texture_size": 1024,
                "remesh": False,
                "remesh_band": 1,
                "remesh_project": 0,
            },
            {
                "name": "fallback_no_remesh_100k",
                "decimation_target": 100000,
                "texture_size": 512,
                "remesh": False,
                "remesh_band": 1,
                "remesh_project": 0,
            },
            {
                "name": "fallback_no_remesh_50k",
                "decimation_target": 50000,
                "texture_size": 512,
                "remesh": False,
                "remesh_band": 1,
                "remesh_project": 0,
            },
            {
                "name": "fallback_no_remesh_25k",
                "decimation_target": 25000,
                "texture_size": 256,
                "remesh": False,
                "remesh_band": 1,
                "remesh_project": 0,
            },
        ]
    if quality == "balanced":
        return [
            {
                "name": "balanced_remesh_500k",
                "decimation_target": 500000,
                "texture_size": 2048,
                "remesh": True,
                "remesh_band": 1,
                "remesh_project": 0.9,
            },
            {
                "name": "balanced_remesh_300k",
                "decimation_target": 300000,
                "texture_size": 1024,
                "remesh": True,
                "remesh_band": 1,
                "remesh_project": 0.9,
            },
            {
                "name": "balanced_remesh_100k",
                "decimation_target": 100000,
                "texture_size": 512,
                "remesh": True,
                "remesh_band": 1,
                "remesh_project": 0.9,
            },
            {
                "name": "balanced_remesh_50k",
                "decimation_target": 50000,
                "texture_size": 512,
                "remesh": True,
                "remesh_band": 1,
                "remesh_project": 0.9,
            },
            {
                "name": "balanced_remesh_25k",
                "decimation_target": 25000,
                "texture_size": 256,
                "remesh": True,
                "remesh_band": 1,
                "remesh_project": 0.9,
            },
        ]
    if quality == "balanced_remesh":
        return [
            {
                "name": "balanced_remesh_500k",
                "decimation_target": 500000,
                "texture_size": 2048,
                "remesh": True,
                "remesh_band": 1,
                "remesh_project": 0.9,
            },
        ]
    if quality == "lowpoly_100k":
        return [
            {
                "name": "lowpoly_100k",
                "decimation_target": 100000,
                "texture_size": 1024,
                "remesh": False,
                "remesh_band": 1,
                "remesh_project": 0,
            },
            {
                "name": "fallback_no_remesh_50k",
                "decimation_target": 50000,
                "texture_size": 512,
                "remesh": False,
                "remesh_band": 1,
                "remesh_project": 0,
            },
            {
                "name": "fallback_no_remesh_25k",
                "decimation_target": 25000,
                "texture_size": 256,
                "remesh": False,
                "remesh_band": 1,
                "remesh_project": 0,
            },
        ]
    if quality == "lowpoly_100k_remesh":
        return [
            {
                "name": "lowpoly_100k_remesh",
                "decimation_target": 100000,
                "texture_size": 1024,
                "remesh": True,
                "remesh_band": 1,
                "remesh_project": 0.9,
            },
        ]
    if quality == "lowpoly_200k":
        return [
            {
                "name": "lowpoly_200k",
                "decimation_target": 200000,
                "texture_size": 1536,
                "remesh": False,
                "remesh_band": 1,
                "remesh_project": 0,
            },
            {
                "name": "fallback_no_remesh_100k",
                "decimation_target": 100000,
                "texture_size": 1024,
                "remesh": False,
                "remesh_band": 1,
                "remesh_project": 0,
            },
            {
                "name": "fallback_no_remesh_50k",
                "decimation_target": 50000,
                "texture_size": 512,
                "remesh": False,
                "remesh_band": 1,
                "remesh_project": 0,
            },
        ]
    if quality == "lowpoly_200k_remesh":
        return [
            {
                "name": "lowpoly_200k_remesh",
                "decimation_target": 200000,
                "texture_size": 1536,
                "remesh": True,
                "remesh_band": 1,
                "remesh_project": 0.9,
            },
        ]
    if quality == "lowpoly_25k":
        return [
            {
                "name": "lowpoly_25k",
                "decimation_target": 25000,
                "texture_size": 256,
                "remesh": False,
                "remesh_band": 1,
                "remesh_project": 0,
            },
            {
                "name": "fallback_no_remesh_10k",
                "decimation_target": 10000,
                "texture_size": 256,
                "remesh": False,
                "remesh_band": 1,
                "remesh_project": 0,
            },
        ]
    if quality == "fast":
        return [
            {
                "name": "fast_no_remesh_300k",
                "decimation_target": 300000,
                "texture_size": 1024,
                "remesh": False,
                "remesh_band": 1,
                "remesh_project": 0,
            },
            {
                "name": "fallback_no_remesh_100k",
                "decimation_target": 100000,
                "texture_size": 512,
                "remesh": False,
                "remesh_band": 1,
                "remesh_project": 0,
            },
            {
                "name": "fallback_no_remesh_50k",
                "decimation_target": 50000,
                "texture_size": 256,
                "remesh": False,
                "remesh_band": 1,
                "remesh_project": 0,
            },
        ]
    raise ValueError(f"Unknown quality preset: {quality}")


def _export_glb(mesh_with_voxel, *, glb_path: Path, quality: str) -> None:
    import o_voxel
    import torch

    ensure_dir(glb_path.parent)
    export_attempts = _export_attempts_for_quality(quality)

    last_error: Exception | None = None
    for attempt in export_attempts:
        try:
            print(
                f"[Export] {glb_path.name} using {attempt['name']} "
                f"(remesh={attempt['remesh']}, decimation={attempt['decimation_target']}, texture={attempt['texture_size']})"
            )
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh_with_voxel.vertices.detach(),
                faces=mesh_with_voxel.faces.detach(),
                attr_volume=mesh_with_voxel.attrs.detach(),
                coords=mesh_with_voxel.coords.detach(),
                attr_layout=mesh_with_voxel.layout,
                voxel_size=mesh_with_voxel.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=attempt['decimation_target'],
                texture_size=attempt['texture_size'],
                remesh=attempt['remesh'],
                remesh_band=attempt['remesh_band'],
                remesh_project=attempt['remesh_project'],
                verbose=True,
            )
            glb.export(str(glb_path), extension_webp=True)
            return
        except RuntimeError as exc:
            last_error = exc
            message = str(exc).lower()
            if 'out of memory' not in message and 'cuda error' not in message:
                raise
            print(f"[Export] {attempt['name']} failed with OOM, retrying with safer settings")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            last_error = exc
            raise

    assert last_error is not None
    raise last_error


def _generate_case(
    *,
    pipeline,
    gt_root: Path,
    output_root: Path,
    dataset: str,
    object_name: str,
    prompt_id: int,
    seed: int,
    num_samples: int,
    quality: str,
    drop_normal: bool,
) -> dict[str, Any]:
    import torch

    prompt_dir = gt_root / dataset / object_name / f"prompt_{prompt_id}"
    edit_image_path = prompt_dir / "2d_edit.png"
    if not edit_image_path.is_file():
        raise FileNotFoundError(f"Missing edit image: {edit_image_path}")

    output_dir = output_root / dataset / object_name / f"prompt_{prompt_id}"
    output_glb = output_dir / "edit.glb"
    if output_glb.is_file():
        return {
            "edit_image": str(edit_image_path),
            "glb_path": str(output_glb),
            "resumed": True,
        }

    edit_image = None
    mesh = None
    material_postprocess = None
    try:
        with Image.open(edit_image_path) as handle:
            edit_image = handle.copy()

        with torch.no_grad():
            mesh = pipeline.run(
                edit_image,
                num_samples=num_samples,
                seed=seed,
                pipeline_type=_pipeline_type_for_quality(quality),
            )[0]
        _export_glb(mesh, glb_path=output_glb, quality=quality)
        material_postprocess = rewrite_glb_materials_to_matte_nonmetal(
            output_glb,
            drop_normal=drop_normal,
        )
    finally:
        del mesh
        del edit_image
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "edit_image": str(edit_image_path),
        "glb_path": str(output_glb),
        "material_postprocess": material_postprocess,
        "resumed": False,
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.case_shard_count < 1:
        raise ValueError("--case-shard-count must be >= 1.")
    if args.case_shard_count > 1 and not args.generate_only:
        raise RuntimeError(
            "Sharded generation must use --generate-only. "
            "Run a final non-sharded --resume pass for render/eval."
        )

    gt_root = args.gt_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    benchmark_root = (
        args.benchmark_root.expanduser().resolve()
        if args.benchmark_root is not None
        else (output_root.parent.parent if output_root.parent.name == "pred" else output_root.parent)
    )
    eval_device = args.eval_device or args.device
    render_gpu_ids = _parse_render_gpu_ids(args.render_gpus, device=args.device) if not args.generate_only else []
    if not args.generate_only and not render_gpu_ids:
        raise RuntimeError("No render GPUs resolved. Pass --render-gpus or use a cuda device.")

    metadata = _load_benchmark_metadata(gt_root)
    cases = _select_cases(metadata, args.limit, gt_root=gt_root)
    if not cases:
        raise RuntimeError("No benchmark cases selected.")
    shard_cases = _select_case_shard(
        cases,
        shard_count=args.case_shard_count,
        shard_index=args.case_shard_index,
    )
    if not shard_cases:
        raise RuntimeError(
            f"No cases selected for shard {args.case_shard_index}/{args.case_shard_count}."
        )

    if output_root.exists():
        if args.case_shard_count > 1 and not args.resume:
            raise RuntimeError(
                "Sharded generation should start from a prepared output dir. "
                "Clean it once, then launch all shards with --resume."
            )
        if not args.resume:
            shutil.rmtree(output_root)
    ensure_dir(output_root)

    start_time = time.time()
    _set_trellis2_env()
    _ensure_trellis2_import_path()
    pipeline = _load_pipeline(model_name=args.model, device=args.device)

    manifest_path = output_root / "direct_manifest.json"
    failed_cases_path = output_root / "direct_failures.json"
    manifest_cases: dict[str, Any] = {}
    failed_cases: dict[str, Any] = {}
    if args.resume and manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_cases.update(existing_manifest.get("generated_cases") or {})
        failed_cases.update(existing_manifest.get("failed_cases") or {})
    if args.resume and failed_cases_path.is_file():
        existing_failed = json.loads(failed_cases_path.read_text(encoding="utf-8"))
        failed_cases.update(existing_failed.get("cases") or {})
    total_cases = len(shard_cases)
    print(
        f"[Shard] case shard {args.case_shard_index}/{args.case_shard_count} "
        f"selected {total_cases}/{len(cases)} cases"
    )
    new_case_count = 0
    for case_index, (dataset, object_name, prompt_id) in enumerate(shard_cases, start=1):
        case_id = f"{dataset}/{object_name}/prompt_{prompt_id}"
        print(f"[Direct] {case_index}/{total_cases} {case_id}")
        try:
            case_result = _generate_case(
                pipeline=pipeline,
                gt_root=gt_root,
                output_root=output_root,
                dataset=dataset,
                object_name=object_name,
                prompt_id=prompt_id,
                seed=args.seed,
                num_samples=args.num_samples,
                quality=args.quality,
                drop_normal=args.drop_normal,
            )
        except Exception as exc:
            failed_cases[case_id] = {
                "dataset": dataset,
                "object_name": object_name,
                "prompt_id": prompt_id,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            write_json(failed_cases_path, {"cases": failed_cases})
            if not args.continue_on_case_error:
                raise
            print(f"[Skip] {case_id} failed: {exc}")
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            continue
        manifest_cases[case_id] = case_result
        failed_cases.pop(case_id, None)
        if not case_result.get("resumed", False):
            new_case_count += 1
        if failed_cases:
            write_json(failed_cases_path, {"cases": failed_cases})
        elif failed_cases_path.exists():
            failed_cases_path.unlink()
        write_json(
            manifest_path,
            {
                "config_name": args.config_name,
                "run_group": args.run_group,
                "model": args.model,
                "device": args.device,
                "eval_device": eval_device,
                "render_gpu_ids": render_gpu_ids,
                "seed": args.seed,
                "num_samples": args.num_samples,
                "quality": args.quality,
                "drop_normal": bool(args.drop_normal),
                "case_shard_count": args.case_shard_count,
                "case_shard_index": args.case_shard_index,
                "metrics": list(args.metrics),
                "failed_cases": failed_cases,
                "cases": [
                    {
                        "dataset": case_dataset,
                        "object_name": case_object_name,
                        "prompt_id": case_prompt_id,
                    }
                    for case_dataset, case_object_name, case_prompt_id in cases
                ],
                "generated_cases": manifest_cases,
            },
        )
        if args.max_new_cases > 0 and new_case_count >= args.max_new_cases:
            print(
                f"[Stop] Reached max new cases for this run: "
                f"{new_case_count}/{args.max_new_cases}"
            )
            break

    if args.generate_only or len(manifest_cases) < total_cases:
        print(
            f"[Done] Direct generation saved {len(manifest_cases)}/{total_cases} cases "
            f"(new this run: {new_case_count})"
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
    print(f"[Done] TRELLIS.2 direct-edit benchmark finished in {total_time:.1f}s")
    print(f"[Done] Pred root: {output_root}")


if __name__ == "__main__":
    main()
