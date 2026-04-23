#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import sys
import tempfile
import time
import traceback
import types
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from trellis_edit.alignment import export_hunyuan21_glb_to_canonical_space
from trellis_edit.common import ensure_dir, release_cuda_memory, utc_now_iso, write_json


DEFAULT_GT_ROOT = Path("/cache/wangxinxing/data/trellis_edit_benchmark/edit3d_data/data")
DEFAULT_OUTPUT_ROOT = Path("/cache/wangxinxing/data/trellis_edit_benchmark/pred/baseline_hunyuan21_direct_edit")
HUNYUAN_ROOT = Path("/home/wangxinxing/3dlocaledit/Hunyuan3D-2.1")
DEFAULT_MODEL = "tencent/Hunyuan3D-2.1"
REALESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
_DEVICE_REEXEC_FLAG = "TRELLIS_HUNYUAN21_DEVICE_REEXEC"
_ORIGINAL_DEVICE_ENV = "TRELLIS_HUNYUAN21_ORIGINAL_DEVICE"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Hunyuan3D-2.1 direct edit-image generation on Edit3D-Bench and save "
            "pred-ready textured edit.glb files after native-space canonical alignment and optional "
            "matte non-metal material rewrite."
        ),
    )
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--limit", type=int, default=300, help="Number of prompt-level cases to include.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Resume from existing prompt outputs.")
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
    parser.add_argument(
        "--keep-materials",
        action="store_true",
        help="Preserve exported GLB materials instead of rewriting them to matte non-metal.",
    )
    parser.add_argument("--config-name", type=str, default="baseline_hunyuan21_direct_edit")
    parser.add_argument("--run-group", type=str, default="basic_baselines")
    return parser


def _ensure_hunyuan_import_path() -> None:
    for path in (HUNYUAN_ROOT, HUNYUAN_ROOT / "hy3dshape", HUNYUAN_ROOT / "hy3dpaint"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _apply_torchvision_fix() -> None:
    try:
        from torchvision_fix import apply_fix

        apply_fix()
    except Exception:
        return


def _install_dummy_bpy() -> None:
    if "bpy" not in sys.modules:
        sys.modules["bpy"] = types.ModuleType("bpy")


def _ensure_realesrgan_weight() -> Path:
    ckpt_dir = ensure_dir(HUNYUAN_ROOT / "hy3dpaint" / "ckpt")
    ckpt_path = ckpt_dir / "RealESRGAN_x4plus.pth"
    if not ckpt_path.is_file():
        urllib.request.urlretrieve(REALESRGAN_URL, ckpt_path)
    return ckpt_path


def _argv_with_device(argv: list[str], replacement_device: str) -> list[str]:
    rewritten = list(argv)
    if "--device" in rewritten:
        index = rewritten.index("--device")
        if index + 1 >= len(rewritten):
            raise ValueError("--device flag is missing its value.")
        rewritten[index + 1] = replacement_device
        return rewritten
    rewritten.extend(["--device", replacement_device])
    return rewritten


def _maybe_reexec_with_visible_device(requested_device: str) -> str:
    if not requested_device.startswith("cuda:"):
        return requested_device
    if requested_device == "cuda:0":
        return requested_device
    if os.environ.get(_DEVICE_REEXEC_FLAG) == "1":
        return "cuda:0"

    physical_index = requested_device.split(":", 1)[1]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = physical_index
    env[_DEVICE_REEXEC_FLAG] = "1"
    env[_ORIGINAL_DEVICE_ENV] = requested_device
    rewritten_argv = _argv_with_device(sys.argv, "cuda:0")
    os.execvpe(sys.executable, [sys.executable, *rewritten_argv], env)
    raise AssertionError("os.execvpe unexpectedly returned")


def _load_benchmark_metadata(gt_root: Path) -> list[dict[str, Any]]:
    metadata_path = gt_root / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _select_cases(metadata: list[dict[str, Any]], limit: int, *, gt_root: Path) -> list[tuple[str, str, int]]:
    cases: list[tuple[str, str, int]] = []
    for row in metadata:
        dataset = str(row["dataset"])
        object_name = str(row["source_model"])
        reference_glb = gt_root / dataset / object_name / "source_model" / "model.glb"
        if not reference_glb.is_file():
            continue
        for prompt_id in (1, 2, 3):
            prompt_dir = gt_root / dataset / object_name / f"prompt_{prompt_id}"
            edit_image = prompt_dir / "2d_edit.png"
            if not prompt_dir.is_dir() or not edit_image.is_file():
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


def _set_case_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _prepare_edit_image(edit_image_path: Path, background_remover) -> Image.Image:
    with Image.open(edit_image_path) as handle:
        image = handle.copy()
    if image.mode == "RGB":
        image = background_remover(image)
    else:
        image = image.convert("RGBA")
    return image


def _quick_convert_with_obj2gltf(obj_path: Path, glb_path: Path, *, workdir: Path) -> None:
    from hy3dpaint.convert_utils import create_glb_with_pbr_materials

    textures = {
        "albedo": str(obj_path).replace(".obj", ".jpg"),
        "metallic": str(obj_path).replace(".obj", "_metallic.jpg"),
        "roughness": str(obj_path).replace(".obj", "_roughness.jpg"),
    }

    previous_cwd = Path.cwd()
    os.chdir(workdir)
    try:
        create_glb_with_pbr_materials(str(obj_path), textures, str(glb_path))
    finally:
        os.chdir(previous_cwd)


def _load_hunyuan_pipelines(*, model: str, device: str):
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dshape.rembg import BackgroundRemover
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

    shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model, device=device)
    realesrgan_ckpt = _ensure_realesrgan_weight()
    paint_conf = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
    paint_conf.device = device
    paint_conf.realesrgan_ckpt_path = str(realesrgan_ckpt)
    paint_conf.multiview_cfg_path = str(HUNYUAN_ROOT / "hy3dpaint" / "cfgs" / "hunyuan-paint-pbr.yaml")
    paint_pipeline = Hunyuan3DPaintPipeline(paint_conf)
    background_remover = BackgroundRemover()
    return background_remover, shape_pipeline, paint_pipeline


def _generate_case(
    *,
    gt_root: Path,
    output_root: Path,
    dataset: str,
    object_name: str,
    prompt_id: int,
    seed: int,
    background_remover,
    shape_pipeline,
    paint_pipeline,
    drop_normal: bool,
    keep_materials: bool,
) -> dict[str, Any]:
    prompt_dir = gt_root / dataset / object_name / f"prompt_{prompt_id}"
    edit_image_path = prompt_dir / "2d_edit.png"
    reference_glb = gt_root / dataset / object_name / "source_model" / "model.glb"
    if not edit_image_path.is_file():
        raise FileNotFoundError(f"Missing edit image: {edit_image_path}")
    if not reference_glb.is_file():
        raise FileNotFoundError(f"Missing reference source model: {reference_glb}")

    output_dir = output_root / dataset / object_name / f"prompt_{prompt_id}"
    output_glb = output_dir / "edit.glb"
    if output_glb.is_file():
        return {
            "edit_image": str(edit_image_path),
            "reference_glb": str(reference_glb),
            "glb_path": str(output_glb),
            "resumed": True,
        }

    ensure_dir(output_dir)
    temp_root = ensure_dir(output_root / "_tmp")
    image = None
    mesh = None
    started_at = time.time()
    with tempfile.TemporaryDirectory(prefix=f"{dataset}_{object_name}_prompt_{prompt_id}_", dir=temp_root) as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        white_mesh_path = temp_dir / "white_mesh.glb"
        textured_obj_path = temp_dir / "textured_mesh.obj"
        textured_glb_path = temp_dir / "textured_mesh.glb"

        _set_case_seed(seed)
        try:
            image = _prepare_edit_image(edit_image_path, background_remover)

            shape_started = time.time()
            mesh = shape_pipeline(image=image)[0]
            white_mesh_path.parent.mkdir(parents=True, exist_ok=True)
            mesh.export(white_mesh_path)
            shape_seconds = round(time.time() - shape_started, 3)

            white_mesh_faces = int(mesh.faces.shape[0])
            white_mesh_vertices = int(mesh.vertices.shape[0])

            paint_started = time.time()
            paint_pipeline(
                mesh_path=str(white_mesh_path),
                image_path=image,
                output_mesh_path=str(textured_obj_path),
                save_glb=False,
            )
            paint_seconds = round(time.time() - paint_started, 3)

            convert_started = time.time()
            _quick_convert_with_obj2gltf(textured_obj_path, textured_glb_path, workdir=temp_dir)
            convert_seconds = round(time.time() - convert_started, 3)

            postprocess_started = time.time()
            postprocess_stats = export_hunyuan21_glb_to_canonical_space(
                input_glb=textured_glb_path,
                output_glb=output_glb,
                apply_matte_nonmetal=not keep_materials,
                drop_normal=drop_normal,
            )
            postprocess_seconds = round(time.time() - postprocess_started, 3)
        finally:
            del mesh
            del image
            gc.collect()
            release_cuda_memory()

    case_payload = {
        "task": "hunyuan21_direct_edit_image_benchmark_case",
        "created_at": utc_now_iso(),
        "dataset": dataset,
        "object_name": object_name,
        "prompt_id": prompt_id,
        "edit_image": str(edit_image_path),
        "reference_glb": str(reference_glb),
        "glb_path": str(output_glb),
        "white_mesh_faces": white_mesh_faces,
        "white_mesh_vertices": white_mesh_vertices,
        "shape_seconds": shape_seconds,
        "paint_seconds": paint_seconds,
        "convert_seconds": convert_seconds,
        "postprocess_seconds": postprocess_seconds,
        "total_seconds": round(time.time() - started_at, 3),
        "postprocess": postprocess_stats,
        "resumed": False,
    }
    write_json(output_dir / "case.json", case_payload)
    return case_payload


def main() -> None:
    args = build_parser().parse_args()
    args.device = _maybe_reexec_with_visible_device(args.device)
    requested_device = os.environ.get(_ORIGINAL_DEVICE_ENV, args.device)
    if args.case_shard_count < 1:
        raise ValueError("--case-shard-count must be >= 1.")
    if args.case_shard_count > 1 and not args.resume:
        raise RuntimeError(
            "Sharded generation should start from a prepared output dir. "
            "Clean it once, then launch all shards with --resume."
        )

    gt_root = args.gt_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

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
        if not args.resume:
            shutil.rmtree(output_root)
    ensure_dir(output_root)

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

    _ensure_hunyuan_import_path()
    _apply_torchvision_fix()
    _install_dummy_bpy()

    start_time = time.time()
    background_remover, shape_pipeline, paint_pipeline = _load_hunyuan_pipelines(
        model=args.model,
        device=args.device,
    )
    total_cases = len(shard_cases)
    new_case_count = 0
    print(
        f"[Shard] case shard {args.case_shard_index}/{args.case_shard_count} "
        f"selected {total_cases}/{len(cases)} cases "
        f"(requested_device={requested_device}, execution_device={args.device})"
    )
    for case_index, (dataset, object_name, prompt_id) in enumerate(shard_cases, start=1):
        case_id = f"{dataset}/{object_name}/prompt_{prompt_id}"
        case_seed = args.seed + case_index - 1
        print(f"[Hunyuan2.1] {case_index}/{total_cases} {case_id} seed={case_seed}")
        try:
            case_result = _generate_case(
                gt_root=gt_root,
                output_root=output_root,
                dataset=dataset,
                object_name=object_name,
                prompt_id=prompt_id,
                seed=case_seed,
                background_remover=background_remover,
                shape_pipeline=shape_pipeline,
                paint_pipeline=paint_pipeline,
                drop_normal=args.drop_normal,
                keep_materials=args.keep_materials,
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
            release_cuda_memory()
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
                "run_name": output_root.name,
                "method": "hunyuan21_direct_edit_image",
                "model": args.model,
                "requested_device": requested_device,
                "device": args.device,
                "seed": args.seed,
                "case_shard_count": args.case_shard_count,
                "case_shard_index": args.case_shard_index,
                "drop_normal": bool(args.drop_normal),
                "keep_materials": bool(args.keep_materials),
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
                "created_at": utc_now_iso(),
            },
        )
        if args.max_new_cases > 0 and new_case_count >= args.max_new_cases:
            print(
                f"[Stop] Reached max new cases for this run: "
                f"{new_case_count}/{args.max_new_cases}"
            )
            break

    total_time = round(time.time() - start_time, 3)
    summary = {
        "config_name": args.config_name,
        "run_group": args.run_group,
        "run_name": output_root.name,
        "method": "hunyuan21_direct_edit_image",
        "model": args.model,
        "requested_device": requested_device,
        "device": args.device,
        "num_selected_cases": len(cases),
        "num_shard_cases": total_cases,
        "num_generated_cases": len(manifest_cases),
        "num_failed_cases": len(failed_cases),
        "new_cases_this_run": new_case_count,
        "drop_normal": bool(args.drop_normal),
        "keep_materials": bool(args.keep_materials),
        "total_seconds": total_time,
        "created_at": utc_now_iso(),
    }
    write_json(output_root / "summary.json", summary)
    print(
        f"[Done] Hunyuan2.1 direct benchmark generation saved {len(manifest_cases)}/{total_cases} cases "
        f"(new this run: {new_case_count})"
    )
    print(f"[Done] Pred root: {output_root}")


if __name__ == "__main__":
    main()
