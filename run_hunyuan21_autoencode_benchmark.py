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
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from trellis_edit.common import (
    ensure_dir,
    export_aligned_glb_to_reference,
    release_cuda_memory,
    utc_now_iso,
    write_json,
)


DEFAULT_GT_ROOT = Path("/cache/wangxinxing/data/trellis_edit_benchmark/edit3d_data/data")
DEFAULT_OUTPUT_ROOT = Path("/cache/wangxinxing/data/trellis_edit_benchmark/pred/baseline_hunyuan21_autoencode")
HUNYUAN_ROOT = Path("/home/wangxinxing/3dlocaledit/Hunyuan3D-2.1")
DEFAULT_MODEL = "tencent/Hunyuan3D-2.1"
REALESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
_DEVICE_REEXEC_FLAG = "TRELLIS_HUNYUAN21_DEVICE_REEXEC"
_ORIGINAL_DEVICE_ENV = "TRELLIS_HUNYUAN21_ORIGINAL_DEVICE"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Hunyuan3D-2.1 autoencoder reconstruction on Edit3D-Bench source GLBs, "
            "generate textured reconstructions, align them to the source-model bbox, "
            "and save pred-ready prompt edit.glb files."
        ),
    )
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--limit", type=int, default=300, help="Number of prompt-level cases to include.")
    parser.add_argument("--reference-prompt-id", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Resume from existing object/prompt outputs.")
    parser.add_argument(
        "--max-new-objects",
        type=int,
        default=0,
        help="Stop after generating this many new objects. 0 means no limit.",
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
    parser.add_argument(
        "--continue-on-object-error",
        action="store_true",
        help="Log per-object failures and continue instead of aborting the whole run.",
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
    parser.add_argument("--config-name", type=str, default="baseline_hunyuan21_autoencode")
    parser.add_argument("--run-group", type=str, default="baseline")
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
        source_glb = gt_root / dataset / object_name / "source_model" / "model.glb"
        if not source_glb.is_file():
            continue
        for prompt_id in (1, 2, 3):
            prompt_dir = gt_root / dataset / object_name / f"prompt_{prompt_id}"
            edit_prompt = prompt_dir / "2d_edit.png"
            render_prompt = prompt_dir / "2d_render.png"
            if not prompt_dir.is_dir() or not edit_prompt.is_file() or not render_prompt.is_file():
                continue
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
        raise ValueError(f"Invalid shard index {shard_index} for shard count {shard_count}.")
    shard_items = [
        item
        for object_idx, item in enumerate(grouped_cases.items())
        if object_idx % shard_count == shard_index
    ]
    return dict(shard_items)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_source_render_image(
    gt_root: Path,
    dataset: str,
    object_name: str,
    *,
    preferred_prompt_id: int,
    prompt_ids: list[int],
) -> tuple[int, Path]:
    candidate_prompt_ids = [preferred_prompt_id] + [pid for pid in prompt_ids if pid != preferred_prompt_id]
    for prompt_id in candidate_prompt_ids:
        image_path = gt_root / dataset / object_name / f"prompt_{prompt_id}" / "2d_render.png"
        if image_path.is_file():
            return prompt_id, image_path
    raise FileNotFoundError(
        f"No prompt_*/2d_render.png found for {dataset}/{object_name} under {gt_root}"
    )


def _prepare_source_image(source_image_path: Path, background_remover) -> Image.Image:
    with Image.open(source_image_path) as handle:
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


def _load_hunyuan_models(*, model: str, device: str):
    from hy3dshape.models.autoencoders import ShapeVAE
    from hy3dshape.rembg import BackgroundRemover
    from hy3dshape.surface_loaders import SharpEdgeSurfaceLoader
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

    vae = ShapeVAE.from_pretrained(
        model,
        device=device,
        dtype=torch.float16,
        use_safetensors=False,
        variant="fp16",
    ).eval()
    loader = SharpEdgeSurfaceLoader(num_sharp_points=0, num_uniform_points=81920)
    realesrgan_ckpt = _ensure_realesrgan_weight()
    paint_conf = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
    paint_conf.device = device
    paint_conf.realesrgan_ckpt_path = str(realesrgan_ckpt)
    paint_conf.multiview_cfg_path = str(HUNYUAN_ROOT / "hy3dpaint" / "cfgs" / "hunyuan-paint-pbr.yaml")
    paint_pipeline = Hunyuan3DPaintPipeline(paint_conf)
    background_remover = BackgroundRemover()
    return {
        "vae": vae,
        "loader": loader,
        "paint_pipeline": paint_pipeline,
        "background_remover": background_remover,
    }


def _reconstruct_object(
    *,
    models_bundle: dict[str, Any],
    gt_root: Path,
    output_root: Path,
    dataset: str,
    object_name: str,
    prompt_ids: list[int],
    preferred_prompt_id: int,
    drop_normal: bool,
    keep_materials: bool,
    seed: int,
) -> dict[str, Any]:
    from hy3dshape.pipelines import export_to_trimesh

    source_glb = gt_root / dataset / object_name / "source_model" / "model.glb"
    if not source_glb.is_file():
        raise FileNotFoundError(f"Missing source model: {source_glb}")

    object_output_dir = output_root / "_object_recon" / dataset / object_name
    prompt_output_dirs = [output_root / dataset / object_name / f"prompt_{prompt_id}" for prompt_id in prompt_ids]
    object_glb_out = object_output_dir / "reconstruction_textured.glb"
    case_payload_out = object_output_dir / "object_case.json"
    all_prompt_outputs_exist = all((prompt_dir / "edit.glb").is_file() for prompt_dir in prompt_output_dirs)
    if object_glb_out.is_file() and all_prompt_outputs_exist:
        return {
            "source_glb": str(source_glb),
            "reference_prompt_id": None,
            "source_image_path": "",
            "object_glb_path": str(object_glb_out),
            "prompt_glb_paths": [str(prompt_dir / "edit.glb") for prompt_dir in prompt_output_dirs],
            "resumed": True,
        }

    reference_prompt_id, source_image_path = _select_source_render_image(
        gt_root,
        dataset,
        object_name,
        preferred_prompt_id=preferred_prompt_id,
        prompt_ids=prompt_ids,
    )

    ensure_dir(object_output_dir)
    temp_root = ensure_dir(output_root / "_tmp")
    with tempfile.TemporaryDirectory(prefix=f"{dataset}_{object_name}_autoencode_", dir=temp_root) as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        white_mesh_path = temp_dir / "reconstruction_white.glb"
        textured_obj_path = temp_dir / "reconstruction_textured.obj"
        textured_glb_path = temp_dir / "reconstruction_textured.glb"

        vae = models_bundle["vae"]
        loader = models_bundle["loader"]
        paint_pipeline = models_bundle["paint_pipeline"]
        background_remover = models_bundle["background_remover"]
        device = models_bundle["device"]

        _set_seed(seed)
        source_image = None
        surface = None
        latents = None
        decoded_latents = None
        mesh = None
        surface_shape = None
        latents_shape = None
        decoded_latents_shape = None
        started_at = time.time()
        try:
            surface = loader(str(source_glb)).to(device, dtype=torch.float16)
            surface_shape = list(surface.shape)

            encode_started = time.time()
            latents = vae.encode(surface)
            encode_seconds = round(time.time() - encode_started, 3)
            latents_shape = list(latents.shape)

            decode_started = time.time()
            decoded_latents = vae.decode(latents)
            decode_seconds = round(time.time() - decode_started, 3)
            decoded_latents_shape = list(decoded_latents.shape)

            mesh_started = time.time()
            mesh = vae.latents2mesh(
                decoded_latents,
                output_type="trimesh",
                bounds=1.01,
                mc_level=0.0,
                num_chunks=20000,
                octree_resolution=256,
                mc_algo="mc",
                enable_pbar=True,
            )
            mesh = export_to_trimesh(mesh)[0]
            mesh_seconds = round(time.time() - mesh_started, 3)
            white_mesh_faces = int(mesh.faces.shape[0])
            white_mesh_vertices = int(mesh.vertices.shape[0])
            mesh.export(white_mesh_path)

            source_image = _prepare_source_image(source_image_path, background_remover)
            paint_started = time.time()
            paint_pipeline(
                mesh_path=str(white_mesh_path),
                image_path=source_image,
                output_mesh_path=str(textured_obj_path),
                save_glb=False,
            )
            paint_seconds = round(time.time() - paint_started, 3)

            convert_started = time.time()
            _quick_convert_with_obj2gltf(textured_obj_path, textured_glb_path, workdir=temp_dir)
            convert_seconds = round(time.time() - convert_started, 3)

            postprocess_started = time.time()
            postprocess_stats = export_aligned_glb_to_reference(
                input_glb=textured_glb_path,
                reference_glb=source_glb,
                output_glb=object_glb_out,
                apply_matte_nonmetal=not keep_materials,
                drop_normal=drop_normal,
            )
            postprocess_seconds = round(time.time() - postprocess_started, 3)
        finally:
            del mesh
            del decoded_latents
            del latents
            del surface
            del source_image
            gc.collect()
            release_cuda_memory()

    prompt_glb_paths: list[str] = []
    for prompt_output_dir in prompt_output_dirs:
        ensure_dir(prompt_output_dir)
        prompt_glb = prompt_output_dir / "edit.glb"
        shutil.copy2(object_glb_out, prompt_glb)
        prompt_glb_paths.append(str(prompt_glb))

    object_payload = {
        "task": "hunyuan21_autoencode_benchmark_object",
        "created_at": utc_now_iso(),
        "dataset": dataset,
        "object_name": object_name,
        "prompt_ids": list(prompt_ids),
        "reference_prompt_id": reference_prompt_id,
        "source_glb": str(source_glb),
        "source_image_path": str(source_image_path),
        "object_glb_path": str(object_glb_out),
        "prompt_glb_paths": prompt_glb_paths,
        "surface_shape": surface_shape,
        "latents_shape": latents_shape,
        "decoded_latents_shape": decoded_latents_shape,
        "white_mesh_faces": white_mesh_faces,
        "white_mesh_vertices": white_mesh_vertices,
        "encode_seconds": encode_seconds,
        "decode_seconds": decode_seconds,
        "mesh_seconds": mesh_seconds,
        "paint_seconds": paint_seconds,
        "convert_seconds": convert_seconds,
        "postprocess_seconds": postprocess_seconds,
        "total_seconds": round(time.time() - started_at, 3),
        "postprocess": postprocess_stats,
        "resumed": False,
    }
    write_json(case_payload_out, object_payload)
    return object_payload


def main() -> None:
    args = build_parser().parse_args()
    args.device = _maybe_reexec_with_visible_device(args.device)
    requested_device = os.environ.get(_ORIGINAL_DEVICE_ENV, args.device)
    if args.object_shard_count < 1:
        raise ValueError("--object-shard-count must be >= 1.")
    if args.object_shard_count > 1 and not args.resume:
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

    if output_root.exists():
        if not args.resume:
            shutil.rmtree(output_root)
    ensure_dir(output_root)

    manifest_path = output_root / "reconstruct_manifest.json"
    failures_path = output_root / "reconstruct_failures.json"
    manifest_objects: dict[str, Any] = {}
    failed_objects: dict[str, Any] = {}
    if args.resume and manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_objects.update(existing_manifest.get("objects") or {})
        failed_objects.update(existing_manifest.get("failed_objects") or {})
    if args.resume and failures_path.is_file():
        existing_failures = json.loads(failures_path.read_text(encoding="utf-8"))
        failed_objects.update(existing_failures.get("objects") or {})

    _ensure_hunyuan_import_path()
    _apply_torchvision_fix()
    _install_dummy_bpy()

    start_time = time.time()
    models_bundle = _load_hunyuan_models(model=args.model, device=args.device)
    models_bundle["device"] = args.device
    total_objects = len(grouped_cases)
    new_object_count = 0
    print(
        f"[Shard] object shard {args.object_shard_index}/{args.object_shard_count} "
        f"selected {total_objects}/{total_objects_all} objects "
        f"(requested_device={requested_device}, execution_device={args.device})"
    )
    for object_index, ((dataset, object_name), prompt_ids) in enumerate(grouped_cases.items(), start=1):
        object_key = f"{dataset}/{object_name}"
        print(
            f"[Hunyuan2.1 Autoencode] {object_index}/{total_objects} {object_key} "
            f"-> prompts {','.join(str(item) for item in prompt_ids)}"
        )
        try:
            object_result = _reconstruct_object(
                models_bundle=models_bundle,
                gt_root=gt_root,
                output_root=output_root,
                dataset=dataset,
                object_name=object_name,
                prompt_ids=prompt_ids,
                preferred_prompt_id=args.reference_prompt_id,
                drop_normal=args.drop_normal,
                keep_materials=args.keep_materials,
                seed=object_index,
            )
        except Exception as exc:
            failed_objects[object_key] = {
                "dataset": dataset,
                "object_name": object_name,
                "prompt_ids": list(prompt_ids),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            write_json(failures_path, {"objects": failed_objects})
            if not args.continue_on_object_error:
                raise
            print(f"[Skip] {object_key} failed: {exc}")
            release_cuda_memory()
            continue

        manifest_objects[object_key] = object_result
        failed_objects.pop(object_key, None)
        if not object_result.get("resumed", False):
            new_object_count += 1
        if failed_objects:
            write_json(failures_path, {"objects": failed_objects})
        elif failures_path.exists():
            failures_path.unlink()
        write_json(
            manifest_path,
            {
                "config_name": args.config_name,
                "run_group": args.run_group,
                "run_name": output_root.name,
                "method": "hunyuan21_autoencode",
                "model": args.model,
                "requested_device": requested_device,
                "device": args.device,
                "reference_prompt_id": args.reference_prompt_id,
                "object_shard_count": args.object_shard_count,
                "object_shard_index": args.object_shard_index,
                "drop_normal": bool(args.drop_normal),
                "keep_materials": bool(args.keep_materials),
                "cases": [
                    {
                        "dataset": case_dataset,
                        "object_name": case_object_name,
                        "prompt_id": case_prompt_id,
                    }
                    for case_dataset, case_object_name, case_prompt_id in cases
                ],
                "objects": manifest_objects,
                "failed_objects": failed_objects,
                "created_at": utc_now_iso(),
            },
        )
        if args.max_new_objects > 0 and new_object_count >= args.max_new_objects:
            print(
                f"[Stop] Reached max new objects for this run: "
                f"{new_object_count}/{args.max_new_objects}"
            )
            break

    total_time = round(time.time() - start_time, 3)
    summary = {
        "config_name": args.config_name,
        "run_group": args.run_group,
        "run_name": output_root.name,
        "method": "hunyuan21_autoencode",
        "model": args.model,
        "requested_device": requested_device,
        "device": args.device,
        "num_selected_cases": len(cases),
        "num_selected_objects": total_objects_all,
        "num_shard_objects": total_objects,
        "num_generated_objects": len(manifest_objects),
        "num_failed_objects": len(failed_objects),
        "new_objects_this_run": new_object_count,
        "reference_prompt_id": args.reference_prompt_id,
        "drop_normal": bool(args.drop_normal),
        "keep_materials": bool(args.keep_materials),
        "total_seconds": total_time,
        "created_at": utc_now_iso(),
    }
    write_json(output_root / "summary.json", summary)
    print(
        f"[Done] Hunyuan2.1 autoencode generation saved {len(manifest_objects)}/{total_objects} objects "
        f"(new this run: {new_object_count})"
    )
    print(f"[Done] Pred root: {output_root}")


if __name__ == "__main__":
    main()
