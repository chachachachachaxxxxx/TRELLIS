#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any

from run_batch_edit_and_eval import (
    DEFAULT_GT_ROOT,
    DEFAULT_METRICS,
    load_edit3d_metadata,
    parse_gpu_list,
    render_all_results,
    run_evaluation,
    save_results,
)
from trellis_edit.alignment import export_hunyuan21_glb_to_canonical_space
from trellis_edit.benchmarking import DEFAULT_BENCHMARK_ROOT
from trellis_edit.common import ensure_dir, release_cuda_memory, utc_now_iso, write_json
from trellis_edit.common.external_3d import (
    load_hunyuan_paint_bundle,
    maybe_reexec_with_visible_device,
    prepare_rgba_image,
    quick_convert_with_obj2gltf,
    requested_device_from_env,
    set_seed as set_external_3d_seed,
    setup_external_3d_imports,
)


REPO_ROOT = Path(__file__).resolve().parent
ANCHORFLOW_ROOT = Path("/home/wangxinxing/3dlocaledit/AnchorFlow")
DEFAULT_OUTPUT_ROOT = Path("/cache/wangxinxing/data/trellis_edit_benchmark/pred/anchorflow_parent_first12")
DEFAULT_MODEL = "tencent/Hunyuan3D-2.1"
DEFAULT_CONFIG_NAME = "anchorflow_parent_first12"
DEFAULT_RUN_GROUP = "editing_methods"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the parent ../AnchorFlow repo on Edit3D-Bench prompt cases, optionally add "
            "Hunyuan2.1 material generation, canonically align outputs, then optionally "
            "render/evaluate/package into benchmark daily pages."
        ),
    )
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--anchorflow-root", type=Path, default=ANCHORFLOW_ROOT)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--render-gpus", type=str, default="", help="Comma-separated GPU ids for benchmark rendering.")
    parser.add_argument("--eval-device", type=str, default="", help="Evaluation device, defaults to --device.")
    parser.add_argument("--limit", type=int, default=12, help="Number of prompt-level cases to include.")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--object-name", type=str, default="")
    parser.add_argument("--prompt-id", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--config-name", type=str, default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--run-group", type=str, default=DEFAULT_RUN_GROUP)
    parser.add_argument("--resume", action="store_true", help="Skip prompt outputs that already exist.")
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
        help="Log prompt-level failures and continue instead of aborting the whole run.",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate prompt-level edit.glb outputs. Skip render/eval/daily packaging.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the selected cases without executing.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-max", type=int, default=41)
    parser.add_argument("--src-guidance-scale", type=float, default=3.5)
    parser.add_argument("--tar-guidance-scale", type=float, default=7.5)
    parser.add_argument("--anchor-noise", action="store_true")
    parser.add_argument("--inversion", action="store_true")
    parser.add_argument("--infer", action="store_true")
    parser.add_argument(
        "--with-materials",
        action="store_true",
        help="Run Hunyuan2.1 paint material generation on the AnchorFlow raw mesh before alignment.",
    )
    parser.add_argument(
        "--drop-normal",
        action="store_true",
        help="Drop normal textures during canonical alignment.",
    )
    parser.add_argument("--disable-anchorflow", action="store_true", help="Run the parent baseline branch instead.")
    return parser


def _select_cases(
    metadata: list[dict[str, Any]],
    *,
    gt_root: Path,
    limit: int,
    dataset: str,
    object_name: str,
    prompt_id: int,
) -> list[tuple[str, str, int]]:
    cases: list[tuple[str, str, int]] = []
    dataset = dataset.strip()
    object_name = object_name.strip()
    for row in metadata:
        row_dataset = str(row.get("dataset") or "").strip()
        row_object = str(row.get("source_model") or "").strip()
        if not row_dataset or not row_object:
            continue
        if dataset and row_dataset != dataset:
            continue
        if object_name and row_object != object_name:
            continue
        source_model = gt_root / row_dataset / row_object / "source_model" / "model.glb"
        if not source_model.is_file():
            continue
        for candidate_prompt_id in (1, 2, 3):
            if prompt_id and candidate_prompt_id != prompt_id:
                continue
            prompt_dir = gt_root / row_dataset / row_object / f"prompt_{candidate_prompt_id}"
            src_img = prompt_dir / "2d_render.png"
            edit_img = prompt_dir / "2d_edit.png"
            if not src_img.is_file() or not edit_img.is_file():
                continue
            cases.append((row_dataset, row_object, candidate_prompt_id))
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


def _ensure_anchorflow_import_path(anchorflow_root: Path) -> None:
    for path in (
        anchorflow_root,
        anchorflow_root / "src",
        anchorflow_root / "hy3dshape",
        anchorflow_root / "hy3dpaint",
    ):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _load_anchorflow_module(anchorflow_root: Path) -> ModuleType:
    _ensure_anchorflow_import_path(anchorflow_root)
    module_path = anchorflow_root / "src" / "anchorflow.py"
    spec = importlib.util.spec_from_file_location("anchorflow_parent_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load AnchorFlow module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("anchorflow_parent_module", module)
    spec.loader.exec_module(module)
    return module


def _load_anchorflow_bundle(*, anchorflow_root: Path, model: str, device: str) -> dict[str, Any]:
    module = _load_anchorflow_module(anchorflow_root)
    vae = module.Hunyuan3DVAE(model_path=model)
    pipeline = module.Hunyuan3DEdit.from_pretrained(model_path=model, device=device)
    return {
        "module": module,
        "vae": vae,
        "edit_pipeline": pipeline,
    }


def _case_output_glb(output_root: Path, dataset: str, object_name: str, prompt_id: int) -> Path:
    return output_root / dataset / object_name / f"prompt_{prompt_id}" / "edit.glb"


def _case_output_dir(output_root: Path, dataset: str, object_name: str, prompt_id: int) -> Path:
    return output_root / dataset / object_name / f"prompt_{prompt_id}"


def _case_raw_glb(output_root: Path, dataset: str, object_name: str, prompt_id: int) -> Path:
    return _case_output_dir(output_root, dataset, object_name, prompt_id) / "edit_raw_hunyuan21.glb"


def _case_textured_raw_glb(output_root: Path, dataset: str, object_name: str, prompt_id: int) -> Path:
    return _case_output_dir(output_root, dataset, object_name, prompt_id) / "edit_textured_raw_hunyuan21.glb"


def _case_geometry_json(output_root: Path, dataset: str, object_name: str, prompt_id: int) -> Path:
    return _case_output_dir(output_root, dataset, object_name, prompt_id) / "case_geometry.json"


def _case_json(output_root: Path, dataset: str, object_name: str, prompt_id: int) -> Path:
    return _case_output_dir(output_root, dataset, object_name, prompt_id) / "case.json"


def _build_case_context(
    *,
    gt_root: Path,
    output_root: Path,
    dataset: str,
    object_name: str,
    prompt_id: int,
) -> dict[str, Any]:
    prompt_dir = gt_root / dataset / object_name / f"prompt_{prompt_id}"
    output_dir = _case_output_dir(output_root, dataset, object_name, prompt_id)
    return {
        "dataset": dataset,
        "object_name": object_name,
        "prompt_id": prompt_id,
        "source_image_path": prompt_dir / "2d_render.png",
        "edit_image_path": prompt_dir / "2d_edit.png",
        "source_mesh_path": gt_root / dataset / object_name / "source_model" / "model.glb",
        "output_dir": output_dir,
        "raw_glb_path": output_dir / "edit_raw_hunyuan21.glb",
        "textured_raw_glb_path": output_dir / "edit_textured_raw_hunyuan21.glb",
        "edit_glb_path": output_dir / "edit.glb",
        "geometry_json_path": output_dir / "case_geometry.json",
        "case_json_path": output_dir / "case.json",
    }


def _load_case_state(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _case_matches_request(
    case_payload: dict[str, Any] | None,
    *,
    with_materials: bool,
    drop_normal: bool,
) -> bool:
    if case_payload is None:
        return False
    return (
        bool(case_payload.get("with_materials")) == bool(with_materials)
        and bool(case_payload.get("drop_normal")) == bool(drop_normal)
    )


def _basic_case_record(
    *,
    case_context: dict[str, Any],
    seed: int,
    with_materials: bool,
    drop_normal: bool,
) -> dict[str, Any]:
    return {
        "dataset": case_context["dataset"],
        "object_name": case_context["object_name"],
        "prompt_id": case_context["prompt_id"],
        "seed": int(seed),
        "source_image_path": str(case_context["source_image_path"]),
        "edit_image_path": str(case_context["edit_image_path"]),
        "source_mesh_path": str(case_context["source_mesh_path"]),
        "raw_glb_path": str(case_context["raw_glb_path"]),
        "edit_glb_path": str(case_context["edit_glb_path"]),
        "with_materials": bool(with_materials),
        "drop_normal": bool(drop_normal),
        "resumed": True,
    }


def _sync_failures_log(failures_path: Path, failed_cases: dict[str, Any]) -> None:
    if failed_cases:
        write_json(failures_path, {"cases": failed_cases})
    elif failures_path.exists():
        failures_path.unlink()


def _write_manifest(
    *,
    manifest_path: Path,
    requested_device: str,
    args: argparse.Namespace,
    cases: list[tuple[str, str, int]],
    generated_cases: dict[str, Any],
    failed_cases: dict[str, Any],
    skipped_count: int,
) -> None:
    write_json(
        manifest_path,
        {
            "config_name": args.config_name,
            "run_group": args.run_group,
            "run_name": args.output_root.name,
            "method": "anchorflow_parent",
            "model": args.model,
            "requested_device": requested_device,
            "device": args.device,
            "case_shard_count": int(args.case_shard_count),
            "case_shard_index": int(args.case_shard_index),
            "with_materials": bool(args.with_materials),
            "drop_normal": bool(args.drop_normal),
            "success_count": len(generated_cases),
            "skipped_count": int(skipped_count),
            "failed_count": len(failed_cases),
            "cases": [
                {
                    "dataset": case_dataset,
                    "object_name": case_object_name,
                    "prompt_id": case_prompt_id,
                }
                for case_dataset, case_object_name, case_prompt_id in cases
            ],
            "generated_cases": generated_cases,
            "failed_cases": failed_cases,
            "created_at": utc_now_iso(),
        },
    )


def _run_geometry_case(
    *,
    bundle: dict[str, Any],
    case_context: dict[str, Any],
    seed: int,
    n_max: int,
    src_guidance_scale: float,
    tar_guidance_scale: float,
    use_anchorflow: bool,
    anchor_noise: bool,
    inversion: bool,
    infer: bool,
) -> dict[str, Any]:
    module = bundle["module"]
    hunyuan_vae = bundle["vae"]
    hunyuan_3dedit = bundle["edit_pipeline"]

    dataset = str(case_context["dataset"])
    object_name = str(case_context["object_name"])
    prompt_id = int(case_context["prompt_id"])
    src_img_path = Path(case_context["source_image_path"])
    tar_img_path = Path(case_context["edit_image_path"])
    src_mesh_path = Path(case_context["source_mesh_path"])
    output_dir = ensure_dir(Path(case_context["output_dir"]))
    raw_glb_path = Path(case_context["raw_glb_path"])

    if not src_img_path.is_file():
        raise FileNotFoundError(f"Missing source image: {src_img_path}")
    if not tar_img_path.is_file():
        raise FileNotFoundError(f"Missing edit image: {tar_img_path}")
    if not src_mesh_path.is_file():
        raise FileNotFoundError(f"Missing source mesh: {src_mesh_path}")

    started_at = time.time()
    module.set_seed(seed)

    src_img = module.load_image(str(src_img_path), save_path=None)
    tar_img = module.load_image(str(tar_img_path), save_path=None)

    encode_started = time.time()
    latents = hunyuan_vae.encode(str(src_mesh_path))
    encode_seconds = round(time.time() - encode_started, 3)

    edit_started = time.time()
    edit_latents, _ = hunyuan_3dedit.denoise(
        latents,
        src_img,
        tar_img,
        {
            "T_steps": 50,
            "n_max": int(n_max),
            "src_guidance_scale": float(src_guidance_scale),
            "tar_guidance_scale": float(tar_guidance_scale),
            "use_anchorflow": bool(use_anchorflow),
            "anchor_noise": bool(anchor_noise),
            "inversion": bool(inversion),
            "infer": bool(infer),
        },
    )
    edit_seconds = round(time.time() - edit_started, 3)

    decode_started = time.time()
    mesh = hunyuan_vae.decode(edit_latents, save_path=str(raw_glb_path))
    decode_seconds = round(time.time() - decode_started, 3)

    payload = {
        "task": "anchorflow_parent_geometry_case",
        "created_at": utc_now_iso(),
        "dataset": dataset,
        "object_name": object_name,
        "prompt_id": prompt_id,
        "seed": int(seed),
        "source_image_path": str(src_img_path),
        "edit_image_path": str(tar_img_path),
        "source_mesh_path": str(src_mesh_path),
        "raw_glb_path": str(raw_glb_path),
        "params": {
            "model": getattr(hunyuan_3dedit, "from_pretrained_kwargs", {}).get("model_path", DEFAULT_MODEL),
            "T_steps": 50,
            "n_max": int(n_max),
            "src_guidance_scale": float(src_guidance_scale),
            "tar_guidance_scale": float(tar_guidance_scale),
            "use_anchorflow": bool(use_anchorflow),
            "anchor_noise": bool(anchor_noise),
            "inversion": bool(inversion),
            "infer": bool(infer),
        },
        "mesh_faces": int(mesh.faces.shape[0]),
        "mesh_vertices": int(mesh.vertices.shape[0]),
        "encode_seconds": encode_seconds,
        "edit_seconds": edit_seconds,
        "decode_seconds": decode_seconds,
        "total_seconds": round(time.time() - started_at, 3),
        "resumed": False,
    }
    write_json(output_dir / "case_geometry.json", payload)
    return payload


def _finalize_case(
    *,
    gt_root: Path,
    output_root: Path,
    case_context: dict[str, Any],
    geometry_payload: dict[str, Any],
    with_materials: bool,
    drop_normal: bool,
    paint_bundle: dict[str, Any] | None,
) -> dict[str, Any]:
    dataset = str(case_context["dataset"])
    object_name = str(case_context["object_name"])
    prompt_id = int(case_context["prompt_id"])
    output_dir = ensure_dir(Path(case_context["output_dir"]))
    edit_image_path = Path(case_context["edit_image_path"])
    raw_glb_path = Path(geometry_payload.get("raw_glb_path") or case_context["raw_glb_path"])
    final_glb_path = Path(case_context["edit_glb_path"])
    textured_raw_glb_path = Path(case_context["textured_raw_glb_path"])
    seed = int(geometry_payload.get("seed") or 0)

    if not edit_image_path.is_file():
        raise FileNotFoundError(f"Missing edit image: {edit_image_path}")
    if not raw_glb_path.is_file():
        raise FileNotFoundError(f"Missing raw AnchorFlow mesh: {raw_glb_path}")

    postprocess_started = time.time()
    payload = {
        "task": "anchorflow_parent_benchmark_case",
        "created_at": utc_now_iso(),
        "dataset": dataset,
        "object_name": object_name,
        "prompt_id": prompt_id,
        "seed": seed,
        "source_image_path": str(case_context["source_image_path"]),
        "edit_image_path": str(edit_image_path),
        "source_mesh_path": str(case_context["source_mesh_path"]),
        "raw_glb_path": str(raw_glb_path),
        "edit_glb_path": str(final_glb_path),
        "params": dict(geometry_payload.get("params") or {}),
        "mesh_faces": geometry_payload.get("mesh_faces"),
        "mesh_vertices": geometry_payload.get("mesh_vertices"),
        "encode_seconds": geometry_payload.get("encode_seconds"),
        "edit_seconds": geometry_payload.get("edit_seconds"),
        "decode_seconds": geometry_payload.get("decode_seconds"),
        "geometry_total_seconds": geometry_payload.get("total_seconds"),
        "with_materials": bool(with_materials),
        "drop_normal": bool(drop_normal),
        "resumed": False,
    }

    image = None
    material_started = time.time()
    align_input_glb = raw_glb_path
    if with_materials:
        if paint_bundle is None:
            raise RuntimeError("Material finalization requires a loaded Hunyuan2.1 paint bundle.")
        temp_root = ensure_dir(output_root / "_tmp")
        with tempfile.TemporaryDirectory(
            prefix=f"{dataset}_{object_name}_prompt_{prompt_id}_material_",
            dir=temp_root,
        ) as temp_dir_raw:
            temp_dir = Path(temp_dir_raw)
            textured_obj_path = temp_dir / "textured_mesh.obj"
            temp_textured_glb_path = temp_dir / "textured_mesh.glb"

            set_external_3d_seed(seed)
            image = prepare_rgba_image(edit_image_path, paint_bundle["background_remover"])

            paint_started = time.time()
            paint_bundle["paint_pipeline"](
                mesh_path=str(raw_glb_path),
                image_path=image,
                output_mesh_path=str(textured_obj_path),
                save_glb=False,
            )
            payload["paint_seconds"] = round(time.time() - paint_started, 3)

            convert_started = time.time()
            quick_convert_with_obj2gltf(textured_obj_path, temp_textured_glb_path, workdir=temp_dir)
            payload["convert_seconds"] = round(time.time() - convert_started, 3)

            shutil.copy2(temp_textured_glb_path, textured_raw_glb_path)
            payload["textured_raw_glb_path"] = str(textured_raw_glb_path)
            align_input_glb = textured_raw_glb_path

    payload["material_stage_seconds"] = round(time.time() - material_started, 3)

    try:
        align_started = time.time()
        postprocess = export_hunyuan21_glb_to_canonical_space(
            input_glb=align_input_glb,
            output_glb=final_glb_path,
            apply_matte_nonmetal=False,
            drop_normal=drop_normal,
        )
        payload["align_seconds"] = round(time.time() - align_started, 3)
        payload["postprocess"] = postprocess
        payload["postprocess_total_seconds"] = round(time.time() - postprocess_started, 3)
        geometry_total = float(geometry_payload.get("total_seconds") or 0.0)
        payload["total_seconds"] = round(geometry_total + payload["postprocess_total_seconds"], 3)
        write_json(output_dir / "case.json", payload)
        return payload
    finally:
        del image
        gc.collect()
        release_cuda_memory()


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_summary(
    *,
    output_root: Path,
    args: argparse.Namespace,
    cases: list[tuple[str, str, int]],
    total_seconds: float,
    requested_device: str,
) -> dict[str, Any]:
    manifest = _load_manifest(output_root / "manifest.json")
    failed_count = int(manifest.get("failed_count") or 0)
    success_count = int(manifest.get("success_count") or 0)
    skipped_count = int(manifest.get("skipped_count") or 0)
    generated_count = 0
    for dataset, object_name, prompt_id in cases:
        if _case_output_glb(output_root, dataset, object_name, prompt_id).is_file():
            generated_count += 1
    payload = {
        "config_name": args.config_name,
        "run_group": args.run_group,
        "run_name": output_root.name,
        "method": "anchorflow_parent",
        "model": args.model,
        "requested_device": requested_device,
        "device": args.device,
        "eval_device": args.eval_device or args.device,
        "render_gpu_ids": parse_gpu_list(args.render_gpus, fallback_device=args.device)
        if args.render_gpus
        else [args.device.split(":", 1)[1]] if args.device.startswith("cuda:") else [],
        "num_selected_cases": len(cases),
        "case_shard_count": int(args.case_shard_count),
        "case_shard_index": int(args.case_shard_index),
        "num_generated_cases": generated_count,
        "num_success_cases": success_count,
        "num_skipped_cases": skipped_count,
        "num_failed_cases": failed_count,
        "resume": bool(args.resume),
        "generate_only": bool(args.generate_only),
        "metrics": list(args.metrics),
        "dataset": args.dataset or None,
        "object_name": args.object_name or None,
        "prompt_id": int(args.prompt_id) if int(args.prompt_id) > 0 else None,
        "seed": int(args.seed),
        "n_max": int(args.n_max),
        "src_guidance_scale": float(args.src_guidance_scale),
        "tar_guidance_scale": float(args.tar_guidance_scale),
        "use_anchorflow": not bool(args.disable_anchorflow),
        "anchor_noise": bool(args.anchor_noise),
        "inversion": bool(args.inversion),
        "infer": bool(args.infer),
        "with_materials": bool(args.with_materials),
        "drop_normal": bool(args.drop_normal),
        "total_seconds": round(float(total_seconds), 3),
        "created_at": utc_now_iso(),
    }
    write_json(output_root / "summary.json", payload)
    return payload


def main() -> int:
    args = build_parser().parse_args()
    args.device = maybe_reexec_with_visible_device(args.device)
    requested_device = requested_device_from_env(args.device)
    # Hunyuan2.1 paint imports pull in Blender helpers at module import time.
    # Match the repo's existing benchmark scripts by installing the lightweight
    # compatibility shims before any optional material stage is loaded.
    setup_external_3d_imports()
    args.gt_root = args.gt_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.benchmark_root = args.benchmark_root.expanduser().resolve()
    args.anchorflow_root = args.anchorflow_root.expanduser().resolve()

    if args.case_shard_count < 1:
        raise ValueError("--case-shard-count must be >= 1.")
    if args.case_shard_count > 1 and not args.generate_only:
        raise RuntimeError(
            "Sharded AnchorFlow-parent generation must use --generate-only. "
            "Run a final non-sharded --resume pass for render/eval."
        )
    if not args.anchorflow_root.is_dir():
        raise FileNotFoundError(f"AnchorFlow root not found: {args.anchorflow_root}")

    metadata = load_edit3d_metadata(args.gt_root)
    cases = _select_cases(
        metadata,
        gt_root=args.gt_root,
        limit=int(args.limit),
        dataset=args.dataset,
        object_name=args.object_name,
        prompt_id=int(args.prompt_id),
    )
    if not cases:
        raise RuntimeError("No benchmark cases selected.")
    shard_cases = _select_case_shard(
        cases,
        shard_count=int(args.case_shard_count),
        shard_index=int(args.case_shard_index),
    )
    if not shard_cases:
        raise RuntimeError(
            f"No cases selected for shard {args.case_shard_index}/{args.case_shard_count}."
        )

    if args.output_root.exists() and not args.resume:
        shutil.rmtree(args.output_root)
    ensure_dir(args.output_root)

    print("=" * 80)
    print("AnchorFlow Parent Benchmark")
    print("=" * 80)
    print(f"AnchorFlow root: {args.anchorflow_root}")
    print(f"Output root: {args.output_root}")
    print(f"Requested device: {requested_device}")
    print(f"Execution device: {args.device}")
    print(f"Selected cases: {len(cases)} total, {len(shard_cases)} in this run")
    print(f"Resume: {'yes' if args.resume else 'no'}")
    print(f"Generate only: {'yes' if args.generate_only else 'no'}")
    print(f"Materials: {'yes' if args.with_materials else 'no'}")
    print(f"Drop normal: {'yes' if args.drop_normal else 'no'}")
    print(f"Params: n_max={args.n_max}, sgs={args.src_guidance_scale}, tgs={args.tar_guidance_scale}, "
          f"use_anchorflow={'yes' if not args.disable_anchorflow else 'no'}")
    print("=" * 80)

    if args.dry_run:
        print(json.dumps(
            [
                {
                    "dataset": dataset,
                    "object_name": object_name,
                    "prompt_id": prompt_id,
                }
                for dataset, object_name, prompt_id in shard_cases
            ],
            indent=2,
            ensure_ascii=False,
        ))
        return 0

    manifest_path = args.output_root / "manifest.json"
    failures_path = args.output_root / "failures.json"
    manifest = _load_manifest(manifest_path)
    generated_cases: dict[str, Any] = dict(manifest.get("generated_cases") or {})
    failed_cases: dict[str, Any] = dict(manifest.get("failed_cases") or {})
    skipped_count = 0

    started_at = time.time()
    ready_cases: list[tuple[dict[str, Any], dict[str, Any]]] = []
    bundle = None
    try:
        bundle = _load_anchorflow_bundle(
            anchorflow_root=args.anchorflow_root,
            model=args.model,
            device=args.device,
        )
        total_cases = len(shard_cases)
        for case_index, (dataset, object_name, prompt_id) in enumerate(shard_cases, start=1):
            case_id = f"{dataset}/{object_name}/prompt_{prompt_id}"
            case_context = _build_case_context(
                gt_root=args.gt_root,
                output_root=args.output_root,
                dataset=dataset,
                object_name=object_name,
                prompt_id=prompt_id,
            )
            existing_case = _load_case_state(Path(case_context["case_json_path"]))
            case_seed = int(args.seed) + case_index - 1
            if (
                args.resume
                and Path(case_context["edit_glb_path"]).is_file()
                and _case_matches_request(
                    existing_case,
                    with_materials=bool(args.with_materials),
                    drop_normal=bool(args.drop_normal),
                )
            ):
                print(f"[SKIP] {case_id}")
                generated_cases[case_id] = existing_case or _basic_case_record(
                    case_context=case_context,
                    seed=case_seed,
                    with_materials=bool(args.with_materials),
                    drop_normal=bool(args.drop_normal),
                )
                skipped_count += 1
                _sync_failures_log(failures_path, failed_cases)
                _write_manifest(
                    manifest_path=manifest_path,
                    requested_device=requested_device,
                    args=args,
                    cases=cases,
                    generated_cases=generated_cases,
                    failed_cases=failed_cases,
                    skipped_count=skipped_count,
                )
                continue

            geometry_state = _load_case_state(Path(case_context["geometry_json_path"]))
            if args.resume and Path(case_context["raw_glb_path"]).is_file():
                print(f"[Resume Raw] {case_id}")
                geometry_result = geometry_state or {
                    **_basic_case_record(
                        case_context=case_context,
                        seed=case_seed,
                        with_materials=bool(args.with_materials),
                        drop_normal=bool(args.drop_normal),
                    ),
                    "task": "anchorflow_parent_geometry_case",
                    "params": {
                        "model": args.model,
                        "T_steps": 50,
                        "n_max": int(args.n_max),
                        "src_guidance_scale": float(args.src_guidance_scale),
                        "tar_guidance_scale": float(args.tar_guidance_scale),
                        "use_anchorflow": not bool(args.disable_anchorflow),
                        "anchor_noise": bool(args.anchor_noise),
                        "inversion": bool(args.inversion),
                        "infer": bool(args.infer),
                    },
                }
            else:
                print(f"[AnchorFlow/Parent] {case_index}/{total_cases} {case_id} seed={case_seed}")
                try:
                    geometry_result = _run_geometry_case(
                        bundle=bundle,
                        case_context=case_context,
                        seed=case_seed,
                        n_max=int(args.n_max),
                        src_guidance_scale=float(args.src_guidance_scale),
                        tar_guidance_scale=float(args.tar_guidance_scale),
                        use_anchorflow=not bool(args.disable_anchorflow),
                        anchor_noise=bool(args.anchor_noise),
                        inversion=bool(args.inversion),
                        infer=bool(args.infer),
                    )
                except Exception as exc:
                    failed_cases[case_id] = {
                        "dataset": dataset,
                        "object_name": object_name,
                        "prompt_id": prompt_id,
                        "stage": "geometry",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "created_at": utc_now_iso(),
                    }
                    _sync_failures_log(failures_path, failed_cases)
                    _write_manifest(
                        manifest_path=manifest_path,
                        requested_device=requested_device,
                        args=args,
                        cases=cases,
                        generated_cases=generated_cases,
                        failed_cases=failed_cases,
                        skipped_count=skipped_count,
                    )
                    if not args.continue_on_case_error:
                        raise
                    print(f"[Skip] {case_id} geometry failed: {exc}")
                    gc.collect()
                    release_cuda_memory()
                    continue
            ready_cases.append((case_context, geometry_result))
    finally:
        if bundle is not None:
            del bundle
        gc.collect()
        release_cuda_memory()

    paint_bundle = None
    try:
        if args.with_materials and ready_cases:
            print("[Material] Loading Hunyuan2.1 paint bundle")
            paint_bundle = load_hunyuan_paint_bundle(device=args.device, with_background_remover=True)

        total_ready = len(ready_cases)
        for ready_index, (case_context, geometry_result) in enumerate(ready_cases, start=1):
            case_id = (
                f"{case_context['dataset']}/{case_context['object_name']}/prompt_{case_context['prompt_id']}"
            )
            print(
                f"[Finalize] {ready_index}/{total_ready} {case_id} "
                f"materials={'yes' if args.with_materials else 'no'}"
            )
            try:
                case_result = _finalize_case(
                    gt_root=args.gt_root,
                    output_root=args.output_root,
                    case_context=case_context,
                    geometry_payload=geometry_result,
                    with_materials=bool(args.with_materials),
                    drop_normal=bool(args.drop_normal),
                    paint_bundle=paint_bundle,
                )
            except Exception as exc:
                failed_cases[case_id] = {
                    "dataset": case_context["dataset"],
                    "object_name": case_context["object_name"],
                    "prompt_id": case_context["prompt_id"],
                    "stage": "finalize_material" if args.with_materials else "finalize_align",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "created_at": utc_now_iso(),
                }
                generated_cases.pop(case_id, None)
                _sync_failures_log(failures_path, failed_cases)
                _write_manifest(
                    manifest_path=manifest_path,
                    requested_device=requested_device,
                    args=args,
                    cases=cases,
                    generated_cases=generated_cases,
                    failed_cases=failed_cases,
                    skipped_count=skipped_count,
                )
                if not args.continue_on_case_error:
                    raise
                print(f"[Skip] {case_id} finalize failed: {exc}")
                gc.collect()
                release_cuda_memory()
                continue

            generated_cases[case_id] = case_result
            failed_cases.pop(case_id, None)
            _sync_failures_log(failures_path, failed_cases)
            _write_manifest(
                manifest_path=manifest_path,
                requested_device=requested_device,
                args=args,
                cases=cases,
                generated_cases=generated_cases,
                failed_cases=failed_cases,
                skipped_count=skipped_count,
            )
    finally:
        if paint_bundle is not None:
            del paint_bundle
        gc.collect()
        release_cuda_memory()

    _write_summary(
        output_root=args.output_root,
        args=args,
        cases=cases,
        total_seconds=time.time() - started_at,
        requested_device=requested_device,
    )

    if args.generate_only:
        print(f"[Done] Generated AnchorFlow parent outputs at: {args.output_root}")
        return 0

    render_gpu_ids = (
        parse_gpu_list(args.render_gpus, fallback_device=args.device)
        if args.render_gpus
        else [args.device.split(":", 1)[1]] if args.device.startswith("cuda:") else []
    )
    if not render_gpu_ids:
        raise RuntimeError("Render stage requires at least one GPU id.")

    render_success = render_all_results(
        args.output_root,
        gpu_ids=render_gpu_ids,
        metrics=list(args.metrics),
    )
    if not render_success:
        print("[WARNING] Benchmark rendering reported failures.")

    eval_success, results = run_evaluation(
        gt_root=args.gt_root,
        pred_root=args.output_root,
        metrics=list(args.metrics),
        output_dir=args.output_root / "evaluation_output",
        device=args.eval_device or args.device,
    )
    if not eval_success or results is None:
        raise RuntimeError("Evaluation failed.")

    save_results(
        output_root=args.output_root,
        entrypoint_name="anchorflow_parent",
        config_name=args.config_name,
        run_group=args.run_group,
        gt_root=args.gt_root,
        cases=cases,
        requested_metrics=list(args.metrics),
        benchmark_root=args.benchmark_root,
        skip_benchmark_render=False,
        results=results,
        total_time=time.time() - started_at,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
