#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

from trellis_edit.alignment import export_ultrashape_refine_glb_to_canonical_space
from trellis_edit.common import (
    ensure_dir,
    release_cuda_memory,
    utc_now_iso,
    write_json,
)
from trellis_edit.common.external_3d import (
    DEFAULT_HUNYUAN_MODEL,
    DEFAULT_ULTRASHAPE_CKPT,
    DEFAULT_ULTRASHAPE_CONFIG,
    load_hunyuan_paint_bundle,
    load_hunyuan_shape_bundle,
    load_ultrashape_refine_bundle,
    maybe_reexec_with_visible_device,
    prepare_rgba_image,
    quick_convert_with_obj2gltf,
    requested_device_from_env,
    set_seed,
    setup_external_3d_imports,
    ultrashape_refine_to_mesh,
    write_temp_obj_dir,
)


DEFAULT_GT_ROOT = Path("/cache/wangxinxing/data/trellis_edit_benchmark/edit3d_data/data")
DEFAULT_OUTPUT_ROOT = Path("/cache/wangxinxing/data/trellis_edit_benchmark/pred/ultrashape_direct_material_smoke")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a single UltraShape direct-edit smoke case with Hunyuan2.1 coarse generation and "
            "Hunyuan2.1 paint material synthesis, then export a canonically aligned textured edit.glb."
        ),
    )
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--object-name", type=str, default="")
    parser.add_argument("--prompt-id", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--hunyuan-model", type=str, default=DEFAULT_HUNYUAN_MODEL)
    parser.add_argument("--hunyuan-octree-res", type=int, default=256)
    parser.add_argument("--hunyuan-chunk-size", type=int, default=8000)
    parser.add_argument("--ultrashape-config", type=Path, default=DEFAULT_ULTRASHAPE_CONFIG)
    parser.add_argument("--ultrashape-ckpt", type=Path, default=DEFAULT_ULTRASHAPE_CKPT)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--num-latents", type=int, default=8192)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--octree-res", type=int, default=256)
    parser.add_argument("--drop-normal", action="store_true")
    parser.add_argument(
        "--keep-materials",
        action="store_true",
        help="Preserve exported GLB materials instead of rewriting them to matte non-metal.",
    )
    return parser


def _select_first_prompt_case(gt_root: Path) -> tuple[str, str, int]:
    metadata_path = gt_root / "metadata.json"
    rows = json.loads(metadata_path.read_text(encoding="utf-8"))
    for row in rows:
        dataset = str(row["dataset"])
        object_name = str(row["source_model"])
        for prompt_id in (1, 2, 3):
            edit_image = gt_root / dataset / object_name / f"prompt_{prompt_id}" / "2d_edit.png"
            reference_glb = gt_root / dataset / object_name / "source_model" / "model.glb"
            if edit_image.is_file() and reference_glb.is_file():
                return dataset, object_name, prompt_id
    raise FileNotFoundError(f"No prompt_*/2d_edit.png found under {gt_root}")


def main() -> None:
    args = build_parser().parse_args()
    args.device = maybe_reexec_with_visible_device(args.device)
    requested_device = requested_device_from_env(args.device)
    setup_external_3d_imports()

    dataset = args.dataset
    object_name = args.object_name
    prompt_id = args.prompt_id
    if not dataset or not object_name:
        dataset, object_name, prompt_id = _select_first_prompt_case(args.gt_root)

    prompt_dir = args.gt_root / dataset / object_name / f"prompt_{prompt_id}"
    reference_glb = args.gt_root / dataset / object_name / "source_model" / "model.glb"
    edit_image_path = prompt_dir / "2d_edit.png"
    if not edit_image_path.is_file():
        raise FileNotFoundError(f"Missing edit image: {edit_image_path}")
    if not reference_glb.is_file():
        raise FileNotFoundError(f"Missing reference glb: {reference_glb}")

    output_dir = ensure_dir(args.output_root / dataset / object_name / f"prompt_{prompt_id}")
    edit_glb_path = output_dir / "edit.glb"
    run_payload = {
        "task": "ultrashape_direct_edit_material_smoke",
        "created_at": utc_now_iso(),
        "dataset": dataset,
        "object_name": object_name,
        "prompt_id": prompt_id,
        "edit_image_path": str(edit_image_path),
        "reference_glb": str(reference_glb),
        "requested_device": requested_device,
        "device": args.device,
        "hunyuan_model": args.hunyuan_model,
        "hunyuan_octree_res": int(args.hunyuan_octree_res),
        "hunyuan_chunk_size": int(args.hunyuan_chunk_size),
        "ultrashape_config": str(args.ultrashape_config),
        "ultrashape_ckpt": str(args.ultrashape_ckpt),
        "seed": int(args.seed),
        "ultrashape_steps": int(args.steps),
        "ultrashape_num_latents": int(args.num_latents),
        "ultrashape_chunk_size": int(args.chunk_size),
        "ultrashape_octree_res": int(args.octree_res),
        "drop_normal": bool(args.drop_normal),
        "keep_materials": bool(args.keep_materials),
    }

    total_started = time.time()
    image = None
    coarse_mesh = None
    shape_bundle = None
    refine_bundle = None
    paint_bundle = None
    with write_temp_obj_dir(f"{dataset}_{object_name}_prompt_{prompt_id}_ultrashape_direct_") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        coarse_white_mesh_path = temp_dir / "coarse_white.glb"
        refined_white_mesh_path = temp_dir / "refined_white.glb"
        textured_obj_path = temp_dir / "textured_mesh.obj"
        textured_glb_path = temp_dir / "textured_mesh.glb"

        try:
            shape_bundle = load_hunyuan_shape_bundle(model=args.hunyuan_model, device=args.device)
            image = prepare_rgba_image(edit_image_path, shape_bundle["background_remover"])

            coarse_started = time.time()
            set_seed(args.seed)
            coarse_mesh = shape_bundle["shape_pipeline"](
                image=image,
                octree_resolution=args.hunyuan_octree_res,
                num_chunks=args.hunyuan_chunk_size,
            )[0]
            coarse_mesh.export(coarse_white_mesh_path)
            run_payload["coarse_shape_seconds"] = round(time.time() - coarse_started, 3)
            run_payload["coarse_white_mesh_path"] = str(coarse_white_mesh_path)
            run_payload["coarse_white_mesh_faces"] = int(coarse_mesh.faces.shape[0])
            run_payload["coarse_white_mesh_vertices"] = int(coarse_mesh.vertices.shape[0])
        finally:
            del coarse_mesh
            del shape_bundle
            gc.collect()
            release_cuda_memory()

        refine_started = time.time()
        try:
            refine_bundle = load_ultrashape_refine_bundle(
                config_path=args.ultrashape_config,
                ckpt_path=args.ultrashape_ckpt,
                device=args.device,
            )
            refine_stats = ultrashape_refine_to_mesh(
                refine_bundle=refine_bundle,
                image=image,
                coarse_mesh_path=coarse_white_mesh_path,
                output_mesh_path=refined_white_mesh_path,
                seed=args.seed,
                num_inference_steps=args.steps,
                num_latents=args.num_latents,
                chunk_size=args.chunk_size,
                octree_res=args.octree_res,
            )
            run_payload.update(refine_stats)
            run_payload["ultrashape_refine_seconds"] = round(time.time() - refine_started, 3)
        finally:
            del refine_bundle
            gc.collect()
            release_cuda_memory()

        paint_started = time.time()
        try:
            paint_bundle = load_hunyuan_paint_bundle(device=args.device, with_background_remover=False)
            paint_bundle["paint_pipeline"](
                mesh_path=str(refined_white_mesh_path),
                image_path=image,
                output_mesh_path=str(textured_obj_path),
                save_glb=False,
            )
            run_payload["paint_seconds"] = round(time.time() - paint_started, 3)
            run_payload["textured_obj_path"] = str(textured_obj_path)
        finally:
            del paint_bundle
            gc.collect()
            release_cuda_memory()

        convert_started = time.time()
        quick_convert_with_obj2gltf(textured_obj_path, textured_glb_path, workdir=temp_dir)
        run_payload["convert_seconds"] = round(time.time() - convert_started, 3)
        run_payload["textured_glb_path"] = str(textured_glb_path)

        postprocess_started = time.time()
        postprocess = export_ultrashape_refine_glb_to_canonical_space(
            input_glb=textured_glb_path,
            output_glb=edit_glb_path,
            apply_matte_nonmetal=not args.keep_materials,
            drop_normal=args.drop_normal,
        )
        run_payload["postprocess_seconds"] = round(time.time() - postprocess_started, 3)
        run_payload["postprocess"] = postprocess

    del image
    gc.collect()
    release_cuda_memory()

    run_payload["edit_glb_path"] = str(edit_glb_path)
    run_payload["total_time_seconds"] = round(time.time() - total_started, 3)
    write_json(output_dir / "run.json", run_payload)
    print(f"[Done] UltraShape direct textured smoke output: {edit_glb_path}")


if __name__ == "__main__":
    main()
