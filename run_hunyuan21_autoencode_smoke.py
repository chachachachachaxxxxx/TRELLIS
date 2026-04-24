#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import types
import urllib.request
from pathlib import Path

import torch
from PIL import Image

from trellis_edit.common import ensure_dir, release_cuda_memory, utc_now_iso, write_json


DEFAULT_GT_ROOT = Path("/cache/wangxinxing/data/trellis_edit_benchmark/edit3d_data/data")
DEFAULT_OUTPUT_ROOT = Path("/cache/wangxinxing/data/trellis_edit_benchmark/others")
HUNYUAN_ROOT = Path("/home/wangxinxing/3dlocaledit/Hunyuan3D-2.1")
DEFAULT_MODEL = "tencent/Hunyuan3D-2.1"
REALESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"


def _ensure_hunyuan_import_path() -> None:
    for path in (HUNYUAN_ROOT, HUNYUAN_ROOT / "hy3dshape", HUNYUAN_ROOT / "hy3dpaint"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _select_first_object(gt_root: Path) -> tuple[str, str]:
    metadata_path = gt_root / "metadata.json"
    rows = json.loads(metadata_path.read_text(encoding="utf-8"))
    for row in rows:
        dataset = str(row["dataset"])
        object_name = str(row["source_model"])
        source_glb = gt_root / dataset / object_name / "source_model" / "model.glb"
        if source_glb.is_file():
            return dataset, object_name
    raise FileNotFoundError(f"No source_model/model.glb found under {gt_root}")


def _select_source_render_image(gt_root: Path, dataset: str, object_name: str, preferred_prompt_id: int = 1) -> tuple[int, Path]:
    candidate_prompt_ids = [preferred_prompt_id] + [pid for pid in (1, 2, 3) if pid != preferred_prompt_id]
    for prompt_id in candidate_prompt_ids:
        image_path = gt_root / dataset / object_name / f"prompt_{prompt_id}" / "2d_render.png"
        if image_path.is_file():
            return prompt_id, image_path
    raise FileNotFoundError(
        f"No prompt_*/2d_render.png found for {dataset}/{object_name} under {gt_root}"
    )


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


def _quick_convert_with_obj2gltf(obj_path: Path, glb_path: Path) -> None:
    from hy3dpaint.convert_utils import create_glb_with_pbr_materials

    textures = {
        "albedo": str(obj_path).replace(".obj", ".jpg"),
        "metallic": str(obj_path).replace(".obj", "_metallic.jpg"),
        "roughness": str(obj_path).replace(".obj", "_roughness.jpg"),
    }
    create_glb_with_pbr_materials(str(obj_path), textures, str(glb_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a single Hunyuan3D-2.1 shape+material joint autoencoder smoke test on the Edit3D benchmark.",
    )
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--object-name", type=str, default="")
    parser.add_argument("--reference-prompt-id", type=int, default=1)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _ensure_hunyuan_import_path()
    _apply_torchvision_fix()
    _install_dummy_bpy()

    from hy3dshape.rembg import BackgroundRemover
    from hy3dshape.models.autoencoders import ShapeVAE
    from hy3dshape.pipelines import export_to_trimesh
    from hy3dshape.surface_loaders import SharpEdgeSurfaceLoader
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

    dataset = args.dataset
    object_name = args.object_name
    if not dataset or not object_name:
        dataset, object_name = _select_first_object(args.gt_root)

    source_glb = args.gt_root / dataset / object_name / "source_model" / "model.glb"
    if not source_glb.is_file():
        raise FileNotFoundError(f"Missing source model: {source_glb}")

    reference_prompt_id, source_image_path = _select_source_render_image(
        args.gt_root,
        dataset,
        object_name,
        preferred_prompt_id=args.reference_prompt_id,
    )

    out_dir = ensure_dir(args.output_root / "hunyuan21_joint_autoencode_smoke" / dataset / object_name)
    out_white_glb = out_dir / "reconstruction_white.glb"
    out_textured_obj = out_dir / "reconstruction_textured.obj"
    out_textured_glb = out_dir / "reconstruction_textured.glb"

    started_at = time.time()
    payload = {
        "task": "hunyuan21_joint_autoencode_smoke",
        "created_at": utc_now_iso(),
        "dataset": dataset,
        "object_name": object_name,
        "reference_prompt_id": reference_prompt_id,
        "source_glb": str(source_glb),
        "source_image_path": str(source_image_path),
        "output_white_glb": str(out_white_glb),
        "output_textured_glb": str(out_textured_glb),
        "model": args.model,
        "device": args.device,
        "params": {
            "surface_loader": {
                "num_sharp_points": 0,
                "num_uniform_points": 81920,
            },
            "vae_export": {
                "bounds": 1.01,
                "mc_level": 0.0,
                "num_chunks": 20000,
                "octree_resolution": 256,
                "mc_algo": "mc",
            },
            "paint": {
                "max_num_view": 6,
                "resolution": 512,
            },
        },
    }

    vae = ShapeVAE.from_pretrained(
        args.model,
        device=args.device,
        dtype=torch.float16,
        use_safetensors=False,
        variant="fp16",
    ).eval()
    loader = SharpEdgeSurfaceLoader(
        num_sharp_points=0,
        num_uniform_points=81920,
    )

    surface = loader(str(source_glb)).to(args.device, dtype=torch.float16)
    payload["surface_shape"] = list(surface.shape)

    encode_started = time.time()
    latents = vae.encode(surface)
    payload["encode_seconds"] = round(time.time() - encode_started, 3)
    payload["latents_shape"] = list(latents.shape)

    decode_started = time.time()
    decoded_latents = vae.decode(latents)
    payload["decode_seconds"] = round(time.time() - decode_started, 3)
    payload["decoded_latents_shape"] = list(decoded_latents.shape)

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
    payload["mesh_seconds"] = round(time.time() - mesh_started, 3)
    payload["mesh_vertices"] = int(mesh.vertices.shape[0])
    payload["mesh_faces"] = int(mesh.faces.shape[0])

    mesh.export(out_white_glb)

    source_image = Image.open(source_image_path)
    if source_image.mode == "RGB":
        rembg = BackgroundRemover()
        source_image = rembg(source_image)
    else:
        source_image = source_image.convert("RGBA")

    realesrgan_ckpt = _ensure_realesrgan_weight()
    paint_started = time.time()
    conf = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
    conf.realesrgan_ckpt_path = str(realesrgan_ckpt)
    conf.multiview_cfg_path = str(HUNYUAN_ROOT / "hy3dpaint" / "cfgs" / "hunyuan-paint-pbr.yaml")
    paint_pipeline = Hunyuan3DPaintPipeline(conf)
    paint_pipeline(
        mesh_path=str(out_white_glb),
        image_path=source_image,
        output_mesh_path=str(out_textured_obj),
        save_glb=False,
    )
    payload["paint_seconds"] = round(time.time() - paint_started, 3)

    convert_started = time.time()
    _quick_convert_with_obj2gltf(out_textured_obj, out_textured_glb)
    payload["convert_seconds"] = round(time.time() - convert_started, 3)
    payload["total_time_seconds"] = round(time.time() - started_at, 3)
    write_json(out_dir / "run.json", payload)
    print(f"[Done] Joint autoencoder smoke output: {out_textured_glb}")

    release_cuda_memory()


if __name__ == "__main__":
    main()
