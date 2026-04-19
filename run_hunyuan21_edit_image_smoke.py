#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import types
import urllib.request
from pathlib import Path

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


def _apply_torchvision_fix() -> None:
    try:
        from torchvision_fix import apply_fix

        apply_fix()
    except Exception:
        return


def _install_dummy_bpy() -> None:
    if "bpy" not in sys.modules:
        sys.modules["bpy"] = types.ModuleType("bpy")


def _select_first_prompt_case(gt_root: Path) -> tuple[str, str, int]:
    metadata_path = gt_root / "metadata.json"
    rows = json.loads(metadata_path.read_text(encoding="utf-8"))
    for row in rows:
        dataset = str(row["dataset"])
        object_name = str(row["source_model"])
        for prompt_id in (1, 2, 3):
            edit_image = gt_root / dataset / object_name / f"prompt_{prompt_id}" / "2d_edit.png"
            if edit_image.is_file():
                return dataset, object_name, prompt_id
    raise FileNotFoundError(f"No prompt_*/2d_edit.png found under {gt_root}")


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
        description="Run a single Hunyuan3D-2.1 direct edit-image generation smoke test on the Edit3D benchmark.",
    )
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--object-name", type=str, default="")
    parser.add_argument("--prompt-id", type=int, default=1)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _ensure_hunyuan_import_path()
    _apply_torchvision_fix()
    _install_dummy_bpy()

    from hy3dshape.rembg import BackgroundRemover
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

    dataset = args.dataset
    object_name = args.object_name
    prompt_id = args.prompt_id
    if not dataset or not object_name:
        dataset, object_name, prompt_id = _select_first_prompt_case(args.gt_root)

    edit_image_path = args.gt_root / dataset / object_name / f"prompt_{prompt_id}" / "2d_edit.png"
    if not edit_image_path.is_file():
        raise FileNotFoundError(f"Missing edit image: {edit_image_path}")

    out_dir = ensure_dir(
        args.output_root / "hunyuan21_edit_image_direct_smoke" / dataset / object_name / f"prompt_{prompt_id}"
    )
    white_mesh_path = out_dir / "white_mesh.glb"
    textured_obj_path = out_dir / "textured_mesh.obj"
    textured_glb_path = out_dir / "textured_mesh.glb"

    payload = {
        "task": "hunyuan21_edit_image_direct_smoke",
        "created_at": utc_now_iso(),
        "dataset": dataset,
        "object_name": object_name,
        "prompt_id": prompt_id,
        "edit_image_path": str(edit_image_path),
        "output_white_mesh": str(white_mesh_path),
        "output_textured_glb": str(textured_glb_path),
        "model": args.model,
        "shape_params": "pipeline defaults from Hunyuan3DDiTFlowMatchingPipeline.__call__",
        "paint_params": {
            "max_num_view": 6,
            "resolution": 512,
        },
    }

    realesrgan_ckpt = _ensure_realesrgan_weight()

    image = Image.open(edit_image_path)
    if image.mode == "RGB":
        rembg = BackgroundRemover()
        image = rembg(image)
    else:
        image = image.convert("RGBA")

    started_at = time.time()
    shape_started = time.time()
    shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(args.model)
    mesh = shape_pipeline(image=image)[0]
    payload["shape_seconds"] = round(time.time() - shape_started, 3)
    white_mesh_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(white_mesh_path)
    payload["white_mesh_faces"] = int(mesh.faces.shape[0])
    payload["white_mesh_vertices"] = int(mesh.vertices.shape[0])

    paint_started = time.time()
    conf = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
    conf.realesrgan_ckpt_path = str(realesrgan_ckpt)
    conf.multiview_cfg_path = str(HUNYUAN_ROOT / "hy3dpaint" / "cfgs" / "hunyuan-paint-pbr.yaml")
    paint_pipeline = Hunyuan3DPaintPipeline(conf)
    paint_pipeline(
        mesh_path=str(white_mesh_path),
        image_path=image,
        output_mesh_path=str(textured_obj_path),
        save_glb=False,
    )
    payload["paint_seconds"] = round(time.time() - paint_started, 3)

    convert_started = time.time()
    _quick_convert_with_obj2gltf(textured_obj_path, textured_glb_path)
    payload["convert_seconds"] = round(time.time() - convert_started, 3)
    payload["total_seconds"] = round(time.time() - started_at, 3)

    write_json(out_dir / "run.json", payload)
    print(f"[Done] Direct edit-image smoke output: {textured_glb_path}")

    release_cuda_memory()


if __name__ == "__main__":
    main()
