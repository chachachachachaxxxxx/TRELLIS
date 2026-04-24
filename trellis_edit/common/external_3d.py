from __future__ import annotations

import contextlib
import os
import random
import sys
import tempfile
import types
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


HUNYUAN_ROOT = Path("/home/wangxinxing/3dlocaledit/Hunyuan3D-2.1")
ULTRASHAPE_ROOT = Path("/home/wangxinxing/3dlocaledit/UltraShape-1.0")
DEFAULT_HUNYUAN_MODEL = "tencent/Hunyuan3D-2.1"
DEFAULT_ULTRASHAPE_CONFIG = ULTRASHAPE_ROOT / "configs" / "infer_dit_refine.yaml"
DEFAULT_ULTRASHAPE_CKPT = ULTRASHAPE_ROOT / "checkpoints" / "ultrashape_v1.pt"
REALESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
_DEVICE_REEXEC_FLAG = "TRELLIS_EXTERNAL_3D_DEVICE_REEXEC"
_ORIGINAL_DEVICE_ENV = "TRELLIS_EXTERNAL_3D_ORIGINAL_DEVICE"


def ensure_hunyuan_import_path() -> None:
    for path in (HUNYUAN_ROOT, HUNYUAN_ROOT / "hy3dshape", HUNYUAN_ROOT / "hy3dpaint"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def ensure_ultrashape_import_path() -> None:
    path_str = str(ULTRASHAPE_ROOT)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def apply_torchvision_fix() -> None:
    try:
        from torchvision_fix import apply_fix

        apply_fix()
    except Exception:
        return


def install_dummy_bpy() -> None:
    if "bpy" not in sys.modules:
        sys.modules["bpy"] = types.ModuleType("bpy")


def setup_external_3d_imports() -> None:
    ensure_hunyuan_import_path()
    ensure_ultrashape_import_path()
    apply_torchvision_fix()
    install_dummy_bpy()


def argv_with_device(argv: list[str], replacement_device: str) -> list[str]:
    rewritten = list(argv)
    if "--device" in rewritten:
        index = rewritten.index("--device")
        if index + 1 >= len(rewritten):
            raise ValueError("--device flag is missing its value.")
        rewritten[index + 1] = replacement_device
        return rewritten
    rewritten.extend(["--device", replacement_device])
    return rewritten


def maybe_reexec_with_visible_device(requested_device: str) -> str:
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
    rewritten_argv = argv_with_device(sys.argv, "cuda:0")
    os.execvpe(sys.executable, [sys.executable, *rewritten_argv], env)
    raise AssertionError("os.execvpe unexpectedly returned")


def requested_device_from_env(default_device: str) -> str:
    return os.environ.get(_ORIGINAL_DEVICE_ENV, default_device)


def ensure_realesrgan_weight() -> Path:
    ckpt_dir = HUNYUAN_ROOT / "hy3dpaint" / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "RealESRGAN_x4plus.pth"
    if not ckpt_path.is_file():
        urllib.request.urlretrieve(REALESRGAN_URL, ckpt_path)
    return ckpt_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_rgba_image(image_path: Path | str, background_remover) -> Image.Image:
    with Image.open(image_path) as handle:
        image = handle.copy()
    if image.mode == "RGB":
        image = background_remover(image)
    else:
        image = image.convert("RGBA")
    return image


def quick_convert_with_obj2gltf(
    obj_path: Path | str,
    glb_path: Path | str,
    *,
    workdir: Path | str | None = None,
) -> None:
    ensure_hunyuan_import_path()
    from hy3dpaint.convert_utils import create_glb_with_pbr_materials

    obj_resolved = Path(obj_path).expanduser().resolve()
    glb_resolved = Path(glb_path).expanduser().resolve()
    textures = {
        "albedo": str(obj_resolved).replace(".obj", ".jpg"),
        "metallic": str(obj_resolved).replace(".obj", "_metallic.jpg"),
        "roughness": str(obj_resolved).replace(".obj", "_roughness.jpg"),
    }

    target_workdir = Path(workdir).expanduser().resolve() if workdir is not None else obj_resolved.parent
    previous_cwd = Path.cwd()
    os.chdir(target_workdir)
    try:
        create_glb_with_pbr_materials(str(obj_resolved), textures, str(glb_resolved))
    finally:
        os.chdir(previous_cwd)


def load_hunyuan_shape_bundle(*, model: str, device: str) -> dict[str, Any]:
    ensure_hunyuan_import_path()
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dshape.rembg import BackgroundRemover

    return {
        "shape_pipeline": Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model, device=device),
        "background_remover": BackgroundRemover(),
    }


def load_hunyuan_paint_bundle(*, device: str, with_background_remover: bool = True) -> dict[str, Any]:
    ensure_hunyuan_import_path()
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

    bundle: dict[str, Any] = {}
    if with_background_remover:
        from hy3dshape.rembg import BackgroundRemover

        bundle["background_remover"] = BackgroundRemover()

    paint_conf = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
    paint_conf.device = device
    paint_conf.realesrgan_ckpt_path = str(ensure_realesrgan_weight())
    paint_conf.multiview_cfg_path = str(HUNYUAN_ROOT / "hy3dpaint" / "cfgs" / "hunyuan-paint-pbr.yaml")
    bundle["paint_pipeline"] = Hunyuan3DPaintPipeline(paint_conf)
    return bundle


def load_ultrashape_refine_bundle(
    *,
    config_path: Path | str,
    ckpt_path: Path | str,
    device: str,
) -> dict[str, Any]:
    ensure_ultrashape_import_path()
    from omegaconf import OmegaConf
    from ultrashape.pipelines import UltraShapePipeline
    from ultrashape.surface_loaders import SharpEdgeSurfaceLoader
    from ultrashape.utils.misc import instantiate_from_config

    config = OmegaConf.load(str(config_path))
    vae = instantiate_from_config(config.model.params.vae_config)
    dit = instantiate_from_config(config.model.params.dit_cfg)
    conditioner = instantiate_from_config(config.model.params.conditioner_config)
    scheduler = instantiate_from_config(config.model.params.scheduler_cfg)
    image_processor = instantiate_from_config(config.model.params.image_processor_cfg)

    weights = torch.load(str(ckpt_path), map_location="cpu")
    vae.load_state_dict(weights["vae"], strict=True)
    dit.load_state_dict(weights["dit"], strict=True)
    conditioner.load_state_dict(weights["conditioner"], strict=True)

    vae.eval().to(device)
    dit.eval().to(device)
    conditioner.eval().to(device)
    if hasattr(vae, "enable_flashvdm_decoder"):
        vae.enable_flashvdm_decoder()

    pipeline = UltraShapePipeline(
        vae=vae,
        model=dit,
        scheduler=scheduler,
        conditioner=conditioner,
        image_processor=image_processor,
    )
    loader = SharpEdgeSurfaceLoader(
        num_sharp_points=204800,
        num_uniform_points=204800,
    )
    return {
        "pipeline": pipeline,
        "loader": loader,
        "voxel_res": int(config.model.params.vae_config.params.voxel_query_res),
        "device": device,
    }


def ultrashape_refine_to_mesh(
    *,
    refine_bundle: dict[str, Any],
    image: Image.Image,
    coarse_mesh_path: Path | str,
    output_mesh_path: Path | str,
    seed: int,
    num_inference_steps: int,
    num_latents: int,
    chunk_size: int,
    octree_res: int,
    normalize_scale: float = 0.99,
) -> dict[str, Any]:
    ensure_ultrashape_import_path()
    from ultrashape.utils import voxelize_from_point

    device = torch.device(refine_bundle["device"])
    output_path = Path(output_mesh_path).expanduser().resolve()
    set_seed(seed)

    surface = refine_bundle["loader"](str(coarse_mesh_path), normalize_scale=normalize_scale).to(
        device,
        dtype=torch.float16,
    )
    point_cloud = surface[:, :, :3]
    _, voxel_idx = voxelize_from_point(point_cloud, num_latents, resolution=refine_bundle["voxel_res"])
    generator = torch.Generator(device).manual_seed(seed)
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )
    with autocast_context:
        meshes, _ = refine_bundle["pipeline"](
            image=image,
            voxel_cond=voxel_idx,
            generator=generator,
            box_v=1.0,
            mc_level=0.0,
            octree_resolution=octree_res,
            num_inference_steps=num_inference_steps,
            num_chunks=chunk_size,
        )

    mesh = meshes[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path)
    return {
        "output_mesh_path": str(output_path),
        "surface_shape": list(surface.shape),
        "refined_mesh_faces": int(mesh.faces.shape[0]),
        "refined_mesh_vertices": int(mesh.vertices.shape[0]),
        "voxel_token_count": int(voxel_idx.shape[1]),
        "voxel_res": int(refine_bundle["voxel_res"]),
        "num_inference_steps": int(num_inference_steps),
        "num_latents": int(num_latents),
        "chunk_size": int(chunk_size),
        "octree_res": int(octree_res),
        "normalize_scale": float(normalize_scale),
    }


def load_ultrashape_vae_bundle(
    *,
    config_path: Path | str,
    ckpt_path: Path | str,
    device: str,
    num_latents: int | None = None,
) -> dict[str, Any]:
    ensure_ultrashape_import_path()
    from omegaconf import OmegaConf
    from ultrashape.surface_loaders import SharpEdgeSurfaceLoader
    from ultrashape.utils.misc import instantiate_from_config

    config = OmegaConf.load(str(config_path))
    vae_params = config.model.params.vae_config.params
    if num_latents is not None:
        vae_params.num_latents = int(num_latents)
    vae = instantiate_from_config(config.model.params.vae_config)
    weights = torch.load(str(ckpt_path), map_location="cpu")
    vae.load_state_dict(weights["vae"], strict=True)
    vae.eval().to(device)
    if hasattr(vae, "enable_flashvdm_decoder"):
        vae.enable_flashvdm_decoder()

    loader = SharpEdgeSurfaceLoader(
        num_sharp_points=204800,
        num_uniform_points=204800,
    )
    return {
        "vae": vae,
        "loader": loader,
        "device": device,
        "num_latents": int(vae_params.num_latents),
        "voxel_res": int(vae_params.get("voxel_query_res", 0) or 0),
    }


def ultrashape_autoencode_to_mesh(
    *,
    vae_bundle: dict[str, Any],
    source_mesh_path: Path | str,
    output_mesh_path: Path | str,
    octree_res: int,
    chunk_size: int,
    normalize_scale: float = 0.99,
) -> dict[str, Any]:
    ensure_ultrashape_import_path()
    from ultrashape.pipelines import export_to_trimesh

    device = torch.device(vae_bundle["device"])
    output_path = Path(output_mesh_path).expanduser().resolve()
    surface = vae_bundle["loader"](str(source_mesh_path), normalize_scale=normalize_scale).to(
        device,
        dtype=torch.float16,
    )

    with torch.inference_mode():
        latents, voxel_idx = vae_bundle["vae"].encode(surface, sample_posterior=False, need_voxel=True)
        decoded_latents = vae_bundle["vae"].decode(latents, voxel_idx=voxel_idx)
        outputs, _ = vae_bundle["vae"].latents2mesh(
            decoded_latents,
            bounds=1.01,
            mc_level=0.0,
            num_chunks=chunk_size,
            octree_resolution=octree_res,
            mc_algo="mc",
            enable_pbar=True,
        )
        meshes = export_to_trimesh(outputs)

    mesh = meshes[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path)
    return {
        "output_mesh_path": str(output_path),
        "surface_shape": list(surface.shape),
        "latents_shape": list(latents.shape),
        "decoded_latents_shape": list(decoded_latents.shape),
        "voxel_token_count": int(voxel_idx.shape[1]),
        "num_latents": int(vae_bundle.get("num_latents") or latents.shape[1]),
        "voxel_res": int(vae_bundle.get("voxel_res") or 0),
        "reconstructed_mesh_faces": int(mesh.faces.shape[0]),
        "reconstructed_mesh_vertices": int(mesh.vertices.shape[0]),
        "chunk_size": int(chunk_size),
        "octree_res": int(octree_res),
        "normalize_scale": float(normalize_scale),
    }


def write_temp_obj_dir(prefix: str, temp_root: Path | str | None = None) -> tempfile.TemporaryDirectory[str]:
    if temp_root is None:
        return tempfile.TemporaryDirectory(prefix=prefix)
    resolved_root = Path(temp_root).expanduser().resolve()
    resolved_root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix=prefix, dir=resolved_root)
