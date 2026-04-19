#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from run_batch_edit_and_eval import (
    DEFAULT_METRICS,
    render_all_results,
    run_evaluation,
    save_results,
)
from trellis_edit.common import ensure_dir, rewrite_glb_materials_to_matte_nonmetal, write_json


DEFAULT_GT_ROOT = Path("/cache/wangxinxing/data/trellis_edit_benchmark/edit3d_data/data")
DEFAULT_OUTPUT_ROOT = Path("/cache/wangxinxing/data/trellis_edit_benchmark/pred/trellis2_autoencode")
DEFAULT_MODEL = "microsoft/TRELLIS.2-4B"
DEFAULT_RESOLUTION = 1024
DEFAULT_QUALITY = "adaptive"
TRELLIS2_REPO = Path("/home/wangxinxing/3dlocaledit/TRELLIS.2")
BLENDER_LINK = "https://ftp.halifax.rwth-aachen.de/blender/release/Blender4.5/blender-4.5.1-linux-x64.tar.xz"
BLENDER_ARCHIVE = Path("/tmp/blender-4.5.1-linux-x64.tar.xz")
BLENDER_DIR = Path("/home/wangxinxing/opt/blender-5.0.0-linux-x64")
BLENDER_PATH = BLENDER_DIR / "blender"
BLENDER_PILLOW_STAMP = BLENDER_DIR / ".pillow_installed"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark TRELLIS.2 encoder->decoder reconstruction on the Edit3D benchmark.",
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
    parser.add_argument("--config-name", type=str, default="baseline_trellis2_autoencode")
    parser.add_argument("--run-group", type=str, default="baseline")
    parser.add_argument(
        "--quality",
        type=str,
        default=DEFAULT_QUALITY,
        choices=("adaptive", "high", "balanced", "balanced_remesh", "lowpoly_25k", "fast"),
        help="Export quality preset. balanced is the recommended fixed preset for full reruns.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Optional autoencode resolution override. Defaults depend on --quality.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from existing object/prompt outputs.")
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only reconstruct object GLBs and prompt edit.glb files. Skip render/eval.",
    )
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
        "--object-list-file",
        type=Path,
        default=None,
        help="Optional text file listing objects to run, one per line as dataset<TAB>object_name or dataset/object_name.",
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
    return parser


def _set_trellis2_env() -> None:
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _ensure_trellis2_import_path() -> None:
    trellis2_path = str(TRELLIS2_REPO.resolve())
    if trellis2_path not in sys.path:
        sys.path.insert(0, trellis2_path)


def _resolved_load_device(requested_device: str) -> str:
    if "CUDA_VISIBLE_DEVICES" in os.environ and requested_device.startswith("cuda:"):
        return "cuda:0"
    return requested_device


def _resolved_torch_device(device: str):
    import torch

    return torch.device(_resolved_load_device(device))


def _release_cuda_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        return


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


def _run_command(cmd: list[str], *, cwd: Path | None = None) -> None:
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def _ensure_blender_ready() -> None:
    if not BLENDER_PATH.exists():
        _run_command(["wget", BLENDER_LINK, "-O", str(BLENDER_ARCHIVE)])
        _run_command(["tar", "-xf", str(BLENDER_ARCHIVE), "-C", str(BLENDER_DIR.parent)])
    if not BLENDER_PILLOW_STAMP.exists():
        install_script = TRELLIS2_REPO / "data_toolkit" / "blender_script" / "install_pillow.py"
        _run_command([str(BLENDER_PATH), "-b", "--python", str(install_script)])
        BLENDER_PILLOW_STAMP.touch()


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


def _load_object_allowlist(path: Path) -> set[tuple[str, str]]:
    allowlist: set[tuple[str, str]] = set()
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                dataset, object_name = line.split("\t", 1)
            else:
                dataset, object_name = line.split("/", 1)
            allowlist.add((dataset.strip(), object_name.strip()))
    return allowlist


def _filter_grouped_cases_by_allowlist(
    grouped_cases: dict[tuple[str, str], list[int]],
    *,
    allowlist: set[tuple[str, str]],
) -> dict[tuple[str, str], list[int]]:
    return {
        key: prompt_ids
        for key, prompt_ids in grouped_cases.items()
        if key in allowlist
    }


def _normalize_vertices(vertices_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    vertices = vertices_np.astype(np.float32)
    vertices_min = vertices.min(axis=0)
    vertices_max = vertices.max(axis=0)
    center = (vertices_min + vertices_max) / 2.0
    scale = 0.99999 / float((vertices_max - vertices_min).max())
    normalized = (vertices - center) * scale
    return normalized, center, scale


def _dump_blender_pickle(
    *,
    script_name: str,
    object_path: Path,
    output_path: Path,
) -> None:
    if output_path.is_file():
        return
    ensure_dir(output_path.parent)
    blender_script = TRELLIS2_REPO / "data_toolkit" / "blender_script" / script_name
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_output = Path(temp_dir) / f"{output_path.stem}.pickle"
        cmd = [
            str(BLENDER_PATH),
            "-b",
            "-P",
            str(blender_script),
            "--",
            "--object",
            str(object_path),
            "--output_path",
            str(temp_output),
        ]
        _run_command(cmd)
        if not temp_output.is_file():
            error_path = Path(f"{temp_output}_error.txt")
            if error_path.is_file():
                raise RuntimeError(error_path.read_text(encoding="utf-8"))
            raise RuntimeError(f"Blender dump did not produce {temp_output}")
        shutil.move(str(temp_output), str(output_path))


def _load_shape_encoder_inputs(mesh_dump_path: Path, *, resolution: int, device: str):
    import torch
    import o_voxel
    from trellis2.modules.sparse import SparseTensor

    with open(mesh_dump_path, "rb") as handle:
        dump = pickle.load(handle)

    start = 0
    vertices_list: list[np.ndarray] = []
    faces_list: list[np.ndarray] = []
    for obj in dump["objects"]:
        obj_vertices = obj["vertices"]
        obj_faces = obj["faces"]
        if obj_vertices.size == 0 or obj_faces.size == 0:
            continue
        vertices_list.append(obj_vertices)
        faces_list.append(obj_faces + start)
        start += len(obj_vertices)
    if not vertices_list or not faces_list:
        raise RuntimeError(f"No mesh geometry found in dump: {mesh_dump_path}")

    vertices_np = np.concatenate(vertices_list, axis=0)
    faces_np = np.concatenate(faces_list, axis=0)
    normalized_vertices, _, _ = _normalize_vertices(vertices_np)
    vertices = torch.from_numpy(normalized_vertices).float()
    faces = torch.from_numpy(faces_np).long()

    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices.cpu(),
        faces.cpu(),
        grid_size=resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0,
        boundary_weight=0.2,
        regularization_weight=1e-2,
        timing=False,
    )
    dual_vertices = dual_vertices * resolution - voxel_indices
    sparse_vertices = SparseTensor(
        feats=dual_vertices.float(),
        coords=torch.cat([torch.zeros_like(voxel_indices[:, 0:1]), voxel_indices], dim=-1),
    ).to(_resolved_torch_device(device))
    sparse_intersected = sparse_vertices.replace(intersected.bool()).to(_resolved_torch_device(device))
    return sparse_vertices, sparse_intersected


def _load_tex_encoder_inputs(pbr_dump_path: Path, *, resolution: int, device: str):
    import torch
    import o_voxel
    from trellis2.modules.sparse import SparseTensor

    with open(pbr_dump_path, "rb") as handle:
        dump = pickle.load(handle)

    for material in dump["materials"]:
        if material["alphaTexture"] is not None and material["alphaMode"] == "OPAQUE":
            material["alphaMode"] = "BLEND"
    dump["materials"].append(
        {
            "baseColorFactor": [0.8, 0.8, 0.8],
            "alphaFactor": 1.0,
            "metallicFactor": 0.0,
            "roughnessFactor": 0.5,
            "alphaMode": "OPAQUE",
            "alphaCutoff": 0.5,
            "baseColorTexture": None,
            "alphaTexture": None,
            "metallicTexture": None,
            "roughnessTexture": None,
        }
    )
    dump["objects"] = [
        obj
        for obj in dump["objects"]
        if obj["vertices"].size != 0 and obj["faces"].size != 0
    ]
    if not dump["objects"]:
        raise RuntimeError(f"No PBR geometry found in dump: {pbr_dump_path}")

    all_vertices = np.concatenate([obj["vertices"] for obj in dump["objects"]], axis=0)
    _, center, scale = _normalize_vertices(all_vertices)
    for obj in dump["objects"]:
        obj["vertices"] = ((obj["vertices"].astype(np.float32) - center) * scale).astype(np.float32)
        obj["mat_ids"][obj["mat_ids"] == -1] = len(dump["materials"]) - 1

    coords, attr = o_voxel.convert.blender_dump_to_volumetric_attr(
        dump,
        grid_size=resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        mip_level_offset=0,
        verbose=False,
        timing=False,
    )
    feats = torch.concat(
        [
            attr["base_color"],
            attr["metallic"],
            attr["roughness"],
            attr["alpha"],
        ],
        dim=-1,
    )
    feats = feats / 255.0 * 2.0 - 1.0
    return SparseTensor(
        feats=feats.float(),
        coords=torch.cat([torch.zeros_like(coords[:, 0:1]), coords], dim=-1),
    ).to(_resolved_torch_device(device))


def _load_models(*, model_name: str, device: str):
    _set_trellis2_env()
    _ensure_trellis2_import_path()

    import trellis2.models as models
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline

    target_device = _resolved_torch_device(device)

    shape_encoder = models.from_pretrained(f"{model_name}/ckpts/shape_enc_next_dc_f16c32_fp16").eval()
    tex_encoder = models.from_pretrained(f"{model_name}/ckpts/tex_enc_next_dc_f16c32_fp16").eval()
    decoder = Trellis2ImageTo3DPipeline(
        models={
            "shape_slat_decoder": models.from_pretrained(f"{model_name}/ckpts/shape_dec_next_dc_f16c32_fp16"),
            "tex_slat_decoder": models.from_pretrained(f"{model_name}/ckpts/tex_dec_next_dc_f16c32_fp16"),
        },
        sparse_structure_sampler=None,
        shape_slat_sampler=None,
        tex_slat_sampler=None,
        sparse_structure_sampler_params=None,
        shape_slat_sampler_params=None,
        tex_slat_sampler_params=None,
        shape_slat_normalization=None,
        tex_slat_normalization=None,
        image_cond_model=None,
        rembg_model=None,
        low_vram=True,
        default_pipeline_type="1024_cascade",
    )
    decoder._device = target_device
    decoder.pbr_attr_layout = {
        "base_color": slice(0, 3),
        "metallic": slice(3, 4),
        "roughness": slice(4, 5),
        "alpha": slice(5, 6),
    }
    shape_encoder.to(target_device)
    tex_encoder.to(target_device)
    return {
        "shape_encoder": shape_encoder,
        "tex_encoder": tex_encoder,
        "decoder": decoder,
    }


def _export_glb(mesh_with_voxel, *, glb_path: Path) -> None:
    import o_voxel
    import torch

    ensure_dir(glb_path.parent)
    export_attempts = getattr(_export_glb, "_attempts", None)
    if export_attempts is None:
        raise RuntimeError("GLB export attempts have not been configured.")

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
            print(f"[Export] {attempt['name']} failed ({exc}), retrying with safer settings")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            last_error = exc
            raise

    assert last_error is not None
    raise last_error


def _reconstruct_object(
    *,
    models_bundle,
    gt_root: Path,
    output_root: Path,
    resolution: int,
    device: str,
    dataset: str,
    object_name: str,
    prompt_ids: list[int],
    drop_normal: bool,
) -> dict[str, Any]:
    import torch

    object_glb = gt_root / dataset / object_name / "source_model" / "model.glb"
    if not object_glb.is_file():
        raise FileNotFoundError(f"Missing source GLB: {object_glb}")

    object_output_dir = output_root / "_object_recon" / dataset / object_name
    object_glb_out = object_output_dir / "sample_00.glb"
    prompt_output_dirs = [
        output_root / dataset / object_name / f"prompt_{prompt_id}"
        for prompt_id in prompt_ids
    ]
    if object_glb_out.is_file() and all((prompt_dir / "edit.glb").is_file() for prompt_dir in prompt_output_dirs):
        return {
            "source_glb": str(object_glb),
            "object_output_dir": str(object_output_dir),
            "glb_path": str(object_glb_out),
            "prompt_glb_paths": [str(prompt_dir / "edit.glb") for prompt_dir in prompt_output_dirs],
            "resumed": True,
        }

    cache_root = ensure_dir(output_root / "_cache" / dataset / object_name)
    mesh_dump_path = cache_root / "mesh_dump.pickle"
    pbr_dump_path = cache_root / "pbr_dump.pickle"
    _dump_blender_pickle(script_name="dump_mesh.py", object_path=object_glb, output_path=mesh_dump_path)
    _dump_blender_pickle(script_name="dump_pbr.py", object_path=object_glb, output_path=pbr_dump_path)

    prompt_glb_paths: list[str] = []
    with torch.no_grad():
        shape_vertices, shape_intersected = _load_shape_encoder_inputs(
            mesh_dump_path,
            resolution=resolution,
            device=device,
        )
        shape_slat = models_bundle["shape_encoder"](shape_vertices, shape_intersected)
        del shape_vertices
        del shape_intersected
        _release_cuda_memory()

        tex_voxels = _load_tex_encoder_inputs(
            pbr_dump_path,
            resolution=resolution,
            device=device,
        )
        tex_slat = models_bundle["tex_encoder"](tex_voxels)
        del tex_voxels
        _release_cuda_memory()

        recon_mesh = models_bundle["decoder"].decode_latent(
            shape_slat,
            tex_slat,
            resolution,
        )[0]
        del shape_slat
        del tex_slat
        _release_cuda_memory()

    _export_glb(recon_mesh, glb_path=object_glb_out)
    del recon_mesh
    _release_cuda_memory()
    material_postprocess = rewrite_glb_materials_to_matte_nonmetal(
        object_glb_out,
        drop_normal=drop_normal,
    )

    for prompt_dir in prompt_output_dirs:
        ensure_dir(prompt_dir)
        prompt_glb = prompt_dir / "edit.glb"
        shutil.copy2(object_glb_out, prompt_glb)
        prompt_glb_paths.append(str(prompt_glb))

    return {
        "source_glb": str(object_glb),
        "object_output_dir": str(object_output_dir),
        "glb_path": str(object_glb_out),
        "prompt_glb_paths": prompt_glb_paths,
        "material_postprocess": material_postprocess,
        "resumed": False,
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.object_shard_count < 1:
        raise ValueError("--object-shard-count must be >= 1.")
    if args.object_shard_count > 1 and not args.generate_only:
        raise RuntimeError(
            "Sharded reconstruction must use --generate-only. "
            "Run a final non-sharded --resume pass for render/eval."
        )

    quality_default_resolution = {
        "adaptive": 1024,
        "high": 1024,
        "balanced": 1024,
        "balanced_remesh": 1024,
        "lowpoly_25k": 1024,
        "fast": 512,
    }
    resolution = args.resolution or quality_default_resolution[args.quality]
    export_attempts_by_quality = {
        "adaptive": [
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
        ],
        "high": [
            {
                "name": "high",
                "decimation_target": 1000000,
                "texture_size": 4096,
                "remesh": True,
                "remesh_band": 1,
                "remesh_project": 0,
            }
        ],
        "balanced": [
            {
                "name": "balanced",
                "decimation_target": 500000,
                "texture_size": 2048,
                "remesh": True,
                "remesh_band": 1,
                "remesh_project": 0.9,
            }
        ],
        "balanced_remesh": [
            {
                "name": "balanced_remesh",
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
                "texture_size": 1024,
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
            }
        ],
        "lowpoly_25k": [
            {
                "name": "lowpoly_25k",
                "decimation_target": 25000,
                "texture_size": 256,
                "remesh": False,
                "remesh_band": 1,
                "remesh_project": 0,
            }
        ],
        "fast": [
            {
                "name": "fast",
                "decimation_target": 300000,
                "texture_size": 1024,
                "remesh": False,
                "remesh_band": 1,
                "remesh_project": 0,
            }
        ],
    }
    _export_glb._attempts = export_attempts_by_quality[args.quality]

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
    grouped_cases = _group_cases_by_object(cases)
    total_objects_all = len(grouped_cases)
    if args.object_list_file is not None:
        allowlist = _load_object_allowlist(args.object_list_file.expanduser().resolve())
        grouped_cases = _filter_grouped_cases_by_allowlist(grouped_cases, allowlist=allowlist)
        print(f"[Filter] object allowlist kept {len(grouped_cases)}/{total_objects_all} objects")
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
        if args.object_shard_count > 1 and not args.resume:
            raise RuntimeError(
                "Sharded generation should start from a prepared output dir. "
                "Clean it once, then launch all shards with --resume."
            )
        if not args.resume:
            shutil.rmtree(output_root)
    ensure_dir(output_root)

    start_time = time.time()
    _ensure_blender_ready()
    models_bundle = _load_models(model_name=args.model, device=args.device)

    manifest_objects: dict[str, Any] = {}
    failed_objects: dict[str, Any] = {}
    total_objects = len(grouped_cases)
    print(
        f"[Shard] object shard {args.object_shard_index}/{args.object_shard_count} "
        f"selected {total_objects}/{total_objects_all} objects"
    )
    new_object_count = 0
    for object_index, ((dataset, object_name), prompt_ids) in enumerate(grouped_cases.items(), start=1):
        object_key = f"{dataset}/{object_name}"
        print(
            f"[Autoencode] {object_index}/{total_objects} {object_key} "
            f"-> prompts {','.join(str(item) for item in prompt_ids)}"
        )
        try:
            object_result = _reconstruct_object(
                models_bundle=models_bundle,
                gt_root=gt_root,
                output_root=output_root,
                resolution=resolution,
                device=args.device,
                dataset=dataset,
                object_name=object_name,
                prompt_ids=prompt_ids,
                drop_normal=args.drop_normal,
            )
        except Exception as exc:
            failed_objects[object_key] = {
                "dataset": dataset,
                "object_name": object_name,
                "prompt_ids": list(prompt_ids),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            write_json(output_root / "reconstruct_failures.json", {"objects": failed_objects})
            if not args.continue_on_object_error:
                raise
            print(f"[Skip] {object_key} failed: {exc}")
            _release_cuda_memory()
            continue
        manifest_objects[object_key] = object_result
        if not object_result.get("resumed", False):
            new_object_count += 1
        write_json(
            output_root / "reconstruct_manifest.json",
            {
                "config_name": args.config_name,
                "run_group": args.run_group,
                "model": args.model,
                "device": args.device,
                "eval_device": eval_device,
                "render_gpu_ids": render_gpu_ids,
                "quality": args.quality,
                "resolution": resolution,
                "drop_normal": bool(args.drop_normal),
                "object_shard_count": args.object_shard_count,
                "object_shard_index": args.object_shard_index,
                "metrics": list(args.metrics),
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
            },
        )
        if args.max_new_objects > 0 and new_object_count >= args.max_new_objects:
            print(
                f"[Stop] Reached max new objects for this run: "
                f"{new_object_count}/{args.max_new_objects}"
            )
            break

    if args.generate_only or len(manifest_objects) < total_objects:
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
    print(f"[Done] TRELLIS.2 autoencode benchmark finished in {total_time:.1f}s")
    print(f"[Done] Pred root: {output_root}")


if __name__ == "__main__":
    main()
