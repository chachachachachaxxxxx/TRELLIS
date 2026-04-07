import argparse
import copy
import json
import os
from pathlib import Path
from typing import Dict, List

import imageio
import numpy as np
import open3d as o3d
import torch
import trimesh
from PIL import Image

from trellis.modules import sparse as sp
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import postprocessing_utils, render_utils
from output_layout import build_output_layout, sanitize_output_name


os.environ.setdefault("SPCONV_ALGO", "native")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize all major intermediates of TRELLIS variant editing."
    )
    parser.add_argument(
        "--model",
        default="microsoft/TRELLIS-text-xlarge",
        help="Hugging Face repo or local model path.",
    )
    parser.add_argument(
        "--mesh",
        default="assets/T.ply",
        help="Base mesh path used for variant editing.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Editing prompt for variant generation.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional explicit output directory. Defaults to outputs/variant_intermediates/<case-name>/",
    )
    parser.add_argument(
        "--case-name",
        default="",
        help="Optional case name used when --output-dir is not provided.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=120)
    parser.add_argument(
        "--export-glb",
        action="store_true",
        help="Also export a textured GLB for the decoded result.",
    )
    parser.add_argument(
        "--slat-steps",
        type=int,
        default=None,
        help="Optional override for slat sampler steps.",
    )
    parser.add_argument(
        "--slat-cfg-strength",
        type=float,
        default=None,
        help="Optional override for slat sampler classifier-free guidance strength.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def tensor_stats(tensor: torch.Tensor) -> Dict[str, float]:
    array = tensor.detach().float().cpu()
    return {
        "shape": list(array.shape),
        "min": float(array.min().item()),
        "max": float(array.max().item()),
        "mean": float(array.mean().item()),
        "std": float(array.std().item()),
    }


def save_json(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.size == 0:
        return np.zeros_like(array, dtype=np.uint8)
    low = float(array.min())
    high = float(array.max())
    if high - low < 1e-8:
        return np.zeros_like(array, dtype=np.uint8)
    scaled = (array - low) / (high - low)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def save_heatmap(path: Path, array: np.ndarray, upscale: int = 4) -> None:
    image = Image.fromarray(normalize_to_uint8(array))
    image = image.resize((image.width * upscale, image.height * upscale), Image.Resampling.NEAREST)
    image.save(path)


def make_projection_strip(coords: np.ndarray, size: int = 256) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    if coords.shape[0] == 0:
        blank = np.zeros((size, size), dtype=np.uint8)
        return np.concatenate([blank, blank, blank], axis=1)

    mins = coords.min(axis=0, keepdims=True)
    maxs = coords.max(axis=0, keepdims=True)
    denom = np.maximum(maxs - mins, 1e-6)
    norm = (coords - mins) / denom

    def project(a: int, b: int) -> np.ndarray:
        canvas = np.zeros((size, size), dtype=np.uint8)
        xy = np.stack([norm[:, a], norm[:, b]], axis=1)
        ij = np.clip((xy * (size - 1)).round().astype(np.int32), 0, size - 1)
        canvas[size - 1 - ij[:, 1], ij[:, 0]] = 255
        return canvas

    xy = project(0, 1)
    xz = project(0, 2)
    yz = project(1, 2)
    return np.concatenate([xy, xz, yz], axis=1)


def compute_pca_colors(features: np.ndarray) -> np.ndarray:
    feats = np.asarray(features, dtype=np.float32)
    if feats.ndim != 2 or feats.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    centered = feats - feats.mean(axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        basis = vh[:3].T
        proj = centered @ basis
    except np.linalg.LinAlgError:
        proj = centered[:, : min(3, centered.shape[1])]
    if proj.shape[1] < 3:
        proj = np.pad(proj, ((0, 0), (0, 3 - proj.shape[1])))
    proj_min = proj.min(axis=0, keepdims=True)
    proj_max = proj.max(axis=0, keepdims=True)
    denom = np.maximum(proj_max - proj_min, 1e-6)
    return (proj - proj_min) / denom


def save_point_cloud(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None and len(colors) == len(points):
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    o3d.io.write_point_cloud(str(path), cloud)


def save_mesh(path: Path, mesh) -> None:
    vertices = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(path)


def save_video(path: Path, frames: List[np.ndarray], fps: int = 30) -> None:
    imageio.mimsave(path, frames, fps=fps)


def save_text_conditioning(
    out_dir: Path,
    pipeline: TrellisTextTo3DPipeline,
    prompt: str,
    cond: Dict[str, torch.Tensor],
) -> None:
    tokenizer = pipeline.text_cond_model["tokenizer"]
    encoding = tokenizer(
        [prompt],
        max_length=77,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    token_ids = encoding["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    cond_tensor = cond["cond"][0].detach().float().cpu()
    neg_tensor = cond["neg_cond"][0].detach().float().cpu()

    np.save(out_dir / "cond.npy", cond_tensor.numpy())
    np.save(out_dir / "neg_cond.npy", neg_tensor.numpy())
    save_heatmap(out_dir / "cond_heatmap.png", cond_tensor.numpy(), upscale=3)
    save_heatmap(out_dir / "neg_cond_heatmap.png", neg_tensor.numpy(), upscale=3)

    token_norms = cond_tensor.norm(dim=-1).numpy()
    rows = []
    for idx, (token, token_id, norm_value) in enumerate(zip(tokens, token_ids, token_norms)):
        rows.append(
            {
                "index": idx,
                "token": token,
                "token_id": token_id,
                "l2_norm": float(norm_value),
            }
        )
    with (out_dir / "tokens.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    save_json(
        out_dir / "summary.json",
        {
            "prompt": prompt,
            "cond": tensor_stats(cond_tensor),
            "neg_cond": tensor_stats(neg_tensor),
        },
    )


def save_voxel_stage(out_dir: Path, original_mesh: o3d.geometry.TriangleMesh, normalized_mesh: o3d.geometry.TriangleMesh, coords: torch.Tensor) -> None:
    o3d.io.write_triangle_mesh(str(out_dir / "base_mesh_original.ply"), original_mesh)
    o3d.io.write_triangle_mesh(str(out_dir / "base_mesh_normalized.ply"), normalized_mesh)

    coords_np = coords.detach().cpu().numpy().astype(np.float32)
    np.save(out_dir / "coords_xyz.npy", coords_np)
    save_point_cloud(out_dir / "coords_xyz.ply", coords_np)

    projection_strip = make_projection_strip(coords_np, size=256)
    Image.fromarray(projection_strip).save(out_dir / "coords_projections.png")

    save_json(
        out_dir / "summary.json",
        {
            "num_voxels": int(coords_np.shape[0]),
            "coord_min": coords_np.min(axis=0).tolist() if len(coords_np) else [0, 0, 0],
            "coord_max": coords_np.max(axis=0).tolist() if len(coords_np) else [0, 0, 0],
        },
    )


def save_sparse_tensor_stage(out_dir: Path, tensor: sp.SparseTensor, name: str) -> None:
    coords = tensor.coords.detach().cpu().numpy()
    feats = tensor.feats.detach().float().cpu().numpy()
    np.save(out_dir / f"{name}_coords.npy", coords)
    np.save(out_dir / f"{name}_feats.npy", feats)

    xyz = coords[:, -3:].astype(np.float32)
    colors = compute_pca_colors(feats.reshape(feats.shape[0], -1))
    save_point_cloud(out_dir / f"{name}_pca_colors.ply", xyz, colors)

    projection_strip = make_projection_strip(xyz, size=256)
    Image.fromarray(projection_strip).save(out_dir / f"{name}_coords_projections.png")

    save_json(
        out_dir / f"{name}_summary.json",
        {
            "coords": {
                "shape": list(coords.shape),
                "min": coords.min(axis=0).tolist() if len(coords) else [],
                "max": coords.max(axis=0).tolist() if len(coords) else [],
            },
            "feats": {
                "shape": list(feats.shape),
                "min": float(feats.min()) if feats.size else 0.0,
                "max": float(feats.max()) if feats.size else 0.0,
                "mean": float(feats.mean()) if feats.size else 0.0,
                "std": float(feats.std()) if feats.size else 0.0,
            },
        },
    )


def save_decoded_outputs(
    out_dir: Path,
    outputs: Dict[str, List],
    resolution: int,
    num_frames: int,
    export_glb: bool,
) -> None:
    sample_dir = out_dir / "sample_0"
    ensure_dir(sample_dir)

    if "mesh" in outputs:
        mesh = outputs["mesh"][0]
        save_mesh(sample_dir / "mesh.ply", mesh)
        mesh_video = render_utils.render_video(mesh, resolution=resolution, num_frames=num_frames)["normal"]
        save_video(sample_dir / "mesh.mp4", mesh_video)
        save_json(
            sample_dir / "mesh_summary.json",
            {
                "num_vertices": int(mesh.vertices.shape[0]),
                "num_faces": int(mesh.faces.shape[0]),
                "success": bool(mesh.success),
            },
        )

    if "gaussian" in outputs:
        gaussian = outputs["gaussian"][0]
        gaussian.save_ply(sample_dir / "gaussian.ply")
        gaussian_video = render_utils.render_video(gaussian, resolution=resolution, num_frames=num_frames)["color"]
        save_video(sample_dir / "gaussian.mp4", gaussian_video)
        save_json(
            sample_dir / "gaussian_summary.json",
            {
                "num_points": int(gaussian.get_xyz.shape[0]),
                "xyz": tensor_stats(gaussian.get_xyz),
                "opacity": tensor_stats(gaussian.get_opacity),
                "scaling": tensor_stats(gaussian.get_scaling),
            },
        )

    if "radiance_field" in outputs:
        radiance_field = outputs["radiance_field"][0]
        rf_video = render_utils.render_video(radiance_field, resolution=resolution, num_frames=num_frames)["color"]
        save_video(sample_dir / "radiance_field.mp4", rf_video)

    if export_glb and "gaussian" in outputs and "mesh" in outputs:
        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            simplify=0.95,
            texture_size=1024,
        )
        glb.export(sample_dir / "sample.glb")


def main() -> None:
    args = parse_args()
    case_name = args.case_name.strip() or sanitize_output_name(
        f"{Path(args.mesh).stem}_{args.prompt}",
        max_len=96,
        fallback="variant_intermediates",
    )
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = build_output_layout("variant_intermediates", case_name).case_dir
    ensure_dir(output_dir)

    slat_sampler_params = {}
    if args.slat_steps is not None:
        slat_sampler_params["steps"] = args.slat_steps
    if args.slat_cfg_strength is not None:
        slat_sampler_params["cfg_strength"] = args.slat_cfg_strength

    pipeline = TrellisTextTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()

    meta = {
        "method_name": "variant_intermediates",
        "case_name": case_name,
        "output_dir": str(output_dir),
        "model": args.model,
        "mesh": args.mesh,
        "prompt": args.prompt,
        "seed": args.seed,
        "num_samples": args.num_samples,
        "resolution": args.resolution,
        "num_frames": args.num_frames,
        "slat_sampler_params": slat_sampler_params,
    }
    save_json(output_dir / "run_config.json", meta)

    cond_dir = output_dir / "01_conditioning"
    voxel_dir = output_dir / "02_voxelized_mesh"
    slat_dir = output_dir / "03_slat"
    decoded_dir = output_dir / "04_decoded"
    for path in [cond_dir, voxel_dir, slat_dir, decoded_dir]:
        ensure_dir(path)

    cond = pipeline.get_cond([args.prompt])
    save_text_conditioning(cond_dir, pipeline, args.prompt, cond)

    base_mesh_original = o3d.io.read_triangle_mesh(args.mesh)
    base_mesh_for_voxel = copy.deepcopy(base_mesh_original)
    coords_xyz = pipeline.voxelize(base_mesh_for_voxel)
    save_voxel_stage(voxel_dir, base_mesh_original, base_mesh_for_voxel, coords_xyz)

    coords = torch.cat(
        [
            torch.arange(args.num_samples, device=coords_xyz.device)
            .repeat_interleave(coords_xyz.shape[0], dim=0)[:, None]
            .int(),
            coords_xyz.repeat(args.num_samples, 1),
        ],
        dim=1,
    )
    np.save(slat_dir / "batched_coords.npy", coords.detach().cpu().numpy())

    torch.manual_seed(args.seed)
    flow_model = pipeline.models["slat_flow_model"]
    noise = sp.SparseTensor(
        feats=torch.randn(coords.shape[0], flow_model.in_channels, device=pipeline.device),
        coords=coords,
    )
    save_sparse_tensor_stage(slat_dir, noise, "slat_noise")

    merged_sampler_params = {**pipeline.slat_sampler_params, **slat_sampler_params}
    raw_slat = pipeline.slat_sampler.sample(
        flow_model,
        noise,
        **cond,
        **merged_sampler_params,
        verbose=True,
    ).samples
    save_sparse_tensor_stage(slat_dir, raw_slat, "slat_raw")

    std = torch.tensor(pipeline.slat_normalization["std"], device=raw_slat.device)[None]
    mean = torch.tensor(pipeline.slat_normalization["mean"], device=raw_slat.device)[None]
    normalized_slat = raw_slat * std + mean
    save_sparse_tensor_stage(slat_dir, normalized_slat, "slat_normalized")

    outputs = pipeline.decode_slat(normalized_slat, ["mesh", "gaussian", "radiance_field"])
    save_decoded_outputs(decoded_dir, outputs, args.resolution, args.num_frames, args.export_glb)


if __name__ == "__main__":
    main()
