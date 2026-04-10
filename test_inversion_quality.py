#!/usr/bin/env python3
"""Test inversion quality by comparing inverse and forward trajectories.

This script tests inversion quality across two orthogonal dimensions:
1. Base method: Euler (1st order) vs RF-Solver (2nd order)
2. Corrector steps: 0, 1, 2, 5, ... (refinement iterations per timestep)

The corrector-predictor strategy is NOT a separate method but an orthogonal
enhancement that can be applied to any base method.

Output structure:
outputs/inversion_test/<case_name>/
├── <method_name>/
│   ├── inverse_trajectory/  # Latents from data → noise
│   ├── forward_trajectory/  # Latents from noise → data
│   ├── metrics.json         # Similarity metrics
│   └── config.json          # Method configuration
└── comparison_report.json   # Cross-method comparison
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Dict, List

import torch

# Set environment variables before importing trellis
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

from trellis.pipelines import TrellisImageTo3DPipeline

from editing.inversion.base_inverter import BaseInverter, InversionResult
from editing.inversion.euler_inverter import EulerInverter
from editing.inversion.rf_solver_inverter import RFSolverInverter
from editing.inversion.trajectory_metrics import compute_trajectory_similarity
from editing.preprocess.asset_3d import feats_to_slat, ply_to_coords


def parse_args():
    parser = argparse.ArgumentParser(description="Test inversion quality")
    parser.add_argument("--source-voxels", type=str, required=True, help="Path to source voxels PLY")
    parser.add_argument("--source-features", type=str, required=True, help="Path to source features PT")
    parser.add_argument("--source-image", type=str, required=True, help="Path to source image")
    parser.add_argument("--case-name", type=str, required=True, help="Test case name")
    parser.add_argument("--stage", choices=["ss", "slat"], default="slat", help="Stage to test (ss=sparse structure, slat=SLAT)")
    parser.add_argument("--methods", nargs="+", default=["euler", "rf"], help="Methods to test")
    parser.add_argument("--corrector-steps", type=int, default=0, help="Corrector steps for iterative refinement")
    parser.add_argument("--steps", type=int, default=12, help="Number of sampling steps")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    return parser.parse_args()


def setup_pipeline(seed: int):
    """Initialize TRELLIS pipeline."""
    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipeline.cuda()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return pipeline


def load_source_assets(args, pipeline):
    """Load source assets for testing."""
    resolution = pipeline.sparse_structure_sampler_params.get("grid_size", 64)

    if args.stage == "ss":
        # Test sparse structure stage
        coords = ply_to_coords(args.source_voxels, pipeline.device, resolution)
        from editing.preprocess.asset_3d import coords_to_voxel
        voxel = coords_to_voxel(coords, pipeline.device, resolution)
        encoder = pipeline.models["sparse_structure_encoder"]
        sample = encoder(voxel)
        model = pipeline.models["sparse_structure_flow_model"]
        params = pipeline.sparse_structure_sampler_params
    else:
        # Test SLAT stage
        from trellis.modules import sparse as sp
        slat = feats_to_slat(pipeline, args.source_features, sp.SparseTensor)
        from editing.inversion.rf_inversion import get_slat_norm_tensors
        mean, std = get_slat_norm_tensors(pipeline, slat.device, slat.feats.dtype)
        sample = (slat - mean) / std
        model = pipeline.models["slat_flow_model"]
        params = pipeline.slat_sampler_params

    return sample, model, params


def get_inverters(args) -> List[BaseInverter]:
    """Create inverter instances based on args."""
    inverters = []

    if "euler" in args.methods:
        inverters.append(EulerInverter(corrector_steps=args.corrector_steps))

    if "rf" in args.methods:
        inverters.append(RFSolverInverter(corrector_steps=args.corrector_steps))

    return inverters


def save_inversion_result(
    result: InversionResult,
    inverter: BaseInverter,
    out_dir: Path,
):
    """Save inversion result to disk."""
    method_dir = out_dir / inverter.name
    method_dir.mkdir(parents=True, exist_ok=True)

    # Save trajectories (just metadata, not full tensors to save space)
    inv_meta = {
        "timesteps": result.inverse_trajectory.timesteps,
        "num_latents": len(result.inverse_trajectory.latents),
        "metadata": result.inverse_trajectory.metadata,
    }
    fwd_meta = {
        "timesteps": result.forward_trajectory.timesteps,
        "num_latents": len(result.forward_trajectory.latents),
        "metadata": result.forward_trajectory.metadata,
    }

    with open(method_dir / "inverse_trajectory.json", "w") as f:
        json.dump(inv_meta, f, indent=2)

    with open(method_dir / "forward_trajectory.json", "w") as f:
        json.dump(fwd_meta, f, indent=2)

    # Compute and save metrics
    metrics = compute_trajectory_similarity(
        inverse_trajectory=result.inverse_trajectory.latents,
        forward_trajectory=result.forward_trajectory.latents,
        timesteps=result.inverse_trajectory.timesteps,
    )

    if result.similarity_metrics:
        metrics.update(result.similarity_metrics)

    with open(method_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save config
    config = inverter.get_config()
    with open(method_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{inverter.name} metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")


def generate_comparison_report(out_dir: Path, inverters: List[BaseInverter]):
    """Generate cross-method comparison report."""
    comparison = {}

    for inverter in inverters:
        method_dir = out_dir / inverter.name
        metrics_path = method_dir / "metrics.json"

        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            comparison[inverter.name] = metrics

    with open(out_dir / "comparison_report.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "=" * 60)
    print("COMPARISON REPORT")
    print("=" * 60)

    if not comparison:
        print("No results to compare")
        return

    # Print comparison table
    metric_names = list(next(iter(comparison.values())).keys())
    print(f"\n{'Method':<30} | " + " | ".join(f"{m:<15}" for m in metric_names))
    print("-" * (30 + 3 + (18 * len(metric_names))))

    for method_name, metrics in comparison.items():
        values = " | ".join(f"{metrics[m]:>15.6f}" for m in metric_names)
        print(f"{method_name:<30} | {values}")


def main():
    args = parse_args()

    # Setup
    print(f"Testing inversion quality for {args.stage} stage")
    print(f"Case: {args.case_name}")
    print(f"Methods: {args.methods}")
    print(f"Steps: {args.steps}")
    print(f"Corrector steps: {args.corrector_steps}")
    if "uniedit" in args.methods:
        print(f"UniEdit omega: {args.uniedit_omega}")

    pipeline = setup_pipeline(args.seed)
    sample, model, params = load_source_assets(args, pipeline)

    # Get condition
    from PIL import Image
    source_image = Image.open(args.source_image)
    cond_dict = pipeline.get_cond([source_image])

    # Output directory
    out_dir = Path("outputs") / "inversion_test" / args.case_name / args.stage
    out_dir.mkdir(parents=True, exist_ok=True)

    # Test each inverter
    inverters = get_inverters(args)

    for inverter in inverters:
        print(f"\n{'=' * 60}")
        print(f"Testing: {inverter.name}")
        print(f"{'=' * 60}")

        result = inverter.invert(
            model=model,
            sample=sample,
            cond_dict=cond_dict,
            steps=args.steps,
            rescale_t=params["rescale_t"],
            cfg_strength=params["cfg_strength"],
            cfg_interval=(0.5, 1.0),
            verbose=True,
        )

        save_inversion_result(result, inverter, out_dir)

        # Cleanup
        del result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Generate comparison report
    generate_comparison_report(out_dir, inverters)

    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
