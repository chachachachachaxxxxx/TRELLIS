"""
Generate source assets (voxels.ply + features.npz) from source model.

This module provides functionality to generate complete source assets
required by RF inversion and fusion methods.

Two approaches are supported:
1. From existing 3D model: Render multi-view images and extract features (RECOMMENDED)
2. From source image: Use TRELLIS pipeline sampling (faster but less accurate)
"""
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Literal
from PIL import Image


def generate_source_assets_from_model(
    model_path: Path,
    output_dir: Path,
    num_views: int = 150,
    resolution: int = 512,
    feature_model: str = "dinov2_vitl14_reg",
    batch_size: int = 10,
    engine: str = "CYCLES",  # 默认使用 CYCLES
) -> Dict[str, Path]:
    """
    Generate source assets from an existing 3D model (RECOMMENDED).

    This approach:
    1. Renders multi-view images from the 3D model
    2. Extracts features using DINOv2
    3. Generates voxels.ply and features.npz

    This produces higher quality features than sampling from TRELLIS,
    as it preserves the original model's details.

    Args:
        model_path: Path to input 3D model (.glb, .ply, etc.)
        output_dir: Directory to save assets
        num_views: Number of views to render
        resolution: Image resolution for rendering
        feature_model: DINOv2 model variant
        batch_size: Batch size for feature extraction

    Returns:
        Dictionary with paths to generated files

    Note:
        Requires: temp/bpy_render.py and temp/extract_feature.py
    """
    try:
        from editing.rendering import render_3d_model
        from editing.preprocess.feature_extraction import extract_features
    except ImportError:
        raise ImportError(
            "Multi-view rendering requires editing.rendering and editing.preprocess.feature_extraction modules."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Render multi-view images
    render_result = render_3d_model(
        file_path=str(model_path),
        output_dir=str(output_dir),
        num_views=num_views,
        resolution=resolution,
        engine=engine,
        save_mesh=True,
    )

    # Step 2: Extract features
    extract_features(
        str(output_dir),
        model=feature_model,
        batch_size=batch_size,
    )

    voxels_path = output_dir / "voxels.ply"
    features_path = output_dir / "features.npz"
    transforms_path = output_dir / "transforms.json"

    if not voxels_path.exists():
        # If voxels.ply doesn't exist, use mesh.ply
        mesh_path = output_dir / "mesh.ply"
        if mesh_path.exists():
            voxels_path = mesh_path

    return {
        "voxels_path": voxels_path,
        "features_path": features_path,
        "transforms_path": transforms_path,
        "num_views": num_views,
        "method": "multi_view_rendering",
    }


def generate_source_assets_from_image(
    pipeline,
    source_image: Image.Image,
    output_dir: Path,
    seed: int = 42,
    sparse_structure_steps: int = 12,
    slat_steps: int = 12,
    sparse_structure_cfg: float = 7.5,
    slat_cfg: float = 3.0,
    save_final_outputs: bool = True,
) -> Dict[str, Path]:
    """
    Generate source assets from a source image (FASTER but less accurate).

    This function runs the TRELLIS pipeline to generate:
    - voxels.ply: Sparse structure coordinates
    - features.npz: SLAT features
    - sample.glb: Final 3D model (optional)
    - sample.ply: Final 3D model (optional)

    Note: This method is faster but produces lower quality features
    compared to multi-view rendering, as TRELLIS sampling introduces
    some loss of detail from the original model.

    Args:
        pipeline: TRELLIS image-to-3D pipeline
        source_image: Source image (PIL Image)
        output_dir: Directory to save assets
        seed: Random seed
        sparse_structure_steps: Number of steps for sparse structure sampling
        slat_steps: Number of steps for SLAT sampling
        sparse_structure_cfg: CFG strength for sparse structure
        slat_cfg: CFG strength for SLAT
        save_final_outputs: Whether to save GLB and PLY files

    Returns:
        Dictionary with paths to generated files
    """
    from trellis.utils import postprocessing_utils

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preprocess image
    image = pipeline.preprocess_image(source_image)

    # Encode condition
    cond = pipeline.get_cond([image])

    # Sample sparse structure
    torch.manual_seed(seed)
    coords = pipeline.sample_sparse_structure(
        cond,
        num_samples=1,
        sampler_params={
            "steps": sparse_structure_steps,
            "cfg_strength": sparse_structure_cfg,
        },
    )

    # Sample SLAT
    slat = pipeline.sample_slat(
        cond,
        coords,
        sampler_params={
            "steps": slat_steps,
            "cfg_strength": slat_cfg,
        },
    )

    # Save voxels.ply
    voxels_path = output_dir / "voxels.ply"
    coords_np = coords.cpu().numpy()
    with open(voxels_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(coords_np)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for coord in coords_np:
            f.write(f"{coord[0]} {coord[1]} {coord[2]}\n")

    # Save features.npz
    features_path = output_dir / "features.npz"
    feats = slat.feats.detach().cpu().numpy()
    coords_slat = slat.coords.detach().cpu().numpy()
    np.savez(
        features_path,
        feats=feats,
        coords=coords_slat,
    )

    result = {
        "voxels_path": voxels_path,
        "features_path": features_path,
        "num_voxels": len(coords_np),
        "slat_shape": feats.shape,
        "method": "trellis_sampling",
    }

    # Optionally save final outputs
    if save_final_outputs:
        outputs = pipeline.decode_slat(slat, formats=["mesh", "gaussian"])

        # Save GLB
        glb_path = output_dir / "sample.glb"
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=0.95,
            texture_size=1024,
        )
        glb.export(str(glb_path))
        result["glb_path"] = glb_path

        # Save PLY
        ply_path = output_dir / "sample.ply"
        outputs['gaussian'][0].save_ply(str(ply_path))
        result["ply_path"] = ply_path

    return result


def ensure_source_assets(
    pipeline,
    asset_dir: Path,
    source_model: Optional[Path] = None,
    source_image: Optional[Image.Image] = None,
    method: Literal["auto", "multi_view", "trellis_sampling"] = "auto",
    **generation_kwargs,
) -> Dict[str, Path]:
    """
    Ensure source assets exist, generating them if necessary.

    Args:
        pipeline: TRELLIS image-to-3D pipeline
        asset_dir: Directory containing or to contain assets
        source_model: Path to source 3D model (for multi-view rendering)
        source_image: Source image (for TRELLIS sampling)
        method: Generation method:
            - "auto": Use multi_view if source_model provided, else trellis_sampling
            - "multi_view": Render multi-view and extract features (RECOMMENDED)
            - "trellis_sampling": Use TRELLIS pipeline sampling (faster but less accurate)
        **generation_kwargs: Additional arguments for generation functions

    Returns:
        Dictionary with paths to asset files

    Raises:
        FileNotFoundError: If assets don't exist and neither source_model nor source_image is provided
        ValueError: If method is invalid
    """
    asset_dir = Path(asset_dir)
    voxels_path = asset_dir / "voxels.ply"
    features_path = asset_dir / "features.npz"

    # Check if assets already exist
    if voxels_path.exists() and features_path.exists():
        return {
            "voxels_path": voxels_path,
            "features_path": features_path,
            "generated": False,
            "method": "existing",
        }

    # Determine generation method
    if method == "auto":
        if source_model is not None:
            method = "multi_view"
        elif source_image is not None:
            method = "trellis_sampling"
        else:
            raise FileNotFoundError(
                f"Source assets not found in {asset_dir} and neither source_model nor source_image provided"
            )

    # Generate assets
    if method == "multi_view":
        if source_model is None:
            raise ValueError("source_model is required for multi_view method")
        result = generate_source_assets_from_model(
            model_path=source_model,
            output_dir=asset_dir,
            **generation_kwargs,
        )
    elif method == "trellis_sampling":
        if source_image is None:
            raise ValueError("source_image is required for trellis_sampling method")
        result = generate_source_assets_from_image(
            pipeline=pipeline,
            source_image=source_image,
            output_dir=asset_dir,
            **generation_kwargs,
        )
    else:
        raise ValueError(f"Invalid method: {method}. Must be 'auto', 'multi_view', or 'trellis_sampling'")

    result["generated"] = True
    return result
