"""Debug memory usage for OOM methods."""
import os
os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPARSE_ATTN_BACKEND"] = "xformers"
os.environ["SPCONV_ALGO"] = "native"

import torch
import gc
from pathlib import Path

def print_memory_stats(stage_name: str):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\n[{stage_name}]")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
        print(f"  Max Allocated: {max_allocated:.2f} GB")

def clear_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def test_text_prompt_to_prompt():
    """Test text_prompt_to_prompt memory usage."""
    print("\n" + "="*60)
    print("Testing text_prompt_to_prompt")
    print("="*60)

    clear_memory()
    print_memory_stats("Initial")

    # Import pipeline
    from trellis.pipelines import TrellisTextTo3DPipeline
    print_memory_stats("After import")

    # Load pipeline
    pipeline = TrellisTextTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-text-large")
    pipeline.to("cuda")
    print_memory_stats("After loading pipeline")

    # Prepare inputs
    source_prompt = "a cute cat statue"
    edit_prompt = "a cute tiger statue"

    # Encode conditions
    source_cond_dict = pipeline.get_cond([source_prompt])
    edit_cond_dict = pipeline.get_cond([edit_prompt])
    print_memory_stats("After encoding conditions")

    # Sample sparse structure
    torch.manual_seed(1)
    coords = pipeline.sample_sparse_structure(
        edit_cond_dict,
        num_samples=1,
        sampler_params={"steps": 12},  # Reduced steps
    )
    print_memory_stats("After sampling sparse structure")

    # Sample SLAT
    slat = pipeline.sample_slat(
        edit_cond_dict,
        coords,
        sampler_params={"steps": 12},  # Reduced steps
    )
    print_memory_stats("After sampling SLAT")

    # Decode - this is where OOM happens
    try:
        outputs = pipeline.decode_slat(slat, ["mesh"])  # Only mesh, no gaussian/radiance_field
        print_memory_stats("After decode (mesh only)")
    except RuntimeError as e:
        print(f"\n!!! OOM at decode: {e}")
        print_memory_stats("At OOM")
        return

    print("\n✓ text_prompt_to_prompt completed successfully")

def test_rf_inversion():
    """Test image_prompt_to_prompt_rf_inversion memory usage."""
    print("\n" + "="*60)
    print("Testing image_prompt_to_prompt_rf_inversion")
    print("="*60)

    clear_memory()
    print_memory_stats("Initial")

    # Import pipeline
    from trellis.pipelines import TrellisImageTo3DPipeline
    print_memory_stats("After import")

    # Load pipeline
    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipeline.to("cuda")
    print_memory_stats("After loading pipeline")

    # Check if source assets exist
    source_voxels = Path("assets/edit_example/source_assets/voxels.ply")
    source_features = Path("assets/edit_example/source_assets/slat.npz")

    if not source_voxels.exists() or not source_features.exists():
        print(f"\n!!! Source assets not found:")
        print(f"  {source_voxels}: {source_voxels.exists()}")
        print(f"  {source_features}: {source_features.exists()}")
        return

    # Load source assets
    from editing.preprocess.asset_3d import ply_to_coords, feats_to_slat
    from trellis.modules import sparse as sp

    resolution = 64
    source_coords = ply_to_coords(source_voxels, pipeline.device, resolution)
    print_memory_stats("After loading source coords")

    source_slat = feats_to_slat(pipeline, source_features, sp.SparseTensor)
    print_memory_stats("After loading source SLAT")

    # Load and encode images
    from PIL import Image
    source_image = Image.open("assets/edit_example/images/2d_render.png")
    edit_image = Image.open("assets/edit_example/images/2d_edit.png")

    source_cond_dict = pipeline.get_cond([source_image])
    edit_cond_dict = pipeline.get_cond([edit_image])
    print_memory_stats("After encoding conditions")

    # Invert sparse structure - this is where OOM might happen
    from editing.inversion import invert_sparse_structure
    from editing.preprocess.asset_3d import coords_to_voxel

    source_voxel = coords_to_voxel(source_coords, pipeline.device, resolution)
    print_memory_stats("After coords_to_voxel")

    try:
        torch.manual_seed(1)
        ss_terminal_noise = invert_sparse_structure(
            pipeline=pipeline,
            cond_src=source_cond_dict,
            voxel_src=source_voxel,
            params={"steps": 12},  # Reduced steps
            cfg_interval=(0.0, 1.0),
            verbose=True,
        )
        print_memory_stats("After inverting sparse structure")
    except RuntimeError as e:
        print(f"\n!!! OOM at sparse structure inversion: {e}")
        print_memory_stats("At OOM")
        return

    # Invert SLAT
    from editing.inversion import invert_slat

    try:
        slat_terminal_noise = invert_slat(
            pipeline=pipeline,
            cond_src=source_cond_dict,
            slat_src=source_slat,
            params={"steps": 12},  # Reduced steps
            cfg_interval=(0.0, 1.0),
            verbose=True,
        )
        print_memory_stats("After inverting SLAT")
    except RuntimeError as e:
        print(f"\n!!! OOM at SLAT inversion: {e}")
        print_memory_stats("At OOM")
        return

    print("\n✓ RF inversion completed successfully")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python debug_memory_usage.py [text|rf|both]")
        sys.exit(1)

    mode = sys.argv[1]

    if mode in ["text", "both"]:
        test_text_prompt_to_prompt()
        clear_memory()

    if mode in ["rf", "both"]:
        test_rf_inversion()
