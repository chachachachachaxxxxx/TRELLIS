# TRELLIS Inference Examples

This directory contains example scripts for running TRELLIS inference pipelines.

## Scripts

### Basic Inference
- **`example.py`** - Image-to-3D basic inference
- **`example_text.py`** - Text-to-3D basic inference
- **`example_multi_image.py`** - Multi-image-to-3D inference
- **`example_annotated.py`** - Annotated inference example with detailed comments

### Advanced Inference
- **`example_variant.py`** - Generate variants from existing 3D models
- **`example_flux_krea_to_3d.py`** - Integration with Flux/Krea for image generation + 3D

### Batch Processing
- **`batch_generate_from_edit_images.py`** - Batch generate 3D models from 2d_edit.png images
- **`test_hard5_edit_direct.sh`** - Test script for hard5 dataset (parallel, 4 GPUs)
- **`test_hard5_edit_direct_sequential.sh`** - Test script for hard5 dataset (sequential)

## Usage

### Basic Examples

All scripts can be run directly from the project root:

```bash
# Image-to-3D
python trellis_inference/example.py

# Text-to-3D
python trellis_inference/example_text.py

# Multi-image-to-3D
python trellis_inference/example_multi_image.py

# Variants
python trellis_inference/example_variant.py
```

### Batch Processing

Generate 3D models from multiple 2d_edit.png images:

```bash
# Parallel execution (4 GPUs)
./trellis_inference/test_hard5_edit_direct.sh

# Sequential execution
./trellis_inference/test_hard5_edit_direct_sequential.sh

# Custom batch processing
python trellis_inference/batch_generate_from_edit_images.py \
  --asset-dir assets/your_cases \
  --output-dir outputs/your_results \
  --parallel \
  --gpus "0,1,2,3"
```

**Batch script features:**
- Automatically discovers all `images/2d_edit.png` files in asset directories
- Supports parallel execution across multiple GPUs
- Generates GLB models, preview videos, and point clouds
- Output structure: `outputs/<output_dir>/<case_name>/edit.glb`

## Environment Setup

Make sure to activate the `hammer` conda environment before running:

```bash
conda activate hammer
```

Set environment variables as needed:

```bash
export SPCONV_ALGO=native  # For one-time runs
export ATTN_BACKEND=flash_attn  # or xformers
```

## Output

By default, outputs are saved to `outputs/<method_name>/<case_name>/` following the project's output layout convention.

For batch processing, each case gets its own directory with:
- `edit.glb` - Final 3D model
- `gaussian.mp4` - Gaussian rendering preview
- `radiance_field.mp4` - Radiance field preview
- `mesh.mp4` - Mesh preview
- `gaussian.ply` - Point cloud file
