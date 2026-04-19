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
- **`batch_generate_from_multiview_targets.py`** - Batch generate TRELLIS multi-view baselines from benchmark `input_views/*/target.png`
- **`test_hard5_edit_direct.sh`** - Test script for hard5 dataset (parallel, 4 GPUs)
- **`test_hard5_edit_direct_sequential.sh`** - Test script for hard5 dataset (sequential)
- **`run_edit3d_mv_trellis_target_stochastic.sh`** - Benchmark multi-view baseline with TRELLIS `stochastic` mode
- **`run_edit3d_mv_trellis_target_view_aligned.sh`** - Benchmark multi-view baseline with latent-rotation-aligned stochastic denoising
- **`run_edit3d_mv_trellis_target_stochastic_order_ablation.sh`** - Benchmark `stochastic` mode with different multi-view read orders
- **`run_edit3d_mv_trellis_target_multidiffusion.sh`** - Benchmark multi-view baseline with TRELLIS `multidiffusion` mode

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

# Edit3D-Bench multi-view target baseline
python trellis_inference/batch_generate_from_multiview_targets.py \
  --dataset-root /cache/wangxinxing/data/trellis_edit_benchmark/edit3d_mv_pseudosource_micro10 \
  --output-root /cache/wangxinxing/data/trellis_edit_benchmark/pred_mv \
  --run-name trellis_mv_target_stochastic_seed1 \
  --mode stochastic \
  --view-order azimuth \
  --parallel \
  --gpus "0,1,2,3"

# View-aligned latent rotation baseline
python trellis_inference/batch_generate_from_multiview_targets.py \
  --dataset-root /cache/wangxinxing/data/trellis_edit_benchmark/edit3d_mv_pseudosource_micro10 \
  --output-root /cache/wangxinxing/data/trellis_edit_benchmark/pred_mv \
  --run-name trellis_mv_target_view_aligned_stochastic_seed1 \
  --mode view_aligned_stochastic \
  --view-order azimuth \
  --parallel \
  --gpus "0,1,2,3"

# Stochastic read-order ablation
./trellis_inference/run_edit3d_mv_trellis_target_stochastic_order_ablation.sh
```

**Batch script features:**
- Automatically discovers all `images/2d_edit.png` files in asset directories
- Supports parallel execution across multiple GPUs
- Generates GLB models, preview videos, and point clouds
- Output structure: `outputs/<output_dir>/<case_name>/edit.glb`

**Multi-view benchmark script features:**
- Reads conditioning images from `case.json -> input_views[*].target_path`
- Supports TRELLIS multi-image modes: `stochastic`, `view_aligned_stochastic`, and `multidiffusion`
- `view_aligned_stochastic` rotates the current `z_t` around latent-space `y` to match the active view azimuth before each denoising call, then rotates the predicted update back to canonical space
- Supports multi-view read-order ablations for `stochastic` runs
- Writes benchmark-friendly output structure: `pred_mv/<run_name>/cases/<case_id>/edit.glb`
- Writes `pred_mv/<run_name>/manifest.json` with case metadata and run status

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
