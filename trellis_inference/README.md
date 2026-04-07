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

## Usage

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
