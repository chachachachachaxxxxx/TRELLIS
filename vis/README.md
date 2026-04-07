# Visualization Scripts

This directory contains scripts for visualizing TRELLIS intermediate results and debugging.

## Scripts

- **`example_visualize_stages.py`** - Visualize intermediate stages of the TRELLIS pipeline
  - Sparse structure stage
  - SLAT stage
  - Mesh extraction
  - Useful for debugging and understanding the pipeline

- **`visualize_variant_intermediates.py`** - Visualize intermediate results during variant generation
  - Shows how variants are generated from base models
  - Displays intermediate denoising steps

## Usage

Run from the project root:

```bash
# Visualize pipeline stages
python vis/example_visualize_stages.py

# Visualize variant intermediates
python vis/visualize_variant_intermediates.py
```

## Output

Visualization outputs are typically saved alongside the main outputs in the `outputs/` directory, with additional debug images and intermediate results.

## Dependencies

These scripts may require additional visualization dependencies:
- matplotlib
- imageio
- PIL/Pillow

All should be included in the `hammer` conda environment.
