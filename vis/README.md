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

- **`visualize_boundary_alignment.py`** - Boundary-focused voxel comparison for `source / edit / mask`
  - Only keeps the visualization centered on the mask boundary and mask-internal overlap
  - Renders voxels as nested small cubes instead of point clouds, so `mask/source/edit` boundary layers can overlap visibly
  - Reports `boundary_aligned`, `boundary_missing`, `boundary_added`, and `edit_leak_near_boundary` in the stats summary
  - Writes a standalone HTML report plus a JSON stats summary

## Usage

Run from the project root:

```bash
# Visualize pipeline stages
python vis/example_visualize_stages.py

# Visualize variant intermediates
python vis/visualize_variant_intermediates.py

# Visualize boundary alignment for source/edit/mask voxels
python vis/visualize_boundary_alignment.py \
  --mask assets/edit_example/voxels_delete.ply \
  --source assets/edit_example/voxels.ply \
  --edit outputs/p2p_latent_blend_ss/test/edit/ss/coords.ply \
  --output-dir outputs/boundary_alignment_vis
```

## Output

Visualization outputs are typically saved alongside the main outputs in the `outputs/` directory, with additional debug images and intermediate results.

For `visualize_boundary_alignment.py`, the output directory contains:
- `boundary_alignment.html` - standalone interactive report
- `boundary_alignment_stats.json` - overlap and boundary metrics

## Boundary Report Layers

The boundary report exposes these voxel layers:
- `Mask Boundary` - mask voxels restricted to the boundary only
- `Boundary Aligned` - source and edit both occupy the same boundary voxels
- `Boundary Missing` - source occupied the boundary, edit did not
- `Boundary Added` - edit occupies boundary voxels not present in source
- `In-Mask Overlap` - source/edit overlap inside the mask
- `In-Mask Source Only` - source-only voxels inside the mask
- `In-Mask Edit Only` - edit-only voxels inside the mask
- `Edit Leak Near Boundary` - edit voxels outside the mask but adjacent to the mask boundary

## Dependencies

These scripts may require additional visualization dependencies:
- matplotlib
- imageio
- PIL/Pillow

All should be included in the `hammer` conda environment.
