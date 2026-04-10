# Inversion Quality Testing Framework

## Overview

This framework tests the quality of different inversion methods by comparing inverse (data → noise) and forward (noise → data) trajectories. The goal is to evaluate how well each method can reconstruct the original data after inversion.

## Architecture

### Core Components

1. **Base Inverter** (`editing/inversion/base_inverter.py`)
   - Abstract base class for all inverters
   - Defines `InversionResult` and `InversionTrajectory` data structures
   - Standardizes the inversion interface

2. **Trajectory Metrics** (`editing/inversion/trajectory_metrics.py`)
   - Computes similarity metrics between trajectories
   - Supports both dense and sparse tensors
   - Metrics: L2 distance, cosine similarity, coordinate overlap

3. **Inverter Implementations**
   - `EulerInverter`: First-order Euler method with optional corrector
   - `RFSolverInverter`: Second-order RF-Solver with optional corrector

### Directory Structure

```
outputs/inversion_test/<case_name>/<stage>/
├── euler_first_order_c0/
│   ├── inverse_trajectory.json
│   ├── forward_trajectory.json
│   ├── metrics.json
│   └── config.json
├── euler_first_order_c2/
│   └── ...
├── rf_solver_second_order_c0/
│   └── ...
├── rf_solver_second_order_c2/
│   └── ...
└── comparison_report.json
```

## Two Core Inversion Methods + Corrector-Predictor Strategy

### 1. First-Order Euler

**Algorithm:**
```
x_{t+h} = x_t + h·f(x_t, t)
```

**Characteristics:**
- Simplest method
- First-order accuracy
- Fast but less accurate
- Good baseline for comparison

**Usage:**
```bash
# Euler without corrector
python test_inversion_quality.py --methods euler --corrector-steps 0 ...

# Euler with corrector (2 iterations per step)
python test_inversion_quality.py --methods euler --corrector-steps 2 ...
```

### 2. Second-Order RF-Solver

**Algorithm:**
```
x_{t+h} = x_t + h·f(x_t, t) - ½·h²·∂_t f(x_t, t)
```

where `∂_t f` is approximated via midpoint finite difference.

**Characteristics:**
- Second-order accuracy
- Better trajectory approximation
- Current default in TRELLIS
- Optional corrector-predictor iterations

**Usage:**
```bash
# Without corrector
python test_inversion_quality.py --methods rf --corrector-steps 0 ...

# With corrector (2 iterations per step)
python test_inversion_quality.py --methods rf --corrector-steps 2 ...
```

### 3. Corrector-Predictor Strategy

The corrector-predictor strategy is **not a separate method** but an **orthogonal enhancement** that can be applied to both Euler and RF-Solver methods.

**Algorithm:**
```python
# Predictor: Standard update (Euler or RF-Solver)
x = predictor_step(x, t_curr, t_next)

# Corrector: Multiple refinement iterations
for _ in range(corrector_steps):
    pred = model(x, t_next)
    x = x + noise_scale * pred
```

**Characteristics:**
- Can be applied to any base method (Euler or RF-Solver)
- Adds refinement iterations at each timestep
- Improves trajectory accuracy at cost of computation
- Inspired by Langevin dynamics

**Usage:**
```bash
# Euler + Corrector
python test_inversion_quality.py --methods euler --corrector-steps 2 ...

# RF-Solver + Corrector
python test_inversion_quality.py --methods rf --corrector-steps 2 ...
```

**Note:** The corrector-predictor strategy is orthogonal to the base method choice. Both Euler and RF-Solver can use corrector steps.

## Testing Dimensions

The framework tests inversion quality across two orthogonal dimensions:

1. **Base Method**: Euler (1st order) vs RF-Solver (2nd order)
2. **Corrector Steps**: 0, 1, 2, 5, ... (number of refinement iterations)

**Example Test Matrix:**
```
Euler + C=0 (baseline)
Euler + C=1
Euler + C=2
RF-Solver + C=0 (current default)
RF-Solver + C=1
RF-Solver + C=2
```

## Testing Stages

### Stage 1: Sparse Structure (ss)

Tests inversion on the voxel structure stage:
- Input: Voxel coordinates → Encoded latent
- Model: `sparse_structure_flow_model`
- Output: Voxel structure

```bash
python test_inversion_quality.py --stage ss ...
```

### Stage 2: SLAT Features (slat)

Tests inversion on the SLAT feature stage (most important):
- Input: SLAT features (normalized)
- Model: `slat_flow_model`
- Output: SLAT features

```bash
python test_inversion_quality.py --stage slat ...
```

## Metrics

### L2 Distance Metrics

- `l2_mean`: Average L2 distance across all timesteps
- `l2_max`: Maximum L2 distance across all timesteps
- `l2_final`: L2 distance at final step (t=0, reconstructed data)

### Similarity Metrics

- `cosine_similarity_final`: Cosine similarity at final step
- `coord_overlap_mean`: Average coordinate overlap (sparse tensors only)
- `coord_overlap_final`: Coordinate overlap at final step (sparse tensors only)

### Interpretation

- **Lower L2 distances** = Better reconstruction
- **Higher cosine similarity** = Better alignment
- **Higher coordinate overlap** = Better structure preservation

## Quick Start

### Basic Test

```bash
./test_inversion_quick.sh
```

### Custom Test

```bash
python test_inversion_quality.py \
  --source-voxels assets/edit_example/source_assets/voxels.ply \
  --source-features assets/edit_example/source_assets/features.pt \
  --source-image assets/edit_example/images/2d_render.png \
  --case-name my_test \
  --stage slat \
  --methods euler rf uniedit \
  --corrector-steps 0 \
  --uniedit-omega 1.0 \
  --steps 12 \
  --seed 1
```

### Compare Different Configurations

```bash
# Test different corrector steps
for c in 0 1 2 5; do
  python test_inversion_quality.py \
    --case-name "corrector_$c" \
    --corrector-steps $c \
    --methods rf uniedit \
    ...
done

# Test different omega values
for omega in 0.5 1.0 1.5 2.0; do
  python test_inversion_quality.py \
    --case-name "omega_$omega" \
    --uniedit-omega $omega \
    --methods uniedit \
    ...
done
```

## Expected Results

### Typical Metric Ranges (SLAT stage, 12 steps)

| Method | l2_mean | l2_final | cosine_similarity_final |
|--------|---------|----------|-------------------------|
| Euler (1st order) | 0.05-0.15 | 0.03-0.10 | 0.95-0.98 |
| RF-Solver (2nd order) | 0.02-0.08 | 0.01-0.05 | 0.97-0.99 |
| UniEdit (ω=1.0) | 0.03-0.10 | 0.02-0.06 | 0.96-0.99 |

**Note:** These are approximate ranges. Actual values depend on:
- Number of steps
- CFG strength
- Model architecture
- Input complexity

### What to Look For

1. **RF-Solver should outperform Euler**: Lower L2 distances, higher similarity
2. **Corrector should improve metrics**: More iterations = better reconstruction (diminishing returns)
3. **UniEdit omega effect**: Higher ω may increase guidance strength but could reduce stability

## Integration with Editing Methods

The inversion testing framework is separate from the editing methods but shares core components:

- **Shared**: `RFSolverSampler`, `UniEditRFSolver`, normalization utilities
- **Separate**: Editing methods use inversion for initialization, then apply edits
- **Testing**: This framework focuses purely on inversion quality without editing

## Future Extensions

1. **Sparse Tensor Metrics**: Enhanced metrics for sparse structure stage
2. **Visualization**: Plot trajectory distances over time
3. **Ablation Studies**: Systematic parameter sweeps
4. **Multi-Stage Testing**: Test full pipeline (ss → slat)
5. **Perceptual Metrics**: LPIPS, FID for decoded outputs

## References

- VoxHammer paper: RF-Solver algorithm (Section 3.1)
- UniEdit: Vector/characteristic fusion mechanism
- Predictor-Corrector: Langevin dynamics for diffusion models
