# P2P Latent Blend Implementation Summary

## Overview

Completed implementation of `image_p2p_latent_blend` method - a VoxHammer-style editing approach that combines Prompt-to-Prompt attention injection with per-step latent blending.

## Changes Made (2026-04-09)

### 1. Consolidated V1 and V2
- **Deleted**: Old V1 implementation (simple latent blending without inversion)
- **Renamed**: V2 → V1 (VoxHammer-style implementation with inversion)
- **Reason**: V2 is the correct implementation matching VoxHammer paper

### 2. Method Implementation (`editing/methods/image_p2p_latent_blend.py`)

**Core Features**:
- P2P attention injection during denoising
- Per-step latent blending at both SS and SLAT stages
- Inversion-based approach (required for latent cache)

**Inversion Modes** (ablation study):
- `"simple"`: First-order Euler inversion (fast, default)
- `"rf_solver"`: Second-order RF-Solver inversion (accurate, VoxHammer)

**Blending Strategy**:
- **SS Stage**: Dense latent blending [B, C, D, H, W] based on 3D mask
- **SLAT Stage**: Sparse feature blending based on coordinate matching

**Key Parameters**:
```python
{
    # P2P injection
    "inject_stages": ["sparse_structure", "slat"],
    "sparse_structure_t_start": 1.0,
    "sparse_structure_t_end": 0.3,
    "slat_t_start": 0.8,
    "slat_t_end": 0.0,
    
    # Latent blending
    "blend_ss_enabled": True,
    "blend_slat_enabled": True,
    "ss_blend_mode": "hard",  # or "soft"
    "slat_blend_mode": "hard",  # or "soft"
    "blend_strength": 1.0,
    
    # Inversion
    "inversion_mode": "simple",  # or "rf_solver"
}
```

### 3. Custom Sampler (`editing/samplers/latent_blend_sampler.py`)

Implemented three sampler variants:
- `LatentBlendFlowEulerSampler`: Base sampler with latent blending
- `LatentBlendFlowEulerCfgSampler`: With CFG support
- `LatentBlendFlowEulerGuidanceIntervalSampler`: With late-time CFG (used by method)

**Blending Logic**:
- Dense (SS): `z_t ← M ⊙ z_t + (1-M) ⊙ ẑ_t` (VoxHammer Eq. 4)
- Sparse (SLAT): `∀u ∈ Ωkeep: z_t[u] ← ẑ_t[u]` (VoxHammer Eq. 5)

### 4. Registry Update (`editing/methods/registry.py`)

- Removed `image_p2p_latent_blend_v2` entry
- Updated `image_p2p_latent_blend` description
- Registered `ImageP2PLatentBlendMethod` class

## Implementation Details

### Inversion Process

**SS Stage Inversion**:
1. Generate source SS coords from source image
2. Encode to latent space: `z_s = encoder(voxel)`
3. Invert from data (t=0) to noise (t=1)
4. Cache latents at each timestep: `{f"{t}": latent}`

**SLAT Stage Inversion**:
1. Load source SLAT features from preprocessed assets
2. Normalize: `z = (slat - mean) / std`
3. Invert from data to noise
4. Cache SLAT at each timestep: `{f"{t}": SparseTensor}`

### Blending During Denoising

**At each denoising step**:
1. Get cached source latent at current timestep `t`
2. Blend current latent with source based on mask
3. Proceed with normal Euler sampling

**SS Blending**:
- Build 3D latent mask from 2D spatial mask
- Blend dense tensors element-wise

**SLAT Blending**:
- Find preserve region coords from mask
- Replace features at matching coordinates

### Mask Processing

**Hard Mask**: Binary (0 or 1)
**Soft Mask**: Gaussian blur for smooth boundaries
```python
kernel_size = 3 (SS) or 5 (SLAT)
sigma = kernel_size / 3.0
```

## Usage

```bash
python run_edit_experiment.py \
  --method image_p2p_latent_blend \
  --source-image path/to/source.png \
  --edit-image path/to/edit.png \
  --mask-image path/to/mask.png \
  --source-model path/to/source/assets \
  --case-name my_edit \
  --seed 1 \
  --extra-params inversion_mode=simple blend_ss_enabled=True blend_slat_enabled=True
```

## Requirements

- Source SLAT features (from preprocessing): `features.npz`
- Source/edit/mask images (aligned)
- GPU memory: ~16GB (inversion + denoising)

## Testing Status

- ✅ Code compiles
- ✅ Import successful
- ⏳ End-to-end testing pending
- ⏳ Ablation study (simple vs rf_solver) pending

## Next Steps

1. Test with real data
2. Compare inversion modes (simple vs rf_solver)
3. Tune blending parameters (t_start, t_end, strength)
4. Benchmark memory usage and speed
5. Compare with other methods (UniEdit, pure P2P)

## References

- VoxHammer paper: Latent replacement strategy (Eq. 4-5)
- Prompt-to-Prompt: Attention injection
- TRELLIS: Flow matching framework
