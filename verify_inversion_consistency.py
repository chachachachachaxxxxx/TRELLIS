#!/usr/bin/env python3
"""Verify that inversion produces consistent results."""

import os
import sys
from pathlib import Path

os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPARSE_ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'

sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from PIL import Image
from trellis.pipelines import TRELLISImageTo3DPipeline

# Load pipeline
pipeline = TRELLISImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.to("cuda")

# Load image
image = Image.open("assets/edit_example/images/2d_render.png")

# Run inversion twice with same seed
seed = 1
results = []

for run in range(2):
    print(f"\n{'='*60}")
    print(f"Run {run + 1}: Inversion with seed={seed}")
    print(f"{'='*60}")

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Run generation (which includes inversion internally)
    outputs = pipeline.run(
        image,
        seed=seed,
        formats=["gaussian", "mesh"],
        preprocess_image=True,
        sparse_structure_sampler_params={
            "steps": 12,
            "cfg_strength": 7.5,
        },
        slat_sampler_params={
            "steps": 12,
            "cfg_strength": 3.0,
        },
    )

    # Extract SS coords
    ss_coords = outputs['gaussian'][0].structure.coords
    print(f"SS coords shape: {ss_coords.shape}")
    print(f"SS coords hash: {hash(ss_coords.cpu().numpy().tobytes())}")

    results.append({
        'coords': ss_coords.cpu().numpy(),
        'hash': hash(ss_coords.cpu().numpy().tobytes()),
    })

# Compare results
print(f"\n{'='*60}")
print("Comparison")
print(f"{'='*60}")
print(f"Run 1 coords: {results[0]['coords'].shape}")
print(f"Run 2 coords: {results[1]['coords'].shape}")
print(f"Coords identical: {np.array_equal(results[0]['coords'], results[1]['coords'])}")
print(f"Hash match: {results[0]['hash'] == results[1]['hash']}")

if not np.array_equal(results[0]['coords'], results[1]['coords']):
    print("\n⚠️  WARNING: Inversion is NOT deterministic!")
    print("This could be due to:")
    print("  1. Non-deterministic CUDA operations")
    print("  2. Floating point precision issues")
    print("  3. Uninitialized random state in pipeline")
else:
    print("\n✅ Inversion is deterministic with same seed")
