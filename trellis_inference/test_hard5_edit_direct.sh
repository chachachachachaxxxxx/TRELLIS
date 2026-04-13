#!/bin/bash
# Test script: Generate 3D models from hard5 2d_edit images using TRELLIS directly
# Runs one case per GPU in parallel

cd "$(dirname "$0")/.." || exit 1

export ATTN_BACKEND='flash-attn'
export SPCONV_ALGO='native'

python trellis_inference/batch_generate_from_edit_images.py \
  --asset-dir assets/hard5 \
  --output-dir outputs/hard5_edit_direct \
  --seed 1 \
  --parallel \
  --gpus "0,1,2,3"

echo ""
echo "Results saved to: outputs/hard5_edit_direct/"
echo "Each case directory contains:"
echo "  - edit.glb (final 3D model)"
echo "  - gaussian.mp4, radiance_field.mp4, mesh.mp4 (preview videos)"
echo "  - gaussian.ply (point cloud)"
