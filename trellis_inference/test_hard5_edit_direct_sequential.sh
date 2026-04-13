#!/bin/bash
# Test script: Generate 3D models from hard5 2d_edit images sequentially (one at a time)

cd "$(dirname "$0")/.." || exit 1

export ATTN_BACKEND='flash-attn'
export SPCONV_ALGO='native'

python trellis_inference/batch_generate_from_edit_images.py \
  --asset-dir assets/hard5 \
  --output-dir outputs/hard5_edit_direct \
  --seed 1

echo ""
echo "Results saved to: outputs/hard5_edit_direct/"
