#!/bin/bash

# 简化测试：只测试一个配置
# Soft mask (kernel=5) with full blend strength

set -e

CASE_NAME="p2p_blend_quick_test"
SEED=1
GPU=3

export CUDA_VISIBLE_DEVICES=$GPU
export SPCONV_ALGO=native

echo "=========================================="
echo "Quick test: image_p2p_latent_blend"
echo "Configuration: soft mask (kernel=5)"
echo "=========================================="

python run_edit_experiment.py \
  --method image_p2p_latent_blend \
  --attn-backend xformers \
  --source-image assets/edit_example/images/2d_render.png \
  --edit-image assets/edit_example/images/2d_edit.png \
  --mask-image assets/edit_example/images/2d_mask.png \
  --case-name "$CASE_NAME" \
  --seed $SEED \
  --preprocess \
  --ss-steps 12 \
  --slat-steps 12 \
  --extra-param blend_slat_enabled=true \
  --extra-param slat_blend_mode=soft \
  --extra-param slat_soft_kernel_size=5 \
  --extra-param blend_strength=1.0

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Test passed!"
    echo "Output: outputs/image_p2p_latent_blend/$CASE_NAME"
else
    echo ""
    echo "❌ Test failed"
    exit 1
fi
