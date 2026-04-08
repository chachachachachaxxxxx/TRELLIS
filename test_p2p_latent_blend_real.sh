#!/bin/bash

# 测试 image_p2p_latent_blend 方法
# 在真实样例上测试 mask-based SLAT 特征混合

set -e

CASE_NAME="p2p_blend_real_test"
SEED=1
GPU=3

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$GPU
export SPCONV_ALGO=native

echo "=========================================="
echo "Testing image_p2p_latent_blend method"
echo "=========================================="

# 使用 assets/edit_example 中的真实数据
SOURCE_IMAGE="assets/edit_example/images/2d_render.png"
EDIT_IMAGE="assets/edit_example/images/2d_edit.png"
MASK_IMAGE="assets/edit_example/images/2d_mask.png"

# 检查文件是否存在
if [ ! -f "$SOURCE_IMAGE" ]; then
    echo "❌ Source image not found: $SOURCE_IMAGE"
    exit 1
fi

if [ ! -f "$EDIT_IMAGE" ]; then
    echo "❌ Edit image not found: $EDIT_IMAGE"
    exit 1
fi

if [ ! -f "$MASK_IMAGE" ]; then
    echo "❌ Mask image not found: $MASK_IMAGE"
    exit 1
fi

echo ""
echo "Test 1: Soft mask (kernel=5) - Recommended"
echo "=========================================="
python run_edit_experiment.py \
  --method image_p2p_latent_blend \
  --source-image "$SOURCE_IMAGE" \
  --edit-image "$EDIT_IMAGE" \
  --mask-image "$MASK_IMAGE" \
  --case-name "${CASE_NAME}_soft5" \
  --seed $SEED \
  --preprocess \
  --ss-steps 12 \
  --slat-steps 12 \
  --extra-param blend_slat_enabled=true \
  --extra-param slat_blend_mode=soft \
  --extra-param slat_soft_kernel_size=5 \
  --extra-param blend_strength=1.0

if [ $? -eq 0 ]; then
    echo "✅ Test 1 (soft mask kernel=5) passed"
else
    echo "❌ Test 1 failed"
    exit 1
fi

echo ""
echo "Test 2: Hard mask - Ablation"
echo "=========================================="
python run_edit_experiment.py \
  --method image_p2p_latent_blend \
  --source-image "$SOURCE_IMAGE" \
  --edit-image "$EDIT_IMAGE" \
  --mask-image "$MASK_IMAGE" \
  --case-name "${CASE_NAME}_hard" \
  --seed $SEED \
  --preprocess \
  --ss-steps 12 \
  --slat-steps 12 \
  --extra-param blend_slat_enabled=true \
  --extra-param slat_blend_mode=hard \
  --extra-param blend_strength=1.0

if [ $? -eq 0 ]; then
    echo "✅ Test 2 (hard mask) passed"
else
    echo "❌ Test 2 failed"
    exit 1
fi

echo ""
echo "Test 3: Soft mask (kernel=7) - More smoothing"
echo "=========================================="
python run_edit_experiment.py \
  --method image_p2p_latent_blend \
  --source-image "$SOURCE_IMAGE" \
  --edit-image "$EDIT_IMAGE" \
  --mask-image "$MASK_IMAGE" \
  --case-name "${CASE_NAME}_soft7" \
  --seed $SEED \
  --preprocess \
  --ss-steps 12 \
  --slat-steps 12 \
  --extra-param blend_slat_enabled=true \
  --extra-param slat_blend_mode=soft \
  --extra-param slat_soft_kernel_size=7 \
  --extra-param blend_strength=1.0

if [ $? -eq 0 ]; then
    echo "✅ Test 3 (soft mask kernel=7) passed"
else
    echo "❌ Test 3 failed"
    exit 1
fi

echo ""
echo "Test 4: Reduced blend strength - Ablation"
echo "=========================================="
python run_edit_experiment.py \
  --method image_p2p_latent_blend \
  --source-image "$SOURCE_IMAGE" \
  --edit-image "$EDIT_IMAGE" \
  --mask-image "$MASK_IMAGE" \
  --case-name "${CASE_NAME}_strength05" \
  --seed $SEED \
  --preprocess \
  --ss-steps 12 \
  --slat-steps 12 \
  --extra-param blend_slat_enabled=true \
  --extra-param slat_blend_mode=soft \
  --extra-param slat_soft_kernel_size=5 \
  --extra-param blend_strength=0.5

if [ $? -eq 0 ]; then
    echo "✅ Test 4 (blend_strength=0.5) passed"
else
    echo "❌ Test 4 failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "All tests completed successfully!"
echo "=========================================="
echo ""
echo "Output directories:"
echo "  - outputs/image_p2p_latent_blend/${CASE_NAME}_soft5"
echo "  - outputs/image_p2p_latent_blend/${CASE_NAME}_hard"
echo "  - outputs/image_p2p_latent_blend/${CASE_NAME}_soft7"
echo "  - outputs/image_p2p_latent_blend/${CASE_NAME}_strength05"
echo ""
echo "Compare results to see:"
echo "  1. Soft vs Hard mask effect"
echo "  2. Different kernel sizes (5 vs 7)"
echo "  3. Blend strength impact (1.0 vs 0.5)"
