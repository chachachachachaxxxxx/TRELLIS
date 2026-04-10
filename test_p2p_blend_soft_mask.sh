#!/bin/bash

# Image P2P + Latent Blend 方法测试脚本
# SS 和 SLAT 阶段都使用软掩码（soft mask）进行 latent blending
# 测试软掩码对边界过渡的影响

set -e

# 配置
SOURCE_IMAGE="assets/edit_example/images/2d_render.png"
EDIT_IMAGE="assets/edit_example/images/2d_edit.png"
MASK_IMAGE="assets/edit_example/images/2d_mask.png"
SOURCE_MODEL="outputs/source_assets_test0_multiview"
MASK_GLB="assets/edit_example/mask.glb"
CASE_NAME="p2p_blend_soft_mask"
SEED=1
GPU=2
export CUDA_VISIBLE_DEVICES=$GPU
# 方法配置
METHOD="image_p2p_latent_blend"

# P2P 配置
INJECT_STAGES="sparse_structure,slat"
SS_T_START=1.0
SS_T_END=0.3
SS_STRENGTH=1.0
SLAT_T_START=1.0
SLAT_T_END=0.8
SLAT_STRENGTH=1.0

# Latent Blending 配置 - 使用软掩码
BLEND_SS_ENABLED=true
BLEND_SLAT_ENABLED=true
BLEND_STRENGTH=1.0
SS_BLEND_MODE="soft"        # SS 阶段使用软掩码
SLAT_BLEND_MODE="soft"      # SLAT 阶段使用软掩码
SS_SOFT_KERNEL_SIZE=5       # SS 软掩码高斯核大小
SLAT_SOFT_KERNEL_SIZE=7     # SLAT 软掩码高斯核大小（更大的核 = 更平滑的过渡）

# Inversion 配置
INVERSION_MODE="simple"  # "simple" (Euler) 或 "rf_solver" (VoxHammer)

# Sampler 配置
SS_STEPS=25
SLAT_STEPS=25

echo "=== Image P2P + Latent Blend (Soft Mask) ==="
echo "Source: $SOURCE_IMAGE"
echo "Edit: $EDIT_IMAGE"
echo "Mask: $MASK_IMAGE"
echo "Source Model: $SOURCE_MODEL"
echo "Mask GLB: $MASK_GLB"
echo "Case: $CASE_NAME"
echo "Seed: $SEED"
echo ""
echo "P2P Stages: $INJECT_STAGES"
echo "Blending: SS=$BLEND_SS_ENABLED (mode=$SS_BLEND_MODE, kernel=$SS_SOFT_KERNEL_SIZE), SLAT=$BLEND_SLAT_ENABLED (mode=$SLAT_BLEND_MODE, kernel=$SLAT_SOFT_KERNEL_SIZE)"
echo "Inversion: $INVERSION_MODE"
echo ""

python run_edit_experiment.py \
  --method "$METHOD" \
  --asset-dir "$SOURCE_MODEL" \
  --source-image "$SOURCE_IMAGE" \
  --edit-image "$EDIT_IMAGE" \
  --mask-image "$MASK_IMAGE" \
  --mask-glb "$MASK_GLB" \
  --case-name "$CASE_NAME" \
  --seed "$SEED" \
  --preprocess \
  --sparse-structure-steps "$SS_STEPS" \
  --slat-steps "$SLAT_STEPS" \
  --extra inject_stages="$INJECT_STAGES" \
  --extra sparse_structure_t_start="$SS_T_START" \
  --extra sparse_structure_t_end="$SS_T_END" \
  --extra sparse_structure_strength="$SS_STRENGTH" \
  --extra slat_t_start="$SLAT_T_START" \
  --extra slat_t_end="$SLAT_T_END" \
  --extra slat_strength="$SLAT_STRENGTH" \
  --extra blend_ss_enabled="$BLEND_SS_ENABLED" \
  --extra blend_slat_enabled="$BLEND_SLAT_ENABLED" \
  --extra blend_strength="$BLEND_STRENGTH" \
  --extra ss_blend_mode="$SS_BLEND_MODE" \
  --extra slat_blend_mode="$SLAT_BLEND_MODE" \
  --extra ss_soft_kernel_size="$SS_SOFT_KERNEL_SIZE" \
  --extra slat_soft_kernel_size="$SLAT_SOFT_KERNEL_SIZE" \
  --extra inversion_mode="$INVERSION_MODE" \
  --extra skip_render=true \
  --extra skip_glb=false \
  --extra skip_ply=false

echo ""
echo "=== Test Complete ==="
echo "Output: outputs/$METHOD/$CASE_NAME/"
echo ""
echo "对比测试："
echo "  - 硬掩码（默认）: outputs/$METHOD/p2p_blend_no_slat_p2p/"
echo "  - 软掩码（本次）: outputs/$METHOD/$CASE_NAME/"
