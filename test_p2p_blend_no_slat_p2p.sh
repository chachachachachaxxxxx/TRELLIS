#!/bin/bash

# Image P2P + Latent Blend 方法测试脚本
# 禁用 SLAT 阶段的 P2P，只在 SS 阶段使用 P2P
# 两个阶段都启用 latent blending

set -e

# 配置
SOURCE_IMAGE="assets/edit_example/images/2d_render.png"
EDIT_IMAGE="assets/edit_example/images/2d_edit.png"
MASK_IMAGE="assets/edit_example/images/2d_mask.png"
SOURCE_MODEL="assets/edit_example/source_model"
MASK_GLB="assets/edit_example/mask.glb"
CASE_NAME="p2p_blend_no_slat_p2p"
SEED=1

# 方法配置
METHOD="image_p2p_latent_blend"

# P2P 配置 - 只在 SS 阶段启用
INJECT_STAGES="sparse_structure"  # 禁用 SLAT 阶段的 P2P
SS_T_START=1.0
SS_T_END=0.3
SS_STRENGTH=1.0

# Latent Blending 配置 - 两个阶段都启用
BLEND_SS_ENABLED=true
BLEND_SLAT_ENABLED=true
BLEND_STRENGTH=1.0

# Inversion 配置
INVERSION_MODE="simple"  # "simple" (Euler) 或 "rf_solver" (VoxHammer)

# Sampler 配置
SS_STEPS=25
SLAT_STEPS=25

echo "=== Image P2P + Latent Blend (No SLAT P2P) ==="
echo "Source: $SOURCE_IMAGE"
echo "Edit: $EDIT_IMAGE"
echo "Mask: $MASK_IMAGE"
echo "Source Model: $SOURCE_MODEL"
echo "Mask GLB: $MASK_GLB"
echo "Case: $CASE_NAME"
echo "Seed: $SEED"
echo ""
echo "P2P Stages: $INJECT_STAGES (SLAT P2P disabled)"
echo "Blending: SS=$BLEND_SS_ENABLED, SLAT=$BLEND_SLAT_ENABLED"
echo "Inversion: $INVERSION_MODE"
echo ""

python run_edit_experiment.py \
  --method "$METHOD" \
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
  --extra blend_ss_enabled="$BLEND_SS_ENABLED" \
  --extra blend_slat_enabled="$BLEND_SLAT_ENABLED" \
  --extra blend_strength="$BLEND_STRENGTH" \
  --extra inversion_mode="$INVERSION_MODE" \
  --extra skip_render=true \
  --extra skip_glb=false \
  --extra skip_ply=false

echo ""
echo "=== Test Complete ==="
echo "Output: outputs/$METHOD/$CASE_NAME/"
