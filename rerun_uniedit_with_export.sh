#!/bin/bash
# 重新运行成功的 UniEdit 测试以生成 GLB/PLY

set -e

CASE_NAME="test0"
SEED=1
SOURCE_IMAGE="assets/edit_example/images/2d_render.png"
EDIT_IMAGE="assets/edit_example/images/2d_edit.png"
MASK_IMAGE="assets/edit_example/images/2d_mask.png"
SOURCE_MODEL="outputs/source_assets_test0_multiview"
MASK_GLB="assets/edit_example/mask.glb"

echo "=========================================="
echo "重新运行 UniEdit 测试（生成 GLB/PLY）"
echo "=========================================="

# Test 1: preserve_uniedit
echo "Test 1/2: preserve_uniedit"
python run_edit_experiment.py \
  --method image_uniedit_rf_inversion \
  --source-image "$SOURCE_IMAGE" \
  --edit-image "$EDIT_IMAGE" \
  --mask-image "$MASK_IMAGE" \
  --source-model "$SOURCE_MODEL" \
  --mask-glb "$MASK_GLB" \
  --case-name "${CASE_NAME}_preserve_uniedit" \
  --seed $SEED \
  --preprocess \
  stage2_variant=preserve_uniedit \
  decode_modes=gaussian,mesh

echo "✓ Test 1 完成"
echo ""

# Test 2: free_target
echo "Test 2/2: free_target"
python run_edit_experiment.py \
  --method image_uniedit_rf_inversion \
  --source-image "$SOURCE_IMAGE" \
  --edit-image "$EDIT_IMAGE" \
  --mask-image "$MASK_IMAGE" \
  --source-model "$SOURCE_MODEL" \
  --mask-glb "$MASK_GLB" \
  --case-name "${CASE_NAME}_free_target" \
  --seed $SEED \
  --preprocess \
  stage2_variant=free_target \
  decode_modes=gaussian,mesh

echo "✓ Test 2 完成"
echo ""

echo "=========================================="
echo "测试完成！"
echo "=========================================="
