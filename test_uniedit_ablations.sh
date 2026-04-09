#!/bin/bash
# UniEdit 消融测试脚本
# 测试 3 个 stage2_variant 模式

set -e

CASE_NAME="test0"
SEED=1
SOURCE_IMAGE="assets/edit_example/images/2d_render.png"
EDIT_IMAGE="assets/edit_example/images/2d_edit.png"
MASK_IMAGE="assets/edit_example/images/2d_mask.png"
SOURCE_MODEL="outputs/source_assets_test0_multiview"
MASK_GLB="assets/edit_example/mask.glb"

echo "=========================================="
echo "UniEdit 消融测试"
echo "=========================================="
echo "Case: $CASE_NAME"
echo "Seed: $SEED"
echo ""

# Test 1: preserve_uniedit (重新运行，之前未生成模型)
echo "=========================================="
echo "Test 1/3: preserve_uniedit"
echo "=========================================="
python run_edit_experiment.py \
  --method image_uniedit_rf_inversion \
  --source-image "$SOURCE_IMAGE" \
  --edit-image "$EDIT_IMAGE" \
  --mask-image "$MASK_IMAGE" \
  --asset-dir "$SOURCE_MODEL" \
  --mask-glb "$MASK_GLB" \
  --case-name "${CASE_NAME}_preserve_uniedit" \
  --seed $SEED \
  --preprocess \
  stage2_variant=preserve_uniedit

echo ""
echo "✓ Test 1 完成"
echo ""

# Test 2: free_target
echo "=========================================="
echo "Test 2/3: free_target"
echo "=========================================="
python run_edit_experiment.py \
  --method image_uniedit_rf_inversion \
  --source-image "$SOURCE_IMAGE" \
  --edit-image "$EDIT_IMAGE" \
  --mask-image "$MASK_IMAGE" \
  --asset-dir "$SOURCE_MODEL" \
  --mask-glb "$MASK_GLB" \
  --case-name "${CASE_NAME}_free_target" \
  --seed $SEED \
  --preprocess \
  stage2_variant=free_target

echo ""
echo "✓ Test 2 完成"
echo ""

# Test 3: latent_replace_union
echo "=========================================="
echo "Test 3/3: latent_replace_union"
echo "=========================================="
python run_edit_experiment.py \
  --method image_uniedit_rf_inversion \
  --source-image "$SOURCE_IMAGE" \
  --edit-image "$EDIT_IMAGE" \
  --mask-image "$MASK_IMAGE" \
  --asset-dir "$SOURCE_MODEL" \
  --mask-glb "$MASK_GLB" \
  --case-name "${CASE_NAME}_latent_replace_union" \
  --seed $SEED \
  --preprocess \
  stage2_variant=latent_replace_union

echo ""
echo "✓ Test 3 完成"
echo ""

echo "=========================================="
echo "所有测试完成！"
echo "=========================================="
echo ""
echo "输出目录："
echo "  - outputs/image_uniedit_rf_inversion/${CASE_NAME}_preserve_uniedit/"
echo "  - outputs/image_uniedit_rf_inversion/${CASE_NAME}_free_target/"
echo "  - outputs/image_uniedit_rf_inversion/${CASE_NAME}_latent_replace_union/"
