#!/bin/bash
# 测试 latent_replace_union 的显存优化

set -e

CASE_NAME="test0"
SEED=1
SOURCE_IMAGE="assets/edit_example/images/2d_render.png"
EDIT_IMAGE="assets/edit_example/images/2d_edit.png"
MASK_IMAGE="assets/edit_example/images/2d_mask.png"
SOURCE_MODEL="outputs/source_assets_test0_multiview"
MASK_GLB="assets/edit_example/mask.glb"

echo "=========================================="
echo "测试 latent_replace_union (显存优化)"
echo "=========================================="
echo "Case: $CASE_NAME"
echo "Seed: $SEED"
echo ""

# 使用 PYTORCH_CUDA_ALLOC_CONF 减少碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python run_edit_experiment.py \
  --method image_uniedit_rf_inversion \
  --source-image "$SOURCE_IMAGE" \
  --edit-image "$EDIT_IMAGE" \
  --mask-image "$MASK_IMAGE" \
  --source-model "$SOURCE_MODEL" \
  --mask-glb "$MASK_GLB" \
  --case-name "${CASE_NAME}_latent_replace_union_optimized" \
  --seed $SEED \
  --preprocess \
  stage2_variant=latent_replace_union \
  decode_modes=gaussian,mesh

echo ""
echo "✓ 测试完成"
