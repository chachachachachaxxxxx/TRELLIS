#!/bin/bash
set -e

CASE_NAME="test0"
SEED=1
GPU=3
SOURCE_IMAGE="assets/edit_example/images/2d_render.png"
EDIT_IMAGE="assets/edit_example/images/2d_edit.png"
MASK_IMAGE="assets/edit_example/images/2d_mask.png"
MASK_GLB="assets/edit_example/mask.glb"
SOURCE_MODEL="outputs/source_assets_test0_multiview"

echo "=========================================="
echo "运行剩余测试 (3-6)"
echo "=========================================="
echo ""

# 3. Image Prompt-to-Prompt RF Inversion
echo ">>> 测试 3/6: image_prompt_to_prompt_rf_inversion"
CUDA_VISIBLE_DEVICES=$GPU python run_edit_experiment.py \
  --method image_prompt_to_prompt_rf_inversion \
  --source-model "$SOURCE_MODEL" \
  --source-image "$SOURCE_IMAGE" \
  --edit-image "$EDIT_IMAGE" \
  --mask-image "$MASK_IMAGE" \
  --case-name "$CASE_NAME" \
  --seed $SEED \
  --ss-steps 12 \
  --slat-steps 12

echo "✅ image_prompt_to_prompt_rf_inversion 测试通过"
echo ""

# 4. UniEdit RF Inversion - preserve_uniedit
echo ">>> 测试 4/6: image_uniedit_rf_inversion (preserve_uniedit)"
conda run -n hammer bash -c "ATTN_BACKEND=flash_attn CUDA_VISIBLE_DEVICES=$GPU python run_edit_experiment.py \
  --method image_uniedit_rf_inversion \
  --source-model '$SOURCE_MODEL' \
  --source-image '$SOURCE_IMAGE' \
  --edit-image '$EDIT_IMAGE' \
  --mask-glb '$MASK_GLB' \
  --case-name '$CASE_NAME' \
  --seed $SEED \
  --ss-steps 12 \
  --slat-steps 12 \
  --extra-param stage2_variant=preserve_uniedit \
  --extra-param decode_modes='[\"mesh\"]'"

echo "✅ image_uniedit_rf_inversion (preserve_uniedit) 测试通过"
echo ""

# 5. UniEdit RF Inversion - free_target
echo ">>> 测试 5/6: image_uniedit_rf_inversion (free_target)"
conda run -n hammer bash -c "ATTN_BACKEND=flash_attn CUDA_VISIBLE_DEVICES=$GPU python run_edit_experiment.py \
  --method image_uniedit_rf_inversion \
  --source-model '$SOURCE_MODEL' \
  --source-image '$SOURCE_IMAGE' \
  --edit-image '$EDIT_IMAGE' \
  --mask-glb '$MASK_GLB' \
  --case-name 'test0_free_target' \
  --seed $SEED \
  --ss-steps 12 \
  --slat-steps 12 \
  --extra-param stage2_variant=free_target \
  --extra-param decode_modes='[\"mesh\"]'"

echo "✅ image_uniedit_rf_inversion (free_target) 测试通过"
echo ""

# 6. UniEdit RF Inversion - latent_replace_union
echo ">>> 测试 6/6: image_uniedit_rf_inversion (latent_replace_union)"
conda run -n hammer bash -c "ATTN_BACKEND=flash_attn CUDA_VISIBLE_DEVICES=$GPU python run_edit_experiment.py \
  --method image_uniedit_rf_inversion \
  --source-model '$SOURCE_MODEL' \
  --source-image '$SOURCE_IMAGE' \
  --edit-image '$EDIT_IMAGE' \
  --mask-glb '$MASK_GLB' \
  --case-name 'test0_latent_replace' \
  --seed $SEED \
  --ss-steps 12 \
  --slat-steps 12 \
  --extra-param stage2_variant=latent_replace_union \
  --extra-param decode_modes='[\"mesh\"]'"

echo "✅ image_uniedit_rf_inversion (latent_replace_union) 测试通过"
echo ""

echo "=========================================="
echo "所有剩余测试完成！"
echo "=========================================="
