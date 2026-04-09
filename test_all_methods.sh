#!/bin/bash

# 测试所有已迁移的编辑方法
# 使用统一的 case 名称: test1
# UniEdit 的三种消融模式作为独立方法测试

set -e  # 遇到错误立即退出

CASE_NAME="test2"
SEED=1
GPU=3

# 输入文件路径
SOURCE_IMAGE="assets/edit_example/images/2d_render.png"
EDIT_IMAGE="assets/edit_example/images/2d_edit.png"
MASK_IMAGE="assets/edit_example/images/2d_mask.png"
MASK_GLB="assets/edit_example/mask.glb"

# Source model 路径（使用多视角渲染生成的）
SOURCE_MODEL="outputs/source_assets_test0_multiview"

echo "=========================================="
echo "开始测试所有编辑方法"
echo "Case 名称: $CASE_NAME"
echo "GPU: $GPU"
echo "=========================================="
echo ""

# 检查 source model 是否存在
if [ ! -d "$SOURCE_MODEL" ]; then
    echo "❌ Source model 不存在: $SOURCE_MODEL"
    echo "请先运行: CUDA_VISIBLE_DEVICES=3 python generate_source_assets.py"
    exit 1
fi

if [ ! -f "$SOURCE_MODEL/voxels.ply" ] || [ ! -f "$SOURCE_MODEL/features.npz" ]; then
    echo "❌ Source model 缺少必要文件 (voxels.ply 或 features.npz)"
    exit 1
fi

echo "✓ Source model 已准备: $SOURCE_MODEL"
echo ""

# 1. Image Prompt-to-Prompt
echo ">>> 测试 1/6: image_prompt_to_prompt"
echo "预计时间: ~3 分钟"
CUDA_VISIBLE_DEVICES=$GPU python run_edit_experiment.py \
  --method image_prompt_to_prompt \
  --source-image "$SOURCE_IMAGE" \
  --edit-image "$EDIT_IMAGE" \
  --mask-image "$MASK_IMAGE" \
  --case-name "$CASE_NAME" \
  --seed $SEED \
  --preprocess

if [ $? -eq 0 ]; then
    echo "✅ image_prompt_to_prompt 测试通过"
else
    echo "❌ image_prompt_to_prompt 测试失败"
    exit 1
fi
echo ""

# 2. Image SLAT XOR Fusion
echo ">>> 测试 2/6: image_slat_xor_fusion"
echo "预计时间: ~2 分钟"
CUDA_VISIBLE_DEVICES=$GPU python run_edit_experiment.py \
  --method image_slat_xor_fusion \
  --asset-dir "$SOURCE_MODEL" \
  --edit-image "$EDIT_IMAGE" \
  --case-name "$CASE_NAME" \
  --seed $SEED
# 注意: image_slat_xor_fusion 不需要 mask，融合基于 SLAT 坐标重叠

if [ $? -eq 0 ]; then
    echo "✅ image_slat_xor_fusion 测试通过"
else
    echo "❌ image_slat_xor_fusion 测试失败"
    exit 1
fi
echo ""

# 3. Image Prompt-to-Prompt RF Inversion
echo ">>> 测试 3/6: image_prompt_to_prompt_rf_inversion"
echo "预计时间: ~4 分钟"
CUDA_VISIBLE_DEVICES=$GPU python run_edit_experiment.py \
  --method image_prompt_to_prompt_rf_inversion \
  --asset-dir "$SOURCE_MODEL" \
  --source-image "$SOURCE_IMAGE" \
  --edit-image "$EDIT_IMAGE" \
  --mask-image "$MASK_IMAGE" \
  --case-name "$CASE_NAME" \
  --seed $SEED \
  --ss-steps 12 \
  --slat-steps 12

if [ $? -eq 0 ]; then
    echo "✅ image_prompt_to_prompt_rf_inversion 测试通过"
else
    echo "❌ image_prompt_to_prompt_rf_inversion 测试失败"
    exit 1
fi
echo ""

# 4. UniEdit RF Inversion - preserve_uniedit (默认)
echo ">>> 测试 4/6: image_uniedit_rf_inversion (preserve_uniedit)"
echo "预计时间: ~5 分钟"
if [ ! -f "$MASK_GLB" ]; then
    echo "⚠️  mask.glb 不存在，跳过此测试"
else
    conda run -n hammer bash -c "ATTN_BACKEND=flash_attn CUDA_VISIBLE_DEVICES=$GPU python run_edit_experiment.py \
      --method image_uniedit_rf_inversion \
      --asset-dir '$SOURCE_MODEL' \
      --source-image '$SOURCE_IMAGE' \
      --edit-image '$EDIT_IMAGE' \
      --mask-glb '$MASK_GLB' \
      --case-name '$CASE_NAME' \
      --seed $SEED \
      --ss-steps 12 \
      --slat-steps 12 \
      --extra-param stage2_variant=preserve_uniedit \
      --extra-param decode_modes=mesh,gaussian"

    if [ $? -eq 0 ]; then
        echo "✅ image_uniedit_rf_inversion (preserve_uniedit) 测试通过"
    else
        echo "❌ image_uniedit_rf_inversion (preserve_uniedit) 测试失败"
        exit 1
    fi
fi
echo ""

# 5. UniEdit RF Inversion - free_target
echo ">>> 测试 5/6: image_uniedit_rf_inversion_free_target"
echo "预计时间: ~4 分钟"
if [ ! -f "$MASK_GLB" ]; then
    echo "⚠️  mask.glb 不存在，跳过此测试"
else
    conda run -n hammer bash -c "ATTN_BACKEND=flash_attn CUDA_VISIBLE_DEVICES=$GPU python run_edit_experiment.py \
      --method image_uniedit_rf_inversion \
      --asset-dir '$SOURCE_MODEL' \
      --source-image '$SOURCE_IMAGE' \
      --edit-image '$EDIT_IMAGE' \
      --mask-glb '$MASK_GLB' \
      --case-name 'test0_free_target' \
      --seed $SEED \
      --ss-steps 12 \
      --slat-steps 12 \
      --extra-param stage2_variant=free_target \
      --extra-param decode_modes=mesh,gaussian"

    if [ $? -eq 0 ]; then
        echo "✅ image_uniedit_rf_inversion (free_target) 测试通过"
    else
        echo "❌ image_uniedit_rf_inversion (free_target) 测试失败"
        exit 1
    fi
fi
echo ""

# 6. UniEdit RF Inversion - latent_replace_union
echo ">>> 测试 6/6: image_uniedit_rf_inversion_latent_replace"
echo "预计时间: ~5 分钟"
if [ ! -f "$MASK_GLB" ]; then
    echo "⚠️  mask.glb 不存在，跳过此测试"
else
    conda run -n hammer bash -c "ATTN_BACKEND=flash_attn CUDA_VISIBLE_DEVICES=$GPU python run_edit_experiment.py \
      --method image_uniedit_rf_inversion \
      --asset-dir '$SOURCE_MODEL' \
      --source-image '$SOURCE_IMAGE' \
      --edit-image '$EDIT_IMAGE' \
      --mask-glb '$MASK_GLB' \
      --case-name 'test0_latent_replace' \
      --seed $SEED \
      --ss-steps 12 \
      --slat-steps 12 \
      --extra-param stage2_variant=latent_replace_union \
      --extra-param decode_modes=mesh,gaussian"

    if [ $? -eq 0 ]; then
        echo "✅ image_uniedit_rf_inversion (latent_replace_union) 测试通过"
    else
        echo "❌ image_uniedit_rf_inversion (latent_replace_union) 测试失败"
        exit 1
    fi
fi
echo ""

echo "=========================================="
echo "所有测试完成！"
echo "=========================================="
echo ""
echo "查看结果："
echo "  outputs/image_prompt_to_prompt/$CASE_NAME/"
echo "  outputs/image_slat_xor_fusion/$CASE_NAME/"
echo "  outputs/image_prompt_to_prompt_rf_inversion/$CASE_NAME/"
echo "  outputs/image_uniedit_rf_inversion/$CASE_NAME/"
echo "  outputs/image_uniedit_rf_inversion/test0_free_target/"
echo "  outputs/image_uniedit_rf_inversion/test0_latent_replace/"
