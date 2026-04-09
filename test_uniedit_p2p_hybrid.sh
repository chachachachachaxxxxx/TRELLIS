#!/bin/bash

# 测试 UniEdit + P2P 混合方法
# 在去噪阶段同时使用 UniEdit 的 latent replacement 和 P2P 的 attention injection

set -e

CASE_NAME="hybrid_test1"
SEED=1
GPU=3

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$GPU
export ATTN_BACKEND=flash-attn
export SPARSE_ATTN_BACKEND=flash-attn
export SPCONV_ALGO=native

echo "=========================================="
echo "Testing image_uniedit_p2p_hybrid method"
echo "=========================================="

# 使用 outputs/source_assets_test0_multiview 中的预生成资产（新格式）
SOURCE_MODEL="outputs/source_assets_test0_multiview"
SOURCE_IMAGE="assets/edit_example/images/2d_render.png"
EDIT_IMAGE="assets/edit_example/images/2d_edit.png"
MASK_IMAGE="assets/edit_example/images/2d_mask.png"
MASK_GLB="assets/edit_example/mask.glb"

# 检查文件是否存在
if [ ! -d "$SOURCE_MODEL" ]; then
    echo "❌ Source model not found: $SOURCE_MODEL"
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

if [ ! -f "$MASK_GLB" ]; then
    echo "❌ Mask GLB not found: $MASK_GLB"
    exit 1
fi

echo "Running hybrid method with:"
echo "  - UniEdit latent replacement (stage2_variant=latent_replace_union)"
echo "  - P2P attention injection (inject_stages=slat, strength=0.8)"
echo ""

python run_edit_experiment.py \
  --method image_uniedit_p2p_hybrid \
  --asset-dir "$SOURCE_MODEL" \
  --source-image "$SOURCE_IMAGE" \
  --edit-image "$EDIT_IMAGE" \
  --mask-image "$MASK_IMAGE" \
  --mask-glb "$MASK_GLB" \
  --case-name "$CASE_NAME" \
  --seed $SEED \
  --preprocess \
  --ss-steps 12 \
  --slat-steps 12 \
  --extra-param stage2_variant=latent_replace_union \
  --extra-param inject_stages=slat \
  --extra-param p2p_strength=0.8 \
  --extra-param slat_t_start=0.8 \
  --extra-param slat_t_end=0.0 \
  --extra-param decode_modes=mesh,gaussian

if [ $? -eq 0 ]; then
    echo "✅ image_uniedit_p2p_hybrid 测试通过"
    echo "输出目录: outputs/image_uniedit_p2p_hybrid/$CASE_NAME"
else
    echo "❌ image_uniedit_p2p_hybrid 测试失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
