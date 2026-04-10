#!/bin/bash

# 测试 GSO Dollhouse Happy Brother prompt_1 在禁用 P2P 情况下的结果

# 激活 hammer 环境
conda activate hammer

# 设置环境变量
export ATTN_BACKEND='flash-attn'
export SPCONV_ALGO='native'
export CUDA_VISIBLE_DEVICES=3
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
# 数据路径
DATA_ROOT="/home/wangxinxing/code/Edit3Dpp/data/GSO/3D_Dollhouse_Happy_Brother"
SOURCE_IMG="${DATA_ROOT}/prompt_1/2d_render.png"
EDIT_IMG="${DATA_ROOT}/prompt_1/2d_edit.png"
MASK_IMG="${DATA_ROOT}/prompt_1/2d_mask.png"
ASSET_DIR="/cache/wangxinxing/data/temp/renders/GSO/3D_Dollhouse_Happy_Brother"
MASK_GLB_PATH="/home/wangxinxing/code/Edit3Dpp/data/GSO/3D_Dollhouse_Happy_Brother/prompt_1/3d_edit_region.glb"
# 运行测试 - 禁用 P2P 注入
python run_edit_experiment.py \
  --method image_p2p_latent_blend \
  --asset-dir "${ASSET_DIR}" \
  --source-image "${SOURCE_IMG}" \
  --edit-image "${EDIT_IMG}" \
  --mask-image "${MASK_IMG}" \
  --mask-glb "${MASK_GLB_PATH}" \
  --case-name dollhouse_no_p2p_2 \
  --seed 1 \
  --preprocess \
  --inject-stages "" \
  --blend-ss-enabled \
  --blend-slat-enabled \
  --ss-blend-mode soft \
  --slat-blend-mode soft \
  --ss-soft-kernel-size 5 \
  --slat-soft-kernel-size 7

echo "测试完成，结果保存在: outputs/image_p2p_latent_blend/dollhouse_no_p2p/"
