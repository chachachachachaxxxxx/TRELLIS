#!/bin/bash

# 测试 GSO 数据集在禁用 P2P 情况下的结果
# 使用 YAML 配置文件简化参数管理

# 激活 hammer 环境
source ~/miniforge3/etc/profile.d/conda.sh
conda activate hammer

# 设置环境变量
export ATTN_BACKEND='flash-attn'
export SPCONV_ALGO='native'
export CUDA_VISIBLE_DEVICES=3
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890

# 运行批量编辑和评测（使用 YAML 配置）
python run_batch_edit_and_eval.py \
  --config configs/gso_no_p2p.yaml \
  --max-cases 1

echo ""
echo "============================================"
echo "测试完成！"
echo "============================================"
echo "edit.glb 保存在: /cache/wangxinxing/data/temp/image_p2p_latent_blend_gso_no_p2p/"
echo "评测结果保存在: outputs/results/image_p2p_latent_blend_gso_no_p2p_*/"
echo ""
echo "查看评测报告:"
echo "cat outputs/results/image_p2p_latent_blend_gso_no_p2p_*/report.txt"
echo "============================================"
