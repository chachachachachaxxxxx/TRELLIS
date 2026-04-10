#!/bin/bash

# 使用 YAML 配置运行单个编辑实验的示例

# 激活 hammer 环境
source ~/miniforge3/etc/profile.d/conda.sh
conda activate hammer

# 设置环境变量
export ATTN_BACKEND='flash-attn'
export SPCONV_ALGO='native'
export CUDA_VISIBLE_DEVICES=0

# 使用 YAML 配置运行编辑实验
python run_edit_experiment.py --config configs/edit_experiment_example.yaml

echo ""
echo "============================================"
echo "编辑完成！"
echo "============================================"
echo "输出位置: outputs/image_p2p_latent_blend/test_edit/"
echo "============================================"
