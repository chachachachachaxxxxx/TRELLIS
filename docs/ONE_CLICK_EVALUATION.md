# 一键式评测脚本使用指南

## 概述

`run_full_evaluation.py` 是一个一键式评测脚本，自动完成以下步骤：

1. 将编辑结果转换为 Edit3D-Bench 格式
2. 调用评测系统运行评测
3. 保存结果到 `outputs/results/{method_name}_{timestamp}/`

## 快速开始

### 基本用法

```bash
python run_full_evaluation.py \
  --gt-root /home/wangxinxing/code/Edit3Dpp/data \
  --method-name image_prompt_to_prompt
```

### 快速测试（1个案例）

```bash
python run_full_evaluation.py \
  --gt-root /home/wangxinxing/code/Edit3Dpp/data \
  --method-name image_prompt_to_prompt \
  --max-cases 1
```

### 仅评测特定数据集

```bash
python run_full_evaluation.py \
  --gt-root /home/wangxinxing/code/Edit3Dpp/data \
  --method-name image_prompt_to_prompt \
  --dataset GSO
```

## 参数说明

### 必需参数

- `--gt-root`: Edit3D-Bench Ground Truth 数据根目录
- `--method-name`: 方法名称（例如：`image_prompt_to_prompt`）

### 可选参数

**数据过滤：**
- `--outputs-root`: TRELLIS_EDIT 输出目录（默认：`outputs`）
- `--dataset`: 仅评测指定数据集（例如：`GSO`）
- `--max-cases`: 限制评测案例数量（用于快速测试）

**评测配置：**
- `--metrics`: 评测指标列表（默认：`psnr ssim lpips fid dino_if chamfer clip_t`）
- `--device`: 计算设备（默认：`cuda:0`）
- `--batch-size`: 批处理大小（默认：`32`）

**其他选项：**
- `--skip-prepare`: 跳过数据准备步骤（假设数据已准备好）
- `--keep-temp`: 保留临时评测数据（默认会清理）

## 输出结果

### 评测数据位置

所有 `edit.glb` 文件保存在固定目录：

```
/cache/wangxinxing/data/temp/{method_name}_{timestamp}/
└── {method_name}/
    └── {dataset}/
        └── {object_name}/
            └── prompt_{1,2,3}/
                └── edit.glb
```

**示例：**
```
/cache/wangxinxing/data/temp/image_prompt_to_prompt_20260410_104525/
└── image_prompt_to_prompt/
    └── GSO/
        └── 3D_Dollhouse_Happy_Brother/
            └── prompt_1/
                └── edit.glb
```

**注意：** 这些文件不会被自动清理，需要手动删除。

### 评测结果位置

评测完成后，结果保存在 `outputs/results/{method_name}_{timestamp}/`：

```
outputs/results/image_prompt_to_prompt_20260410_103940/
├── evaluation_results.json    # 完整结果（JSON格式）
├── report.txt                 # 可读报告
├── detailed_results.json      # 详细结果（每个样本）
└── summary.json               # 汇总结果
```

### 结果文件说明

**1. evaluation_results.json** - 完整结果

```json
{
  "method_name": "image_prompt_to_prompt",
  "timestamp": "20260410_103940",
  "datetime": "2026-04-10T10:39:40",
  "success_count": 150,
  "total_time_seconds": 3600.5,
  "total_time_formatted": "1h 0m 0s",
  "evaluation_results": {
    "results": {
      "psnr": {"mean": 25.1234, "std": 2.3456, "count": 150},
      "ssim": {"mean": 0.8765, "std": 0.0456, "count": 150},
      ...
    }
  }
}
```

**2. report.txt** - 可读报告

```
================================================================================
Edit3D-Bench 评测报告
================================================================================

方法名称: image_prompt_to_prompt
评测时间: 2026-04-10 10:39:40
成功案例: 150
总耗时: 1h 0m 0s

================================================================================
评测指标结果
================================================================================

PSNR:
  均值: 25.1234
  标准差: 2.3456
  样本数: 150

SSIM:
  均值: 0.8765
  标准差: 0.0456
  样本数: 150

...
```

## 命名规范

脚本会自动查找编辑结果，支持以下命名模式：

1. `{object_name}_prompt_{prompt_id}/edit/sample_00.glb`
2. `{dataset}_{object_name}_prompt_{prompt_id}/edit/sample_00.glb`
3. `{object_name}_p{prompt_id}/edit/sample_00.glb`
4. `{object_name}/edit/sample_00.glb`

**示例：**

对于 metadata 中的案例：
- Dataset: `GSO`
- Object: `3D_Dollhouse_Happy_Brother`
- Prompt ID: `1`

脚本会查找：
```
outputs/image_prompt_to_prompt/3D_Dollhouse_Happy_Brother_prompt_1/edit/sample_00.glb
outputs/image_prompt_to_prompt/GSO_3D_Dollhouse_Happy_Brother_prompt_1/edit/sample_00.glb
outputs/image_prompt_to_prompt/3D_Dollhouse_Happy_Brother_p1/edit/sample_00.glb
outputs/image_prompt_to_prompt/3D_Dollhouse_Happy_Brother/edit/sample_00.glb
```

## 常见问题

### 1. 找不到编辑结果

**问题：** `[SKIP] No editing result found`

**解决方案：**
- 确保案例名称包含物体名称
- 使用 `--max-cases 1` 测试单个案例
- 检查 `outputs/{method_name}/` 目录结构

### 2. 评测脚本未找到

**问题：** `找不到评测脚本`

**解决方案：**
- 确保 VoxHammer 仓库在正确位置（`VoxHammer/Edit3D-Bench/`）
- 或者使用 `--skip-prepare` 只准备数据，稍后手动评测

### 3. 缺少依赖

**问题：** `ModuleNotFoundError: No module named 'torchmetrics'`

**解决方案：**
```bash
pip install torchmetrics pytorch-fid lpips
```

### 4. CUDA 内存不足

**问题：** CUDA out of memory

**解决方案：**
```bash
# 减小批处理大小
python run_full_evaluation.py \
  --gt-root /path/to/data \
  --method-name image_prompt_to_prompt \
  --batch-size 8

# 或使用 CPU（会很慢）
python run_full_evaluation.py \
  --gt-root /path/to/data \
  --method-name image_prompt_to_prompt \
  --device cpu
```

## 高级用法

### 仅准备数据，不运行评测

```bash
# 准备数据后停止
python batch_prepare_eval.py \
  --gt-root /home/wangxinxing/code/Edit3Dpp/data \
  --method-name image_prompt_to_prompt \
  --eval-root exp_comparison

# 稍后手动运行评测
cd VoxHammer/Edit3D-Bench
python eval_main.py \
  --gt_root /home/wangxinxing/code/Edit3Dpp/data \
  --pred_root ../../exp_comparison/image_prompt_to_prompt \
  --metrics psnr ssim lpips \
  --output_dir evaluation_results
```

### 使用已准备的数据

```bash
# 跳过数据准备，直接评测
python run_full_evaluation.py \
  --gt-root /home/wangxinxing/code/Edit3Dpp/data \
  --method-name image_prompt_to_prompt \
  --skip-prepare
```

### 保留临时数据用于调试

```bash
# 数据默认保留在 /cache/wangxinxing/data/temp/
python run_full_evaluation.py \
  --gt-root /home/wangxinxing/code/Edit3Dpp/data \
  --method-name image_prompt_to_prompt

# 查看保存的 edit.glb 文件
ls /cache/wangxinxing/data/temp/image_prompt_to_prompt_*/image_prompt_to_prompt/

# 手动清理（如需要）
rm -rf /cache/wangxinxing/data/temp/image_prompt_to_prompt_*
```

## 工作流程

完整的评测工作流程：

```bash
# 1. 运行编辑方法
python run_edit_experiment.py \
  --method image_prompt_to_prompt \
  --source-image assets/source.png \
  --edit-image assets/edit.png \
  --mask-image assets/mask.png \
  --case-name 3D_Dollhouse_Happy_Brother_prompt_1 \
  --seed 1

# 2. 运行一键式评测
python run_full_evaluation.py \
  --gt-root /home/wangxinxing/code/Edit3Dpp/data \
  --method-name image_prompt_to_prompt

# 3. 查看结果
cat outputs/results/image_prompt_to_prompt_*/report.txt
```

## 性能优化

### 批量处理多个方法

```bash
# 创建批处理脚本
cat > batch_eval.sh << 'EOF'
#!/bin/bash
methods=("image_prompt_to_prompt" "image_slat_xor_fusion" "text_prompt_to_prompt")

for method in "${methods[@]}"; do
  echo "Evaluating $method..."
  python run_full_evaluation.py \
    --gt-root /home/wangxinxing/code/Edit3Dpp/data \
    --method-name "$method"
done
EOF

chmod +x batch_eval.sh
./batch_eval.sh
```

### 并行评测不同数据集

```bash
# 在不同终端或使用 tmux
python run_full_evaluation.py --dataset GSO --device cuda:0 &
python run_full_evaluation.py --dataset PartObjaverse-Tiny --device cuda:1 &
```

## 相关文档

- `EVAL_OUTPUT_FORMAT.md` - 评测输出格式详细说明
- `EVALUATION_GUIDE.md` - Edit3D-Bench 评测系统完整指南
- `batch_prepare_eval.py` - 数据准备脚本
- `run_full_evaluation.py` - 一键式评测脚本源码
