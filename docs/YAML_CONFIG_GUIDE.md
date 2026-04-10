# YAML 配置文件使用指南

本文档说明如何使用 YAML 配置文件简化脚本参数管理。

## 概述

两个主要脚本现在都支持 YAML 配置文件：

1. `run_edit_experiment.py` - 单个编辑实验
2. `run_batch_edit_and_eval.py` - 批量编辑和评测

使用 YAML 配置的优势：
- 参数集中管理，易于维护
- 避免长命令行参数
- 便于版本控制和复用
- 命令行参数可覆盖配置文件

## 单个编辑实验配置

### 配置文件示例

`configs/edit_experiment_example.yaml`:

```yaml
# 方法名称
method: image_p2p_latent_blend

# 案例名称
case_name: test_edit

# 输入文件
source_image: assets/edit_example/images/2d_render.png
edit_image: assets/edit_example/images/2d_edit.png
mask_image: assets/edit_example/images/2d_mask.png

# 预处理选项
preprocess: true

# 基础参数
seed: 1
model: microsoft/TRELLIS-image-large

# 环境配置
attn_backend: flash-attn
spconv_algo: native

# 采样步数
ss_steps: 25
slat_steps: 25

# 方法参数
method_args:
  inject-stages: ""
  blend-ss-enabled: true
  blend-slat-enabled: true
  ss-blend-mode: soft
  slat-blend-mode: soft
  ss-soft-kernel-size: 5
  slat-soft-kernel-size: 7
```

### 使用方法

```bash
# 使用配置文件
python run_edit_experiment.py --config configs/edit_experiment_example.yaml

# 命令行参数覆盖配置
python run_edit_experiment.py \
  --config configs/edit_experiment_example.yaml \
  --seed 42 \
  --case-name my_custom_name
```

## 批量编辑和评测配置

### 配置文件示例

`configs/gso_no_p2p.yaml`:

```yaml
# Edit3D-Bench GT 数据根目录
gt_root: /home/wangxinxing/code/Edit3Dpp/data

# 方法名称
method_name: image_p2p_latent_blend

# 配置名称
config_name: gso_no_p2p

# 数据集过滤
dataset: GSO

# 基础参数
seed: 1
device: cuda:0

# Assets 路径
assets_root: /cache/wangxinxing/data/temp/renders

# 评测参数
metrics:
  - psnr
  - ssim
  - lpips
  - fid
  - dino_if
  - chamfer
  - clip_t

skip_render: false

# 方法参数
method_args:
  ss-steps: 25
  slat-steps: 25
  inject-stages: ""
  blend-ss-enabled: true
  blend-slat-enabled: true
  ss-blend-mode: soft
  slat-blend-mode: soft
  ss-soft-kernel-size: 5
  slat-soft-kernel-size: 7
```

### 使用方法

```bash
# 使用配置文件
python run_batch_edit_and_eval.py --config configs/gso_no_p2p.yaml

# 限制案例数量
python run_batch_edit_and_eval.py \
  --config configs/gso_no_p2p.yaml \
  --max-cases 5

# 覆盖数据集
python run_batch_edit_and_eval.py \
  --config configs/gso_no_p2p.yaml \
  --dataset PartObjaverse-Tiny
```

## 配置文件参数说明

### 单个编辑实验 (`run_edit_experiment.py`)

| 参数 | 类型 | 说明 |
|------|------|------|
| `method` | string | 方法名称 |
| `case_name` | string | 案例名称 |
| `source_image` | string | 源图像路径 |
| `edit_image` | string | 编辑图像路径 |
| `mask_image` | string | 遮罩图像路径 |
| `mask_glb` | string | 3D 遮罩 GLB 路径（可选）|
| `source_model` | string | 源模型路径（可选）|
| `asset_dir` | string | 预处理 assets 目录（可选）|
| `preprocess` | boolean | 是否预处理 |
| `seed` | integer | 随机种子 |
| `model` | string | 模型路径或 HF repo |
| `attn_backend` | string | 注意力后端 |
| `spconv_algo` | string | spconv 算法 |
| `ss_steps` | integer | SS 采样步数 |
| `slat_steps` | integer | SLAT 采样步数 |
| `skip_render` | boolean | 跳过渲染 |
| `skip_glb` | boolean | 跳过 GLB 导出 |
| `skip_ply` | boolean | 跳过 PLY 导出 |
| `method_args` | dict | 方法特定参数 |

### 批量编辑和评测 (`run_batch_edit_and_eval.py`)

| 参数 | 类型 | 说明 |
|------|------|------|
| `gt_root` | string | Edit3D-Bench GT 数据根目录 |
| `method_name` | string | 方法名称 |
| `config_name` | string | 配置名称 |
| `dataset` | string | 数据集过滤（可选）|
| `object` | string | 物体名称过滤（可选）|
| `prompt_id` | integer | 提示 ID 过滤（可选）|
| `max_cases` | integer | 案例数量限制（可选）|
| `seed` | integer | 随机种子 |
| `device` | string | 计算设备 |
| `assets_root` | string | 预处理 assets 根目录（可选）|
| `metrics` | list | 评测指标列表 |
| `skip_render` | boolean | 跳过渲染 |
| `method_args` | dict | 方法特定参数 |

## 方法参数 (`method_args`)

方法参数以字典形式定义，会自动转换为命令行参数：

```yaml
method_args:
  ss-steps: 25                    # 转换为 --ss-steps 25
  slat-steps: 25                  # 转换为 --slat-steps 25
  inject-stages: ""               # 转换为 --inject-stages ""
  blend-ss-enabled: true          # 转换为 --blend-ss-enabled
  ss-blend-mode: soft             # 转换为 --ss-blend-mode soft
```

注意：
- 布尔值 `true` 转换为标志参数（不带值）
- 空字符串 `""` 会传递空字符串值
- `null` 或省略的参数不会传递

## 示例脚本

### 单个编辑实验

`test_edit_with_config.sh`:

```bash
#!/bin/bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate hammer

export ATTN_BACKEND='flash-attn'
export SPCONV_ALGO='native'
export CUDA_VISIBLE_DEVICES=0

python run_edit_experiment.py --config configs/edit_experiment_example.yaml
```

### 批量编辑和评测

`test_batch_GSO_no_p2p.sh`:

```bash
#!/bin/bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate hammer

export ATTN_BACKEND='flash-attn'
export SPCONV_ALGO='native'
export CUDA_VISIBLE_DEVICES=3

python run_batch_edit_and_eval.py \
  --config configs/gso_no_p2p.yaml \
  --max-cases 1
```

## 最佳实践

1. **配置文件命名**：使用描述性名称，如 `gso_no_p2p.yaml`、`dollhouse_with_p2p.yaml`

2. **版本控制**：将配置文件提交到 git，便于追踪实验参数

3. **参数覆盖**：使用命令行参数覆盖配置文件中的特定值，无需修改配置文件

4. **配置复用**：为不同实验创建多个配置文件，便于快速切换

5. **注释说明**：在配置文件中添加注释，说明参数用途

## 故障排查

### 配置文件未找到

```
[ERROR] Config file not found: configs/xxx.yaml
```

检查：
- 配置文件路径是否正确
- 文件是否存在
- 使用绝对路径或相对于当前目录的路径

### 参数未生效

命令行参数优先级高于配置文件。如果参数未生效：
- 检查是否有命令行参数覆盖了配置
- 检查 YAML 语法是否正确
- 检查参数名称是否正确（使用 `-` 而非 `_`）

### method_args 格式错误

确保 `method_args` 是字典格式：

```yaml
# 正确
method_args:
  ss-steps: 25
  blend-ss-enabled: true

# 错误
method_args:
  - ss-steps: 25
  - blend-ss-enabled: true
```
