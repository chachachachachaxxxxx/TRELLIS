# 支持方法配置的评测脚本

## 两种使用模式

### 模式 1: 评测已有结果

适用于已经运行过实验，想要评测现有结果的情况。

```bash
python run_eval_with_config.py \
  --gt-root /home/wangxinxing/code/Edit3Dpp/data \
  --method-name image_p2p_latent_blend \
  --case-name p2p_blend_no_slat_p2p \
  --dataset GSO \
  --object-name 3D_Dollhouse_Happy_Brother \
  --prompt-id 1
```

### 模式 2: 运行新实验并评测

适用于需要先运行编辑实验，然后立即评测的情况。

```bash
python run_eval_with_config.py \
  --gt-root /home/wangxinxing/code/Edit3Dpp/data \
  --method-name image_p2p_latent_blend \
  --run-experiment \
  --source-image /path/to/source.png \
  --edit-image /path/to/edit.png \
  --mask-image /path/to/mask.png \
  --mask-glb /path/to/mask.glb \
  --dataset GSO \
  --object-name 3D_Dollhouse_Happy_Brother \
  --prompt-id 1 \
  --new-case-name my_experiment \
  --seed 1 \
  --extra-args --ss-steps 25 --slat-steps 25
```

## 参数说明

### 必需参数

- `--gt-root`: Edit3D-Bench Ground Truth 数据根目录
- `--method-name`: 方法名称（如 `image_p2p_latent_blend`）
- `--dataset`: 数据集名称（如 `GSO`）
- `--object-name`: 物体名称（如 `3D_Dollhouse_Happy_Brother`）
- `--prompt-id`: 提示ID（1, 2, 或 3）

### 模式选择（二选一）

- `--case-name`: 使用已有案例名称（模式1）
- `--run-experiment`: 运行新实验（模式2）

### 实验参数（模式2需要）

- `--source-image`: 源图像路径
- `--edit-image`: 编辑图像路径
- `--mask-image`: 掩码图像路径
- `--mask-glb`: 3D掩码GLB路径（可选）
- `--seed`: 随机种子（默认：1）
- `--new-case-name`: 新实验的案例名称（可选，默认自动生成）
- `--extra-args`: 传递给 `run_edit_experiment.py` 的额外参数

### 评测参数

- `--metrics`: 评测指标列表（默认：所有指标）
- `--device`: 计算设备（默认：`cuda:0`）
- `--batch-size`: 批处理大小（默认：32）
- `--skip-render`: 跳过渲染步骤

## 使用示例

### 示例 1: 评测已有的 "只启用 latent replace" 实验

```bash
python run_eval_with_config.py \
  --gt-root /home/wangxinxing/code/Edit3Dpp/data \
  --method-name image_p2p_latent_blend \
  --case-name p2p_blend_no_slat_p2p \
  --dataset GSO \
  --object-name 3D_Dollhouse_Happy_Brother \
  --prompt-id 1
```

### 示例 2: 运行新实验（禁用 P2P）并评测

```bash
python run_eval_with_config.py \
  --gt-root /home/wangxinxing/code/Edit3Dpp/data \
  --method-name image_p2p_latent_blend \
  --run-experiment \
  --source-image /home/wangxinxing/code/Edit3Dpp/data/GSO/3D_Dollhouse_Happy_Brother/prompt_1/2d_render.png \
  --edit-image /home/wangxinxing/code/Edit3Dpp/data/GSO/3D_Dollhouse_Happy_Brother/prompt_1/2d_edit.png \
  --mask-image /home/wangxinxing/code/Edit3Dpp/data/GSO/3D_Dollhouse_Happy_Brother/prompt_1/2d_mask.png \
  --mask-glb /home/wangxinxing/code/Edit3Dpp/data/GSO/3D_Dollhouse_Happy_Brother/prompt_1/3d_edit_region.glb \
  --dataset GSO \
  --object-name 3D_Dollhouse_Happy_Brother \
  --prompt-id 1 \
  --new-case-name latent_only_test \
  --seed 1
```

### 示例 3: 使用自定义参数运行实验

```bash
python run_eval_with_config.py \
  --gt-root /home/wangxinxing/code/Edit3Dpp/data \
  --method-name image_p2p_latent_blend \
  --run-experiment \
  --source-image /path/to/source.png \
  --edit-image /path/to/edit.png \
  --mask-image /path/to/mask.png \
  --mask-glb /path/to/mask.glb \
  --dataset GSO \
  --object-name 3D_Dollhouse_Happy_Brother \
  --prompt-id 1 \
  --new-case-name custom_config \
  --seed 42 \
  --extra-args \
    --ss-steps 30 \
    --slat-steps 30 \
    --sparse-structure-t-start 1.0 \
    --sparse-structure-t-end 0.2
```

## 输出结构

```
/cache/wangxinxing/data/temp/{method_name}_{case_name}_{timestamp}/
├── GSO/
│   └── {object_name}/
│       └── prompt_{prompt_id}/
│           ├── edit.glb
│           ├── images/
│           │   └── render_XXXX.png (16 views)
│           └── videos/
│               └── video_rgb.mp4
└── evaluation_output/
    ├── detailed_results.json
    └── summary.json

outputs/results/{method_name}_{case_name}_{timestamp}/
├── evaluation_results.json
├── report.txt
├── detailed_results.json
└── summary.json
```

## 与原脚本的区别

### `run_full_evaluation.py` (原脚本)
- 简单模式，只需要方法名
- 自动查找所有匹配的案例
- 适合批量评测

### `run_eval_with_config.py` (新脚本)
- 支持方法配置参数
- 可以运行新实验
- 可以传递额外参数给编辑方法
- 适合单个案例的精确控制

## 常见用法

### 快速评测已有结果

```bash
python run_eval_with_config.py \
  --gt-root /home/wangxinxing/code/Edit3Dpp/data \
  --method-name image_p2p_latent_blend \
  --case-name p2p_blend_no_slat_p2p \
  --dataset GSO \
  --object-name 3D_Dollhouse_Happy_Brother \
  --prompt-id 1
```

### 运行消融实验

```bash
# 实验 1: 只启用 SS P2P
python run_eval_with_config.py \
  --gt-root /home/wangxinxing/code/Edit3Dpp/data \
  --method-name image_p2p_latent_blend \
  --run-experiment \
  --source-image ... \
  --edit-image ... \
  --mask-image ... \
  --dataset GSO \
  --object-name 3D_Dollhouse_Happy_Brother \
  --prompt-id 1 \
  --new-case-name ablation_ss_only \
  --extra-args --inject-stages sparse_structure

# 实验 2: 只启用 SLAT P2P
python run_eval_with_config.py \
  --gt-root /home/wangxinxing/code/Edit3Dpp/data \
  --method-name image_p2p_latent_blend \
  --run-experiment \
  --source-image ... \
  --edit-image ... \
  --mask-image ... \
  --dataset GSO \
  --object-name 3D_Dollhouse_Happy_Brother \
  --prompt-id 1 \
  --new-case-name ablation_slat_only \
  --extra-args --inject-stages slat

# 实验 3: 禁用所有 P2P
python run_eval_with_config.py \
  --gt-root /home/wangxinxing/code/Edit3Dpp/data \
  --method-name image_p2p_latent_blend \
  --run-experiment \
  --source-image ... \
  --edit-image ... \
  --mask-image ... \
  --dataset GSO \
  --object-name 3D_Dollhouse_Happy_Brother \
  --prompt-id 1 \
  --new-case-name ablation_no_p2p \
  --extra-args --inject-stages ""
```
