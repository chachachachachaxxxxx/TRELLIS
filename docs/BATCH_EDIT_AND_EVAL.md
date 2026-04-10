# 批量运行编辑实验并评测

## 功能

直接从 Edit3D-Bench 数据集读取测试案例，运行编辑方法生成新结果，并自动评测。

所有结果直接保存到 `/cache/wangxinxing/data/temp/{method_name}_{config_name}_{timestamp}/`

## 基本用法

```bash
python run_batch_edit_and_eval.py \
  --gt-root /home/wangxinxing/code/Edit3Dpp/data \
  --method-name image_p2p_latent_blend \
  --config-name latent_only \
  --max-cases 1 \
  --method-args --ss-steps 25 --slat-steps 25
```

## 参数说明

### 必需参数

- `--gt-root`: Edit3D-Bench GT 数据根目录
- `--method-name`: 方法名称（如 `image_p2p_latent_blend`）
- `--config-name`: 配置名称，用于标识不同的参数配置（如 `latent_only`, `full_p2p`, `ss_only`）

### 数据过滤

- `--dataset`: 数据集过滤（如 `GSO`）
- `--object`: 物体名称过滤（如 `3D_Dollhouse_Happy_Brother`）
- `--prompt-id`: 提示ID过滤（1, 2, 或 3）
- `--max-cases`: 限制处理的案例数量

### 方法参数

- `--seed`: 随机种子（默认：1）
- `--method-args`: 传递给 `run_edit_experiment.py` 的参数（放在最后）

### 评测参数

- `--metrics`: 评测指标列表
- `--device`: 计算设备（默认：`cuda:0`）
- `--skip-render`: 跳过渲染步骤

## 使用示例

### 示例 1: 只启用 Latent Replace（禁用 P2P）

```bash
python run_batch_edit_and_eval.py \
  --gt-root /home/wangxinxing/code/Edit3Dpp/data \
  --method-name image_p2p_latent_blend \
  --config-name latent_only \
  --max-cases 1 \
  --seed 1 \
  --method-args \
    --ss-steps 25 \
    --slat-steps 25
```

### 示例 2: 只启用 SS P2P

```bash
python run_batch_edit_and_eval.py \
  --gt-root /home/wangxinxing/code/Edit3Dpp/data \
  --method-name image_p2p_latent_blend \
  --config-name ss_p2p_only \
  --max-cases 1 \
  --method-args \
    --ss-steps 25 \
    --slat-steps 25
```

### 示例 3: 完整 P2P（SS + SLAT）

```bash
python run_batch_edit_and_eval.py \
  --gt-root /home/wangxinxing/code/Edit3Dpp/data \
  --method-name image_p2p_latent_blend \
  --config-name full_p2p \
  --max-cases 1 \
  --method-args \
    --ss-steps 25 \
    --slat-steps 25
```

### 示例 4: 处理特定物体

```bash
python run_batch_edit_and_eval.py \
  --gt-root /home/wangxinxing/code/Edit3Dpp/data \
  --method-name image_p2p_latent_blend \
  --config-name test \
  --dataset GSO \
  --object 3D_Dollhouse_Happy_Brother \
  --prompt-id 1 \
  --method-args \
    --ss-steps 25 \
    --slat-steps 25
```

### 示例 5: 批量处理（前10个案例）

```bash
python run_batch_edit_and_eval.py \
  --gt-root /home/wangxinxing/code/Edit3Dpp/data \
  --method-name image_p2p_latent_blend \
  --config-name batch_test \
  --max-cases 10 \
  --method-args \
    --ss-steps 25 \
    --slat-steps 25
```

## 输出结构

### 评测数据

```
/cache/wangxinxing/data/temp/{method_name}_{config_name}_{timestamp}/
├── GSO/
│   └── {object_name}/
│       └── prompt_{1,2,3}/
│           ├── edit.glb
│           ├── images/
│           │   └── render_XXXX.png (16 views)
│           └── videos/
│               └── video_rgb.mp4
└── evaluation_output/
    ├── detailed_results.json
    └── summary.json
```

### 评测结果

```
outputs/results/{method_name}_{config_name}_{timestamp}/
├── evaluation_results.json
└── report.txt

outputs/results/{method_name}_{config_name}_summary_{timestamp}/
└── batch_summary.json  # 批量处理的汇总结果
```

## 工作流程

1. **读取 metadata** - 从 GT 数据集读取测试案例
2. **运行编辑** - 对每个案例运行编辑方法
3. **保存 GLB** - 将生成的 edit.glb 保存到 `/cache/wangxinxing/data/temp/`
4. **渲染图像** - 渲染 16 个视角的图像
5. **运行评测** - 计算评测指标
6. **保存结果** - 保存评测报告到 `outputs/results/`

## 消融实验示例

```bash
# 实验 1: 只启用 Latent Blend（基线）
python run_batch_edit_and_eval.py \
  --gt-root /home/wangxinxing/code/Edit3Dpp/data \
  --method-name image_p2p_latent_blend \
  --config-name ablation_latent_only \
  --max-cases 5 \
  --method-args --ss-steps 25 --slat-steps 25

# 实验 2: Latent Blend + SS P2P
python run_batch_edit_and_eval.py \
  --gt-root /home/wangxinxing/code/Edit3Dpp/data \
  --method-name image_p2p_latent_blend \
  --config-name ablation_with_ss_p2p \
  --max-cases 5 \
  --method-args --ss-steps 25 --slat-steps 25

# 实验 3: Latent Blend + SLAT P2P
python run_batch_edit_and_eval.py \
  --gt-root /home/wangxinxing/code/Edit3Dpp/data \
  --method-name image_p2p_latent_blend \
  --config-name ablation_with_slat_p2p \
  --max-cases 5 \
  --method-args --ss-steps 25 --slat-steps 25

# 实验 4: 完整方法（Latent Blend + SS P2P + SLAT P2P）
python run_batch_edit_and_eval.py \
  --gt-root /home/wangxinxing/code/Edit3Dpp/data \
  --method-name image_p2p_latent_blend \
  --config-name ablation_full \
  --max-cases 5 \
  --method-args --ss-steps 25 --slat-steps 25
```

## 注意事项

1. **config-name 很重要** - 用于区分不同的参数配置，会体现在输出路径中
2. **method-args 放在最后** - 所有传递给编辑方法的参数都放在 `--method-args` 之后
3. **自动清理临时文件** - 脚本会自动清理 `outputs/{method_name}/temp_*` 临时目录
4. **结果直接保存到 /cache** - 不需要手动复制，所有 edit.glb 直接保存到最终位置

## 与其他脚本的对比

| 脚本 | 用途 | 输入 | 输出 |
|------|------|------|------|
| `run_edit_experiment.py` | 运行单个编辑实验 | 手动指定图像 | `outputs/{method}/{case}/` |
| `run_full_evaluation.py` | 评测已有结果 | 已有的 GLB | 评测报告 |
| `run_batch_edit_and_eval.py` | **批量运行+评测** | **GT 数据集** | **/cache + 评测报告** |

**推荐使用 `run_batch_edit_and_eval.py`** 进行完整的实验和评测流程！
