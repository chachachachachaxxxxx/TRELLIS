# Edit3D-Bench 评测系统

## 核心脚本

### `run_batch_edit_and_eval.py` - 一键式评测脚本

**功能**: 从 Edit3D-Bench 数据集读取测试案例 → 运行编辑方法 → 渲染图像 → 评测

**输出**:
- `edit.glb` 保存在: `/cache/wangxinxing/data/temp/{method_name}_{config_name}_{timestamp}/`
- 评测结果保存在: `outputs/results/{method_name}_{config_name}_{timestamp}/`

## 快速开始

### 基本用法

```bash
python run_batch_edit_and_eval.py \
  --gt-root /home/wangxinxing/code/Edit3Dpp/data \
  --method-name image_p2p_latent_blend \
  --config-name your_config \
  --max-cases 1 \
  --method-args <方法参数>
```

### 使用测试脚本

```bash
# 运行测试
./test_batch_dollhouse_no_p2p.sh

# 查看结果
cat outputs/results/image_p2p_latent_blend_dollhouse_no_p2p_*/report.txt
```

## 参数说明

### 必需参数

- `--gt-root`: Edit3D-Bench GT 数据根目录
- `--method-name`: 方法名称
- `--config-name`: 配置名称（用于标识不同参数配置）

### 数据过滤

- `--dataset`: 数据集过滤（如 `GSO`）
- `--object`: 物体名称过滤
- `--prompt-id`: 提示ID过滤（1, 2, 3）
- `--max-cases`: 限制案例数量

### 方法参数

- `--seed`: 随机种子（默认：1）
- `--device`: 计算设备（默认：`cuda:0`）
- `--method-args`: 传递给编辑方法的参数（放在最后）

### 其他选项

- `--skip-render`: 跳过渲染步骤
- `--metrics`: 评测指标列表

## 使用示例

### 示例 1: 单个案例测试

```bash
python run_batch_edit_and_eval.py \
  --gt-root /home/wangxinxing/code/Edit3Dpp/data \
  --method-name image_p2p_latent_blend \
  --config-name test \
  --dataset GSO \
  --object 3D_Dollhouse_Happy_Brother \
  --prompt-id 1 \
  --seed 1 \
  --method-args \
    --ss-steps 25 \
    --slat-steps 25
```

### 示例 2: 批量处理（前10个案例）

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

### 示例 3: 禁用 P2P（只用 Latent Blend）

```bash
python run_batch_edit_and_eval.py \
  --gt-root /home/wangxinxing/code/Edit3Dpp/data \
  --method-name image_p2p_latent_blend \
  --config-name latent_only \
  --max-cases 1 \
  --method-args \
    --ss-steps 25 \
    --slat-steps 25 \
    --inject-stages "" \
    --blend-ss-enabled \
    --blend-slat-enabled
```

## 输出结构

```
/cache/wangxinxing/data/temp/{method_name}_{config_name}_{timestamp}/
├── GSO/
│   └── {object_name}/
│       └── prompt_{1,2,3}/
│           ├── edit.glb                    # 编辑结果
│           ├── images/
│           │   └── render_XXXX.png        # 16个视角
│           └── videos/
│               └── video_rgb.mp4
└── evaluation_output/
    ├── detailed_results.json
    └── summary.json

outputs/results/{method_name}_{config_name}_{timestamp}/
├── evaluation_results.json                 # 完整结果
└── report.txt                              # 可读报告
```

## 工作流程

1. **读取数据集** - 从 Edit3D-Bench GT 数据读取测试案例
2. **运行编辑** - 使用指定参数运行编辑方法
3. **保存 GLB** - edit.glb 直接保存到 `/cache/wangxinxing/data/temp/`
4. **渲染图像** - 自动渲染 16 个视角的图像
5. **运行评测** - 计算所有评测指标
6. **保存结果** - 评测报告保存到 `outputs/results/`

## 常见方法参数

### image_p2p_latent_blend 方法

```bash
# 基础参数
--ss-steps 25                    # SS 采样步数
--slat-steps 25                  # SLAT 采样步数

# P2P 控制
--inject-stages ""               # 禁用 P2P
--inject-stages sparse_structure # 只启用 SS P2P
--inject-stages slat             # 只启用 SLAT P2P
--inject-stages sparse_structure slat  # 启用全部 P2P

# Latent Blend 控制
--blend-ss-enabled               # 启用 SS Latent Blend
--blend-slat-enabled             # 启用 SLAT Latent Blend
--ss-blend-mode soft             # SS blend 模式（hard/soft）
--slat-blend-mode soft           # SLAT blend 模式（hard/soft）
--ss-soft-kernel-size 5          # SS soft kernel 大小
--slat-soft-kernel-size 7        # SLAT soft kernel 大小
```

## 相关文档

- `docs/BATCH_EDIT_AND_EVAL.md` - 详细使用指南
- `test_batch_dollhouse_no_p2p.sh` - 测试脚本示例
