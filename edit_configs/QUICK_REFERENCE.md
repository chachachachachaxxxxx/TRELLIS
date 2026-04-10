# 配置快速参考

## 快速开始

### 单个样例测试
```bash
python run_edit_experiment.py --config edit_configs/single/image_p2p_example.yaml
```

### 批处理评测
```bash
# 快速测试（5个案例，12步）
python run_batch_edit_and_eval.py --config edit_configs/batch/quick_test.yaml

# GSO 数据集（禁用 P2P）
python run_batch_edit_and_eval.py --config edit_configs/batch/gso_no_p2p.yaml

# Dollhouse 物体（禁用 P2P）
python run_batch_edit_and_eval.py --config edit_configs/batch/dollhouse_no_p2p.yaml

# 完整 Benchmark（启用 P2P）
python run_batch_edit_and_eval.py --config edit_configs/batch/full_benchmark.yaml
```

## 配置文件对比

| 配置文件 | 用途 | 数据集 | P2P | 步数 | 案例数 |
|---------|------|--------|-----|------|--------|
| `quick_test.yaml` | 快速测试 | 全部 | ✗ | 12 | 5 |
| `gso_no_p2p.yaml` | GSO 评测 | GSO | ✗ | 25 | 全部 |
| `dollhouse_no_p2p.yaml` | Dollhouse 评测 | Dollhouse | ✗ | 25 | 3 |
| `full_benchmark.yaml` | 完整评测 | 全部 | ✓ | 25 | 全部 |

## 常用参数覆盖

### 限制案例数量
```bash
python run_batch_edit_and_eval.py \
  --config edit_configs/batch/gso_no_p2p.yaml \
  --max-cases 10
```

### 指定数据集
```bash
python run_batch_edit_and_eval.py \
  --config edit_configs/batch/full_benchmark.yaml \
  --dataset GSO
```

### 指定物体
```bash
python run_batch_edit_and_eval.py \
  --config edit_configs/batch/gso_no_p2p.yaml \
  --object 3D_Dollhouse_Happy_Brother
```

### 指定 Prompt ID
```bash
python run_batch_edit_and_eval.py \
  --config edit_configs/batch/dollhouse_no_p2p.yaml \
  --prompt-id 1
```

### 更改设备
```bash
python run_batch_edit_and_eval.py \
  --config edit_configs/batch/gso_no_p2p.yaml \
  --device cuda:1
```

### 跳过渲染
```bash
python run_batch_edit_and_eval.py \
  --config edit_configs/batch/gso_no_p2p.yaml \
  --skip-render
```

## 方法参数说明

### P2P 控制
- `inject-stages`: P2P 注入阶段
  - `""`: 禁用 P2P
  - `"ss"`: 仅 SS 阶段
  - `"slat"`: 仅 SLAT 阶段
  - `"ss,slat"`: 两阶段都启用
- `p2p-start-step`: P2P 起始步数比例（0.0-1.0）
- `p2p-end-step`: P2P 结束步数比例（0.0-1.0）

### Latent Blend 控制
- `blend-ss-enabled`: 是否启用 SS 阶段 blend
- `blend-slat-enabled`: 是否启用 SLAT 阶段 blend
- `blend-start-step`: Blend 起始步数比例（0.0-1.0）
- `blend-end-step`: Blend 结束步数比例（0.0-1.0）

### Blend 模式
- `ss-blend-mode`: SS 阶段 blend 模式（`hard` 或 `soft`）
- `slat-blend-mode`: SLAT 阶段 blend 模式（`hard` 或 `soft`）
- `ss-soft-kernel-size`: SS soft blend 核大小（奇数）
- `slat-soft-kernel-size`: SLAT soft blend 核大小（奇数）

### 采样步数
- `ss-steps`: Sparse Structure 采样步数（默认 25）
- `slat-steps`: SLAT 采样步数（默认 25）

## 预设配置组合

### 1. 仅 Latent Blend（推荐，速度快）
```yaml
method_args:
  inject-stages: ""
  blend-ss-enabled: true
  blend-slat-enabled: true
  ss-blend-mode: soft
  slat-blend-mode: soft
```

### 2. 仅 P2P（原始方法）
```yaml
method_args:
  inject-stages: "ss,slat"
  blend-ss-enabled: false
  blend-slat-enabled: false
```

### 3. P2P + Latent Blend（完整模式）
```yaml
method_args:
  inject-stages: "ss,slat"
  blend-ss-enabled: true
  blend-slat-enabled: true
```

### 4. 快速测试（减少步数）
```yaml
method_args:
  ss-steps: 12
  slat-steps: 12
  inject-stages: ""
  blend-ss-enabled: true
  blend-slat-enabled: true
```

## 输出位置

### 批处理输出
- **编辑结果**: `/cache/wangxinxing/data/temp/{method_name}_{config_name}/`
- **评测结果**: `outputs/results/{method_name}_{config_name}_{timestamp}/`

### 单个样例输出
- **编辑结果**: `outputs/{method_name}/{case_name}/`

## 常见问题

### Q: 如何创建新配置？
A: 复制现有配置文件，修改参数即可。建议：
- 单个样例：复制 `single/image_p2p_example.yaml`
- 批处理：复制 `batch/quick_test.yaml` 或 `batch/gso_no_p2p.yaml`

### Q: 配置文件和命令行参数冲突怎么办？
A: 命令行参数优先级更高，会覆盖配置文件中的值。

### Q: 如何查看所有可用参数？
A: 运行 `python run_edit_experiment.py --help` 或 `python run_batch_edit_and_eval.py --help`

### Q: 如何复用预处理的 assets？
A: 设置 `assets_root: /cache/wangxinxing/data/temp/renders`，跳过预处理步骤。
