# UniEdit Euler 方法说明

## 概述

`image_uniedit_euler` 是 UniEdit 的简化版本，使用一阶 Euler 积分 + predictor-corrector 策略，而不是完整的 RF-Solver 二阶积分。

## 核心机制

### 1. 一阶 Euler 积分

不使用二阶 Taylor 展开，而是简单的一阶更新：

```
x_{t+h} = x_t + h·f(x_t, t)
```

### 2. Predictor-Corrector 策略

为了提高精度，使用 predictor-corrector 方法：

1. **Predictor**: 在当前点 `t` 预测速度 `v_t`
2. **Step**: 使用 `v_t` 走一步到 `t+h`
3. **Corrector**: 在新点 `t+h` 重新预测速度 `v_{t+h}`
4. **Update**: 使用修正后的速度 `v_{t+h}` 进行最终更新

这比纯一阶 Euler 更准确，但比 RF-Solver 的二阶方法更简单。

### 3. Delayed Inversion/Editing

使用 `alpha` 参数控制延迟反演和延迟编辑：

- **Inversion**: 只反演前 `alpha` 比例的步数（例如 `alpha=0.5` 只反演前一半）
- **Editing**: 只编辑后 `alpha` 比例的步数（例如 `alpha=0.5` 只编辑后一半）

这样可以在中间 junction 点开始编辑，而不是从完全的噪声端开始。

### 4. Source/Target Velocity Fusion

在编辑阶段，同时计算 source 和 target 两个条件下的速度：

```python
pred_src = model(sample, t, source_cond)
pred_tgt = model(sample, t, target_cond)
guidance = pred_tgt - pred_src
```

然后根据速度差异生成编辑区域图：

```python
save_map = compute_uniedit_map(guidance)  # [0, 1] 范围
```

最终速度是融合后的结果：

```python
fused = pred_tgt * save_map + pred_src * (1.0 - save_map)
pred = fused + guidance * ((1.0 + save_map) * omega)
```

- `save_map` 高的区域：更偏向 target
- `save_map` 低的区域：更保留 source
- `omega` 控制额外的编辑强度

## 参数说明

### 核心参数

- `ss_omega` (default: 1.0): Stage 1 (sparse structure) 的编辑强度
- `slat_omega` (default: 1.0): Stage 2 (SLAT features) 的编辑强度
- `ss_alpha` (default: 0.5): Stage 1 的延迟比例（0.5 = 反演前半，编辑后半）
- `slat_alpha` (default: 0.5): Stage 2 的延迟比例
- `zero_init` (default: false): 反演第一步是否使用零速度

### 其他参数

- `cfg_interval` (default: (0.5, 1.0)): CFG 应用的时间区间
- `decode_modes` (default: ["gaussian", "mesh"]): 解码模式
- `skip_render` (default: true): 跳过渲染
- `skip_glb` (default: false): 跳过 GLB 导出
- `skip_ply` (default: false): 跳过 PLY 导出

## 使用方法

### 基本用法

```bash
python run_edit_experiment.py \
  --method image_uniedit_euler \
  --source-image path/to/source.png \
  --edit-image path/to/edit.png \
  --mask-glb path/to/mask.glb \
  --source-model path/to/source/assets \
  --case-name my_edit \
  --seed 1 \
  --preprocess
```

### 调整编辑强度

```bash
# 更激进的编辑
python run_edit_experiment.py \
  --method image_uniedit_euler \
  ... \
  --extra-params ss_omega=2.0 \
  --extra-params slat_omega=2.0
```

### 调整延迟比例

```bash
# 更早开始编辑（反演更少）
python run_edit_experiment.py \
  --method image_uniedit_euler \
  ... \
  --extra-params ss_alpha=0.3 \
  --extra-params slat_alpha=0.3
```

### 使用零初始化

```bash
# 反演第一步使用零速度
python run_edit_experiment.py \
  --method image_uniedit_euler \
  ... \
  --extra-params zero_init=true
```

## 与其他方法的对比

### vs. `image_uniedit_rf_inversion`

- **RF Inversion**: 使用完整的 RF-Solver 二阶积分
  - 优点：更准确的轨迹近似
  - 缺点：计算量更大，需要两次模型调用（midpoint correction）

- **Euler**: 使用一阶 Euler + predictor-corrector
  - 优点：更简单，计算量稍小
  - 缺点：精度略低于二阶方法

### vs. `image_prompt_to_prompt_rf_inversion`

- **P2P**: 使用 Prompt-to-Prompt 的 attention injection
  - 机制：在 cross-attention 层注入 source attention map
  - 适用：需要精确控制 attention 的场景

- **UniEdit Euler**: 使用 source/target velocity fusion
  - 机制：在 latent space 融合 source 和 target 速度
  - 适用：需要区域感知编辑的场景

## 输出结构

```
outputs/image_uniedit_euler/<case_name>/
├── mesh.glb                    # 导出的 GLB 文件
├── mesh.ply                    # 导出的 PLY 文件
├── uniedit_euler_metadata.json # 元数据
└── ...
```

## 实现细节

### 文件结构

- `editing/inversion/uniedit_euler_sampler.py`: UniEdit Euler sampler 实现
- `editing/methods/image_uniedit_euler.py`: 方法类实现
- `editing/methods/registry.py`: 方法注册

### 关键函数

- `UniEditEulerSampler.invert()`: 延迟反演
- `UniEditEulerSampler.edit()`: 延迟编辑
- `invert_once_predictor_corrector()`: 单步反演（predictor-corrector）
- `edit_once_predictor_corrector()`: 单步编辑（predictor-corrector + UniEdit fusion）

## 注意事项

1. **显存占用**: 每步需要计算两次模型（source + target），显存占用较大
2. **步数选择**: 建议使用与训练时相同的步数（通常 12-20 步）
3. **Alpha 选择**: `alpha=0.5` 是一个好的起点，可以根据需要调整
4. **Omega 选择**: `omega=1.0` 是标准设置，增大会使编辑更激进

## 测试

运行测试脚本：

```bash
./test_uniedit_euler.sh
```

或手动测试：

```bash
python run_edit_experiment.py \
  --method image_uniedit_euler \
  --source-image assets/edit_example/images/2d_render.png \
  --edit-image assets/edit_example/images/2d_edit.png \
  --mask-glb assets/edit_example/mask.glb \
  --source-model assets/edit_example \
  --case-name test \
  --seed 1 \
  --preprocess
```

## 参考

- UniEdit-Flow 原理文档（项目内部）
- VoxHammer RF-Solver 实现
- Prompt-to-Prompt 论文
