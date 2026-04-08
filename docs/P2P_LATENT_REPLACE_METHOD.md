# Image P2P with Latent Replace by Mask

## 概述

`image_p2p_latent_replace` 是一个简化的编辑方法，结合：
1. **P2P attention injection**：细粒度的注意力控制
2. **Mask-based latent replacement**：根据 mask 直接替换 latent（比 UniEdit 简单）
3. **Soft mask support**：边界软混合（类似 VoxHammer）

## 与其他方法的区别

### vs Image P2P
- **Image P2P**：只使用 attention injection
- **P2P + Latent Replace**：attention injection + 直接 latent 替换

### vs UniEdit
- **UniEdit**：RF inversion + 双路径去噪 + latent 混合（复杂）
- **P2P + Latent Replace**：单路径去噪 + mask-based 替换（简单）

### vs Hybrid (UniEdit + P2P)
- **Hybrid**：完整 UniEdit + P2P（最复杂，最强保留）
- **P2P + Latent Replace**：简化版本（更快，足够好）

## 核心原理

### 1. Latent Replace by Mask

在去噪的每一步：

```python
for t in timesteps:
    # 1. 对编辑图像去噪一步
    x_edit = denoise_step(x_edit, t, edit_cond)
    
    # 2. 对源图像去噪一步（同步）
    x_source = denoise_step(x_source, t, source_cond)
    
    # 3. 根据 mask 混合 latent
    mask_3d = resize_mask_to_latent(spatial_mask, x_edit.shape)
    x_mixed = mask_3d * x_edit + (1 - mask_3d) * x_source
    
    # 4. 继续用混合后的 latent
    x_edit = x_mixed
```

**关键点**：
- 不需要 RF inversion（比 UniEdit 简单）
- 直接在 latent 空间混合（不是在噪声预测空间）
- Mask 控制哪些区域保留源，哪些区域使用编辑

### 2. Soft Mask（可选）

对于硬边界的 mask，可以应用 Gaussian blur 实现软混合：

```python
# Hard mask: 0 or 1
mask_hard = [[0, 0, 1, 1],
             [0, 0, 1, 1]]

# Soft mask after Gaussian blur: smooth transition
mask_soft = [[0.0, 0.2, 0.8, 1.0],
             [0.0, 0.2, 0.8, 1.0]]
```

**效果**：
- 边界处平滑过渡
- 避免明显的接缝
- 类似 VoxHammer 的软混合

### 3. 与 P2P Attention 的协同

```python
# 在每个去噪步骤
for t in timesteps:
    # P2P 在 model() 内部工作
    x_edit = denoise_step(x_edit, t, edit_cond)  # ← P2P hook 拦截 attention
    x_source = denoise_step(x_source, t, source_cond)  # ← P2P hook 拦截 attention
    
    # Latent replace 在外部工作
    x_mixed = mask * x_edit + (1 - mask) * x_source
    x_edit = x_mixed
```

**两者互补**：
- P2P 控制特征级别的细节
- Latent replace 控制结构级别的保留

## 使用方法

### 基本命令

```bash
python run_edit_experiment.py \
  --method image_p2p_latent_replace \
  --source-image <source.png> \
  --edit-image <edit.png> \
  --mask-image <mask.png> \
  --case-name <case_name> \
  --seed 1 \
  --preprocess
```

### 参数配置

#### P2P 参数（同 image_prompt_to_prompt）

- `inject_stages`: 注入阶段，默认 `["sparse_structure", "slat"]`
- `*_t_start`, `*_t_end`: 时间步范围
- `*_strength`: 注意力混合强度

#### Latent Replacement 参数

- `enable_latent_replace`: 是否启用 latent 替换，默认 `true`
- `latent_replace_steps`: 应用替换的步数比例，默认 `0.5`（前 50% 步骤）
  - `1.0` = 所有步骤都替换（最强保留）
  - `0.5` = 前半段替换（平衡）
  - `0.0` = 不替换（退化为纯 P2P）

#### Soft Mask 参数

- `soft_mask_enabled`: 是否启用软 mask，默认 `false`
- `soft_mask_kernel_size`: Gaussian kernel 大小，默认 `3`
  - 越大越平滑，但过渡区域越宽
  - 推荐：3-7

### 完整示例

```bash
# 基础版本（硬 mask）
python run_edit_experiment.py \
  --method image_p2p_latent_replace \
  --source-image assets/edit_example/images/2d_render.png \
  --edit-image assets/edit_example/images/2d_edit.png \
  --mask-image assets/edit_example/images/2d_mask.png \
  --case-name p2p_replace_test \
  --seed 1 \
  --preprocess \
  --extra-param enable_latent_replace=true \
  --extra-param latent_replace_steps=0.5

# 软 mask 版本（平滑边界）
python run_edit_experiment.py \
  --method image_p2p_latent_replace \
  --source-image assets/edit_example/images/2d_render.png \
  --edit-image assets/edit_example/images/2d_edit.png \
  --mask-image assets/edit_example/images/2d_mask.png \
  --case-name p2p_replace_soft \
  --seed 1 \
  --preprocess \
  --extra-param enable_latent_replace=true \
  --extra-param latent_replace_steps=0.8 \
  --extra-param soft_mask_enabled=true \
  --extra-param soft_mask_kernel_size=5
```

## 输入要求

1. **Images**：
   - `source_image`: 源图像（必需）
   - `edit_image`: 编辑图像（必需）
   - `mask_image`: 2D mask 图像（必需）
     - 白色 (255) = 编辑区域
     - 黑色 (0) = 保留区域

2. **不需要**：
   - ❌ Source assets（不需要预生成的 3D 模型）
   - ❌ 3D mask GLB（只需要 2D mask）
   - ❌ RF inversion（不需要反演）

## 优势

1. **简单**：不需要 RF inversion，不需要预生成资产
2. **快速**：单路径去噪，比 UniEdit 快
3. **有效**：结合 P2P 和 latent replace，效果好
4. **灵活**：支持 soft mask，边界平滑

## 适用场景

- 不需要极致结构保留的编辑任务
- 快速原型和实验
- 边界需要平滑过渡的场景
- 没有预生成 3D 资产的情况

## 参数调优

### 最强保留

```bash
--extra-param latent_replace_steps=1.0 \
--extra-param sparse_structure_strength=1.0 \
--extra-param slat_strength=1.0
```

### 平衡编辑

```bash
--extra-param latent_replace_steps=0.5 \
--extra-param sparse_structure_strength=1.0 \
--extra-param slat_strength=0.8
```

### 最大变化

```bash
--extra-param latent_replace_steps=0.0 \
--extra-param sparse_structure_strength=0.5 \
--extra-param slat_strength=0.5
```

### 平滑边界

```bash
--extra-param soft_mask_enabled=true \
--extra-param soft_mask_kernel_size=5 \
--extra-param latent_replace_steps=0.8
```

## 实现状态

⚠️ **当前状态**：框架已实现，核心 latent replacement 逻辑需要完善

需要实现的部分：
1. Hook 到采样器的去噪循环
2. 在每步同步对源和编辑图像去噪
3. 根据 mask 混合 latent
4. 处理 sparse structure 和 SLAT 的不同空间结构

## 技术挑战

1. **空间对齐**：2D mask 需要映射到 3D latent 空间
2. **Sparse 结构**：稀疏坐标的 mask 应用
3. **采样器 hook**：需要拦截去噪循环

## 未来改进

1. 实现完整的 latent replacement 逻辑
2. 优化 mask 到 latent 的映射
3. 支持 3D mask（从 GLB）
4. 自适应 soft mask（根据内容自动调整）

## 参考

- P2P 原理：`editing/hooks/prompt_to_prompt.py`
- UniEdit 原理：`docs/OOM_AND_UNIEDIT_FIX.md`
- VoxHammer soft blending：类似的软混合思想
