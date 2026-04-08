# Image P2P with Latent Blend (Mask-Based)

## 概述

`image_p2p_latent_blend` 是一个简化的编辑方法，结合：
1. **P2P attention injection**：细粒度的注意力控制
2. **Mask-based latent blending**：在重合体素上根据 mask 混合 SLAT 特征
3. **Soft mask support**：边界软混合（类似 VoxHammer）

## 方法名称说明

- **文件名**：`image_p2p_latent_blend.py`
- **方法名**：`image_p2p_latent_blend`
- **"Blend" vs "Replace"**：使用 "blend" 强调这是后处理混合，而非去噪中替换

## 与其他方法的区别

### vs Image P2P
- **Image P2P**：只使用 attention injection
- **P2P + Latent Blend**：attention injection + 后处理 latent 混合

### vs UniEdit
- **UniEdit**：RF inversion + 去噪中双路径混合（复杂）
- **P2P + Latent Blend**：后处理混合（简单）

### vs Hybrid (UniEdit + P2P)
- **Hybrid**：完整 UniEdit + P2P（最复杂，最强保留）
- **P2P + Latent Blend**：简化版本（更快，足够好）

## 核心原理

### 1. 后处理混合策略

与去噪中替换不同，这个方法采用后处理混合：

```python
# 1. 分别生成源和编辑的完整 3D
source_slat = generate(source_image)
edit_slat = generate(edit_image)

# 2. 找到重合体素
overlapping_voxels = find_overlap(source_coords, edit_coords)

# 3. 对每个重合体素混合特征
for voxel in overlapping_voxels:
    # 将 3D 坐标投影到 2D mask
    mask_value = project_and_sample_mask(voxel, spatial_mask)
    
    # 混合特征
    blended_feat = mask_value * edit_feat + (1 - mask_value) * source_feat
```

**优势**：
- 不需要 hook 采样器内部
- 实现简单，易于理解
- 已完整实现

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
- 类似 VoxHammer 的软混合（参考 `temp/edit_pipeline.py`）

### 3. 与 P2P Attention 的协同

P2P 和 latent blend 在不同阶段工作：

```python
# 阶段 1: 生成时 P2P 工作
source_slat = generate(source_image)  # ← P2P hook 拦截 attention
edit_slat = generate(edit_image)      # ← P2P hook 拦截 attention

# 阶段 2: 后处理混合
blended_slat = blend_features(source_slat, edit_slat, mask)
```

**两者互补**：
- P2P 控制生成时的特征级别细节
- Latent blend 控制最终的结构级别保留

## 使用方法

### 基本命令

```bash
python run_edit_experiment.py \
  --method image_p2p_latent_blend \
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

#### Latent Blending 参数

- `blend_ss_enabled`: 是否在 sparse structure 阶段混合，默认 `false`
  - 通常不需要，因为 SS 是坐标，难以混合
  
- `blend_slat_enabled`: 是否在 SLAT 阶段混合，默认 `true`
  - 这是主要功能，在 SLAT 特征层面混合

- `ss_blend_mode`: SS 阶段的 mask 模式，默认 `"hard"`
  - `"hard"`: 硬边界
  - `"soft"`: Gaussian blur 软边界

- `slat_blend_mode`: SLAT 阶段的 mask 模式，默认 `"soft"`
  - `"hard"`: 硬边界
  - `"soft"`: Gaussian blur 软边界（推荐）

- `ss_soft_kernel_size`: SS 阶段 Gaussian kernel 大小，默认 `3`
- `slat_soft_kernel_size`: SLAT 阶段 Gaussian kernel 大小，默认 `5`
  - 越大越平滑，但过渡区域越宽
  - 推荐：3-7

- `blend_strength`: 整体混合强度，默认 `1.0`
  - `1.0` = 完全混合
  - `0.5` = 半强度混合
  - `0.0` = 不混合（退化为纯编辑）

### 完整示例

```bash
# 基础版本（SLAT 阶段 soft mask）
python run_edit_experiment.py \
  --method image_p2p_latent_blend \
  --source-image assets/edit_example/images/2d_render.png \
  --edit-image assets/edit_example/images/2d_edit.png \
  --mask-image assets/edit_example/images/2d_mask.png \
  --case-name p2p_blend_test \
  --seed 1 \
  --preprocess \
  --extra-param blend_slat_enabled=true \
  --extra-param slat_blend_mode=soft \
  --extra-param slat_soft_kernel_size=5

# 消融实验：hard mask
python run_edit_experiment.py \
  --method image_p2p_latent_blend \
  --source-image assets/edit_example/images/2d_render.png \
  --edit-image assets/edit_example/images/2d_edit.png \
  --mask-image assets/edit_example/images/2d_mask.png \
  --case-name p2p_blend_hard \
  --seed 1 \
  --preprocess \
  --extra-param blend_slat_enabled=true \
  --extra-param slat_blend_mode=hard

# 消融实验：不同 kernel size
python run_edit_experiment.py \
  --method image_p2p_latent_blend \
  --source-image assets/edit_example/images/2d_render.png \
  --edit-image assets/edit_example/images/2d_edit.png \
  --mask-image assets/edit_example/images/2d_mask.png \
  --case-name p2p_blend_kernel7 \
  --seed 1 \
  --preprocess \
  --extra-param blend_slat_enabled=true \
  --extra-param slat_blend_mode=soft \
  --extra-param slat_soft_kernel_size=7
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

✅ **已完整实现**

核心功能：
- ✅ 找到重合体素
- ✅ 3D 坐标投影到 2D mask
- ✅ 根据 mask 值混合特征
- ✅ Hard/soft mask 支持
- ✅ 两阶段独立控制

实现文件：
- `editing/methods/image_p2p_latent_blend.py` - 完整实现
- `_blend_slat_features()` - 核心混合逻辑

## 技术细节

### 重合体素查找

使用 hash-based 快速查找：

```python
# 将 3D 坐标转换为 hash 字符串
src_hash = [f"{x}_{y}_{z}" for x, y, z in src_coords]
edit_hash = [f"{x}_{y}_{z}" for x, y, z in edit_coords]

# 找交集
overlap = set(src_hash) & set(edit_hash)
```

### 3D 到 2D 投影

简单的 top-down 投影：

```python
# 假设 z 是深度，投影 (x, y) 到 mask
u = int((x / resolution) * mask_width)
v = int((y / resolution) * mask_height)
mask_value = mask[v, u]
```

**注意**：这是简化的投影，实际应用可能需要考虑相机参数。

### 特征混合

```python
# mask_value = 0 -> 保留源
# mask_value = 1 -> 使用编辑
blend_weight = (1.0 - mask_value) * blend_strength
blended = blend_weight * source_feat + (1 - blend_weight) * edit_feat
```

## 未来改进

1. **改进投影方法**：使用实际相机参数而非简单 top-down
2. **3D mask 支持**：直接使用 3D mask GLB 而非 2D 投影
3. **自适应 soft mask**：根据内容自动调整 kernel size
4. **Sparse structure 混合**：实现坐标级别的混合（当前只混合 SLAT）

## 消融实验建议

### Hard vs Soft Mask

```bash
# Hard mask
--extra-param slat_blend_mode=hard

# Soft mask (kernel=3)
--extra-param slat_blend_mode=soft --extra-param slat_soft_kernel_size=3

# Soft mask (kernel=5)
--extra-param slat_blend_mode=soft --extra-param slat_soft_kernel_size=5

# Soft mask (kernel=7)
--extra-param slat_blend_mode=soft --extra-param slat_soft_kernel_size=7
```

### 混合强度

```bash
# 完全混合
--extra-param blend_strength=1.0

# 半强度
--extra-param blend_strength=0.5

# 弱混合
--extra-param blend_strength=0.3
```

### 阶段控制

```bash
# 只混合 SLAT（推荐）
--extra-param blend_ss_enabled=false --extra-param blend_slat_enabled=true

# 两阶段都混合
--extra-param blend_ss_enabled=true --extra-param blend_slat_enabled=true

# 不混合（退化为纯 P2P）
--extra-param blend_ss_enabled=false --extra-param blend_slat_enabled=false
```

## 参考

- P2P 原理：`editing/hooks/prompt_to_prompt.py`
- UniEdit 原理：`docs/OOM_AND_UNIEDIT_FIX.md`
- VoxHammer soft blending：类似的软混合思想
