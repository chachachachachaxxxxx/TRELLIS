# Image P2P Latent Blend 参数详解

## 参数分类

该方法有三类参数：
1. **P2P Attention 参数** - 控制生成时的注意力注入
2. **Latent Blend 参数** - 控制后处理的特征混合
3. **Mask 参数** - 控制 mask 的类型和平滑度

---

## 1. P2P Attention 参数

这些参数控制在**生成过程中**的注意力注入。

### Sparse Structure 阶段

```bash
--extra-param sparse_structure_strength=1.0  # 默认 1.0
--extra-param sparse_structure_t_start=1.0   # 默认 1.0
--extra-param sparse_structure_t_end=0.3     # 默认 0.3
```

**作用**：
- `strength`: 控制源注意力的混合程度
  - `1.0` = 完全使用源注意力（最强保留）
  - `0.5` = 源和编辑注意力各占一半
  - `0.0` = 完全使用编辑注意力（无 P2P）
- `t_start/t_end`: 控制在哪些时间步应用 P2P
  - 时间步归一化到 [0, 1]
  - 只在 `[t_end, t_start]` 区间内应用

**工作位置**：在 cross-attention 计算中

**公式**：
```python
# 对于保留区域的 token
if strength >= 1.0:
    attn_mixed = attn_source
else:
    attn_mixed = strength * attn_source + (1 - strength) * attn_edit
```

### SLAT 阶段

```bash
--extra-param slat_strength=1.0     # 默认 1.0
--extra-param slat_t_start=0.8      # 默认 0.8
--extra-param slat_t_end=0.0        # 默认 0.0
```

**作用**：同上，但作用在 SLAT 阶段

---

## 2. Latent Blend 参数

这些参数控制在**生成完成后**的特征混合。

### 整体控制

```bash
--extra-param blend_strength=1.0  # 默认 1.0
```

**作用**：全局缩放因子，控制 mask 的影响程度

**工作位置**：在后处理混合 SLAT 特征时

**公式**：
```python
# 对于每个重合体素
mask_value = mask[voxel_position]  # 从 mask 读取（0-1）
blend_weight = (1.0 - mask_value) * blend_strength  # 源特征权重
blended = blend_weight * source_feat + (1 - blend_weight) * edit_feat
```

**示例**（假设 mask_value = 0.2，接近黑色/保留区域）：
- `blend_strength = 1.0`: `blend_weight = 0.8` → 80% 源 + 20% 编辑
- `blend_strength = 0.5`: `blend_weight = 0.4` → 40% 源 + 60% 编辑
- `blend_strength = 0.0`: `blend_weight = 0.0` → 0% 源 + 100% 编辑

**用途**：
- 消融实验：测试 mask 的重要性
- 微调保留程度
- 调试：快速看到无混合效果

### 阶段控制

```bash
--extra-param blend_ss_enabled=false   # 默认 false
--extra-param blend_slat_enabled=true  # 默认 true
```

**作用**：
- `blend_ss_enabled`: 是否在 Sparse Structure 阶段混合
  - 通常设为 `false`，因为 SS 是坐标，难以混合
- `blend_slat_enabled`: 是否在 SLAT 阶段混合
  - 这是主要功能，通常设为 `true`

---

## 3. Mask 参数

控制 mask 的类型和平滑度。

### Sparse Structure 阶段

```bash
--extra-param ss_blend_mode=hard        # 默认 "hard"
--extra-param ss_soft_kernel_size=3     # 默认 3
```

### SLAT 阶段

```bash
--extra-param slat_blend_mode=soft      # 默认 "soft"（推荐）
--extra-param slat_soft_kernel_size=5   # 默认 5
```

**作用**：
- `blend_mode`:
  - `"hard"`: 硬边界，mask 值为 0 或 1
  - `"soft"`: 软边界，使用 Gaussian blur 平滑过渡
- `soft_kernel_size`: Gaussian kernel 大小
  - 越大越平滑，但过渡区域越宽
  - 推荐：3-7

**效果对比**：
```
Hard mask:
[0, 0, 1, 1]  # 突变

Soft mask (kernel=5):
[0.0, 0.2, 0.8, 1.0]  # 平滑过渡
```

---

## 参数组合建议

### 推荐配置（默认）

```bash
# P2P 参数
--extra-param sparse_structure_strength=1.0 \
--extra-param slat_strength=1.0 \

# Blend 参数
--extra-param blend_strength=1.0 \
--extra-param blend_slat_enabled=true \
--extra-param slat_blend_mode=soft \
--extra-param slat_soft_kernel_size=5
```

**特点**：
- P2P 完全生效（strength=1.0）
- Latent blend 完全遵循 mask（blend_strength=1.0）
- 使用 soft mask 平滑边界

### 消融实验 1：测试 P2P 的作用

```bash
# 禁用 P2P
--extra-param sparse_structure_strength=0.0 \
--extra-param slat_strength=0.0 \

# 保持 Blend
--extra-param blend_strength=1.0
```

**对比**：有 P2P vs 无 P2P

### 消融实验 2：测试 Blend 的作用

```bash
# 保持 P2P
--extra-param sparse_structure_strength=1.0 \
--extra-param slat_strength=1.0 \

# 禁用 Blend
--extra-param blend_strength=0.0
```

**对比**：有 Blend vs 无 Blend

### 消融实验 3：Hard vs Soft Mask

```bash
# Hard mask
--extra-param slat_blend_mode=hard

# Soft mask
--extra-param slat_blend_mode=soft \
--extra-param slat_soft_kernel_size=5
```

**对比**：硬边界 vs 软边界

### 消融实验 4：不同 Kernel Size

```bash
# 小 kernel（较硬）
--extra-param slat_soft_kernel_size=3

# 中 kernel（推荐）
--extra-param slat_soft_kernel_size=5

# 大 kernel（很平滑）
--extra-param slat_soft_kernel_size=7
```

**对比**：不同平滑程度

---

## 参数优先级

当多个参数同时作用时：

1. **阶段开关** (`blend_ss_enabled`, `blend_slat_enabled`)
   - 如果禁用，该阶段的所有其他参数无效

2. **Blend Strength**
   - 全局缩放，影响所有混合

3. **Mask Mode**
   - 决定 mask 的形状（hard/soft）

4. **P2P Strength**
   - 独立作用，不影响 Blend

---

## 完整示例

### 示例 1：最强保留

```bash
python run_edit_experiment.py \
  --method image_p2p_latent_blend \
  --source-image source.png \
  --edit-image edit.png \
  --mask-image mask.png \
  --case-name max_preserve \
  --seed 1 \
  --preprocess \
  --extra-param sparse_structure_strength=1.0 \
  --extra-param slat_strength=1.0 \
  --extra-param blend_strength=1.0 \
  --extra-param slat_blend_mode=soft \
  --extra-param slat_soft_kernel_size=5
```

### 示例 2：平衡编辑

```bash
python run_edit_experiment.py \
  --method image_p2p_latent_blend \
  --source-image source.png \
  --edit-image edit.png \
  --mask-image mask.png \
  --case-name balanced \
  --seed 1 \
  --preprocess \
  --extra-param sparse_structure_strength=0.8 \
  --extra-param slat_strength=0.8 \
  --extra-param blend_strength=0.7 \
  --extra-param slat_blend_mode=soft \
  --extra-param slat_soft_kernel_size=5
```

### 示例 3：最大变化

```bash
python run_edit_experiment.py \
  --method image_p2p_latent_blend \
  --source-image source.png \
  --edit-image edit.png \
  --mask-image mask.png \
  --case-name max_change \
  --seed 1 \
  --preprocess \
  --extra-param sparse_structure_strength=0.0 \
  --extra-param slat_strength=0.0 \
  --extra-param blend_strength=0.0 \
  --extra-param slat_blend_mode=hard
```

---

## 调试技巧

### 问题：保留区域被修改了

**可能原因**：
1. Mask 不正确（白色/黑色反了）
2. Strength 太低

**解决**：
```bash
# 增加所有 strength
--extra-param sparse_structure_strength=1.0 \
--extra-param slat_strength=1.0 \
--extra-param blend_strength=1.0
```

### 问题：编辑区域变化不够

**可能原因**：
1. Strength 太高
2. Mask 覆盖范围太大

**解决**：
```bash
# 降低 strength
--extra-param blend_strength=0.5

# 或使用 hard mask
--extra-param slat_blend_mode=hard
```

### 问题：边界有明显接缝

**可能原因**：使用了 hard mask

**解决**：
```bash
# 使用 soft mask 并增大 kernel
--extra-param slat_blend_mode=soft \
--extra-param slat_soft_kernel_size=7
```

---

## 总结

| 参数类型 | 作用阶段 | 主要用途 | 推荐值 |
|---------|---------|---------|--------|
| P2P Strength | 生成时 | 控制注意力混合 | 1.0 |
| Blend Strength | 后处理 | 控制特征混合 | 1.0 |
| Blend Mode | 后处理 | 控制边界平滑 | soft |
| Kernel Size | 后处理 | 控制平滑程度 | 5 |

**记住**：
- 所有 strength 参数范围都是 [0, 1]
- 1.0 = 最强效果
- 0.0 = 无效果
- 可以独立调整每个参数做消融实验
