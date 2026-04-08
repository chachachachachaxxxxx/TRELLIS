# UniEdit + P2P 混合编辑方法

## 概述

`image_uniedit_p2p_hybrid` 是一个结合了 UniEdit 和 Prompt-to-Prompt 两种技术的混合 3D 编辑方法。

## 核心原理

### 1. UniEdit 的 Latent Replacement

在去噪过程中，UniEdit 使用 RF-Solver 同时对源图像和目标图像进行反向求解：

```
对于每个时间步 t：
  1. 分别预测源和目标的噪声：
     noise_src = model(x_src, t, cond_src)
     noise_tgt = model(x_tgt, t, cond_tgt)
  
  2. 根据 mask 选择性混合：
     noise_mixed = selector * noise_src + (1 - selector) * noise_tgt
     
  3. 更新 latent：
     x = x - dt * noise_mixed
```

**作用**：确保未编辑区域的几何结构完全保留

### 2. Prompt-to-Prompt 的 Attention Injection

P2P 通过 hook 拦截模型的 cross-attention 计算：

```
在 cross-attention 中：
  1. 计算两组注意力图：
     attn_src = softmax(Q @ K_src)   # 源图像的注意力
     attn_edit = softmax(Q @ K_edit) # 编辑图像的注意力
  
  2. 根据 mask 混合注意力：
     对于保留区域的 token：
       attn_mixed[token] = strength * attn_src[token] + (1-strength) * attn_edit[token]
  
  3. 使用混合注意力计算输出：
     output = attn_mixed @ V_edit
```

**作用**：提供细粒度的特征级别控制

### 3. 两者如何协同

两种机制在去噪的每一步**同时生效**：

```python
# UniEdit 的去噪循环
for t in timesteps:
    # P2P hook 在 model() 内部拦截 attention
    noise_src = model(x_src, t, cond_src)  # ← P2P 在这里工作
    noise_tgt = model(x_tgt, t, cond_tgt)  # ← P2P 在这里工作
    
    # UniEdit 在外部混合预测结果
    noise_mixed = selector * noise_src + (1 - selector) * noise_tgt
    x = x - dt * noise_mixed
```

**关键点**：
- P2P 影响 attention 计算（模型内部）
- UniEdit 混合不同条件的预测（模型外部）
- 两者是**正交的**，互不干扰

## 使用方法

### 基本命令

```bash
python run_edit_experiment.py \
  --method image_uniedit_p2p_hybrid \
  --source-model <source_assets_dir> \
  --source-image <source.png> \
  --edit-image <edit.png> \
  --mask-image <mask.png> \
  --mask-glb <mask.glb> \
  --case-name <case_name> \
  --seed 1 \
  --preprocess
```

### 参数配置

#### UniEdit 参数

- `stage2_variant`: SLAT 编辑模式
  - `preserve_uniedit`: 保留源 SLAT 特征（最保守）
  - `free_target`: 自由生成目标 SLAT（最激进）
  - `latent_replace_union`: 混合模式（推荐，默认）

- `ss_omega`: 稀疏结构混合强度 (0-1)，默认 1.0
- `slat_omega`: SLAT 混合强度 (0-1)，默认 1.0
- `cfg_interval`: CFG 应用区间，默认 (0.5, 1.0)

#### P2P 参数

- `inject_stages`: 注入阶段，可选 `sparse_structure`, `slat`
  - 默认：`["slat"]`（只在 SLAT 阶段注入）
  - 推荐：只在 SLAT 阶段注入，避免过度约束

- `p2p_strength`: 注意力混合强度 (0-1)
  - 默认：0.8（略低于 1.0，让 UniEdit 主导）
  - 1.0 = 完全使用源注意力
  - 0.0 = 完全使用编辑注意力

- `slat_t_start`: SLAT 阶段开始时间步（归一化）
  - 默认：0.8（从 80% 开始）

- `slat_t_end`: SLAT 阶段结束时间步（归一化）
  - 默认：0.0（到结束）

- `patch_coverage_threshold`: Patch 覆盖阈值
  - 默认：0.0（任何覆盖都算）

- `query_chunk`: 注意力计算的 chunk 大小
  - 默认：1024（平衡速度和显存）

### 完整示例

```bash
python run_edit_experiment.py \
  --method image_uniedit_p2p_hybrid \
  --source-model outputs/source_assets_test0 \
  --source-image assets/edit_example/images/2d_render.png \
  --edit-image assets/edit_example/images/2d_edit.png \
  --mask-image assets/edit_example/images/2d_mask.png \
  --mask-glb assets/edit_example/mask.glb \
  --case-name hybrid_test \
  --seed 1 \
  --preprocess \
  --ss-steps 12 \
  --slat-steps 12 \
  --extra-param stage2_variant=latent_replace_union \
  --extra-param inject_stages=slat \
  --extra-param p2p_strength=0.8 \
  --extra-param slat_t_start=0.8 \
  --extra-param slat_t_end=0.0 \
  --extra-param decode_modes=mesh,gaussian
```

## 输入要求

1. **Source Assets**：源 3D 模型的资产
   - `voxels.ply`: 稀疏结构
   - `features.npz`: SLAT 特征

2. **Images**：
   - `source_image`: 源图像（可选，用于条件编码）
   - `edit_image`: 编辑图像（必需）
   - `mask_image`: 2D mask 图像（必需，用于 P2P）

3. **3D Mask**：
   - `mask_glb`: 3D mask GLB 文件（必需，用于 UniEdit）

## 优势与适用场景

### 优势

1. **强结构保留**：UniEdit 的 latent replacement 确保未编辑区域的几何完全一致
2. **细节控制**：P2P 的 attention injection 提供特征级别的精细控制
3. **互补性**：两种机制处理不同层次的编辑需求

### 适用场景

- 需要精确保留部分区域的编辑任务
- 需要同时控制结构和细节的复杂编辑
- 对编辑质量要求较高的场景

### 与其他方法对比

| 方法 | 结构保留 | 细节控制 | 灵活性 | 速度 |
|------|---------|---------|--------|------|
| Image P2P | 中 | 高 | 高 | 快 |
| UniEdit | 高 | 中 | 中 | 中 |
| **Hybrid** | **高** | **高** | **中** | **中** |

## 参数调优建议

### 保守编辑（最大化保留）

```bash
--extra-param stage2_variant=preserve_uniedit \
--extra-param p2p_strength=1.0 \
--extra-param slat_omega=1.0
```

### 激进编辑（最大化变化）

```bash
--extra-param stage2_variant=free_target \
--extra-param p2p_strength=0.5 \
--extra-param slat_omega=0.5
```

### 平衡编辑（推荐）

```bash
--extra-param stage2_variant=latent_replace_union \
--extra-param p2p_strength=0.8 \
--extra-param slat_omega=1.0
```

## 技术细节

### 执行流程

1. **准备阶段**：
   - 加载源资产（voxels, features）
   - 加载 mask（GLB 和 image）
   - 编码条件（source_cond, edit_cond）
   - 构建 P2P hook 和 token metadata

2. **Stage 0 - 反演**：
   - 反演稀疏结构：`source_coords → ss_terminal_noise`
   - 反演 SLAT：`source_slat → slat_terminal_noise`

3. **Stage 1 - 稀疏结构编辑**：
   - 使用 UniEdit RF-Solver 编辑稀疏结构
   - 根据 mask 混合源和目标坐标

4. **Stage 2 - SLAT 编辑（混合）**：
   - **同时应用**：
     - UniEdit 的 latent replacement
     - P2P 的 attention injection
   - 生成最终 SLAT

5. **解码**：
   - 解码为 mesh 和 gaussian

### 显存优化

- 在 GLB 导出前自动 offload 模型到 CPU
- 使用 chunked attention 计算（`query_chunk=1024`）
- 及时释放中间结果

## 故障排查

### 常见问题

1. **OOM (Out of Memory)**
   - 减少 `query_chunk` 大小
   - 减少 `ss_steps` 和 `slat_steps`
   - 只解码 mesh：`--extra-param decode_modes=mesh`

2. **编辑效果不明显**
   - 增加 `p2p_strength`
   - 调整 `stage2_variant` 为 `free_target`
   - 检查 mask 是否正确

3. **保留区域被修改**
   - 增加 `slat_omega` 到 1.0
   - 使用 `preserve_uniedit` 模式
   - 检查 mask_glb 是否正确

## 实现文件

- 方法实现：`editing/methods/image_uniedit_p2p_hybrid.py`
- Hook 实现：`editing/hooks/prompt_to_prompt.py`
- RF-Solver：`editing/inversion/rf_sampler.py`
- UniEdit Sampler：`editing/inversion/uniedit_sampler.py`
- 注册表：`editing/methods/registry.py`

## 参考

- UniEdit 原理：`docs/OOM_AND_UNIEDIT_FIX.md`
- P2P 原理：`editing/hooks/prompt_to_prompt.py`
- 方法迁移指南：`docs/METHOD_MIGRATION_GUIDE.md`
