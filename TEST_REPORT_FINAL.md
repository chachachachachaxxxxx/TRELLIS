# TRELLIS 编辑方法测试报告

**测试日期**: 2026-04-08  
**测试 Case**: test0  
**Seed**: 1  
**GPU**: CUDA 3

---

## 测试结果概览

| # | 方法 | 状态 | 输出目录 | 备注 |
|---|------|------|----------|------|
| 1 | image_prompt_to_prompt | ✅ 成功 | `outputs/image_prompt_to_prompt/test0/` | GLB (1.3M), PLY (11M) |
| 2 | image_slat_xor_fusion | ✅ 成功 | `outputs/image_slat_xor_fusion/test0/` | GLB (1.4M), PLY (20M) |
| 3 | image_prompt_to_prompt_rf_inversion | ✅ 成功 | `outputs/image_prompt_to_prompt_rf_inversion/test0/` | RF inversion + P2P |
| 4 | image_uniedit_rf_inversion (preserve_uniedit) | ⚠️ 部分 | `outputs/image_uniedit_rf_inversion/test0/` | 预处理完成，未生成模型 |
| 5 | image_uniedit_rf_inversion (free_target) | ❌ 未运行 | - | 脚本提前终止 |
| 6 | image_uniedit_rf_inversion (latent_replace_union) | ❌ 未运行 | - | 脚本提前终止 |

**完成率**: 3/6 (50%)

---

## 详细结果

### 1. ✅ Image Prompt-to-Prompt

**方法**: 基础的 Prompt-to-Prompt attention 注入

**输出文件**:
- `sample_00.glb` (1.3M)
- `sample_00.ply` (11M)
- 预处理图像和 token 元数据

**验证**: ✅ 通过

---

### 2. ✅ Image SLAT XOR Fusion

**方法**: SLAT 块融合，复用 source 重叠区域

**输出文件**:
- `sample_00.glb` (1.4M)
- `sample_00.ply` (20M)
- `fusion_stats.json`

**融合统计**:
```json
{
  "fused_total_voxel_count": 9232,
  "reused_source_voxel_count_total": 1717,
  "generated_target_voxel_count_total": 7515,
  "source_voxel_count": 7125,
  "target_voxel_count": 9232,
  "overlap_reused_from_source_count": 1717
}
```

**验证**: ✅ 通过

---

### 3. ✅ Image Prompt-to-Prompt RF Inversion

**方法**: RF inversion 初始化 + Prompt-to-Prompt

**输出文件**:
- 完整的编辑结果
- RF inversion 统计

**验证**: ✅ 通过

---

### 4. ⚠️ Image UniEdit RF Inversion (preserve_uniedit)

**方法**: UniEdit 两阶段编辑（preserve_uniedit 模式）

**状态**: 预处理完成，但未生成最终模型

**已生成文件**:
- `config.json`
- `edit_preprocessed.png`
- `source_preprocessed.png`
- `mask_preprocessed.png`

**问题**: 可能在采样或后处理阶段失败

**验证**: ⚠️ 需要调试

---

## Source Assets

**生成方式**: 多视角渲染（CYCLES 引擎，150 视角）

**目录**: `outputs/source_assets_test0_multiview/`

**文件**:
- `voxels.ply` (84K) - 7125 个体素
- `features.npz` (13M) - DINOv2 特征
  - `indices`: (7125, 3)
  - `patchtokens`: (7125, 1024)
- `mesh.ply` (205K)
- `transforms.json` (130K)
- `000.png` ~ `149.png` - 150 个渲染图像

**验证**: ✅ 格式正确

---

## 架构改进

### 新增模块

1. **`editing/rendering/`** - 多视角渲染
   - `bpy_render.py` - Blender 渲染功能
   
2. **`editing/preprocess/feature_extraction.py`** - 特征提取
   - 从多视角图像提取 DINOv2 特征

3. **`editing/preprocess/source_assets.py`** - Source Assets 生成
   - `generate_source_assets_from_model()` - 从 3D 模型生成（推荐）
   - `generate_source_assets_from_image()` - 从图像采样生成（快速）
   - `ensure_source_assets()` - 统一接口

### 清理

- ✅ `temp/` 目录已清空
- ✅ 功能已整合到 `editing/` 模块

---

## 关键发现

### 1. Features.npz 格式要求

**错误格式**（TRELLIS 采样生成）:
```python
{
  'feats': (N, 8),
  'coords': (N, 4)
}
```

**正确格式**（多视角渲染提取）:
```python
{
  'patchtokens': (N, 1024),  # DINOv2 特征
  'indices': (N, 3)           # 体素索引
}
```

**结论**: 必须使用多视角渲染方式生成 source assets

### 2. 渲染引擎选择

- **CYCLES**: 高质量，需要 GPU，速度适中
- **BLENDER_EEVEE**: 快速，但质量较低
- **推荐**: CYCLES + 指定 CUDA 设备

---

## 待完成工作

### 高优先级

1. **调试 UniEdit 方法**
   - 检查为什么 preserve_uniedit 模式未生成最终模型
   - 修复后重新测试

2. **完成剩余测试**
   - image_uniedit_rf_inversion (free_target)
   - image_uniedit_rf_inversion (latent_replace_union)

### 中优先级

3. **优化测试脚本**
   - 添加错误处理
   - 在失败时继续运行后续测试

4. **文档更新**
   - 更新 `docs/REFACTORING_STATUS.md`
   - 添加 source assets 生成指南

---

## 总结

✅ **成功验证了 3 个编辑方法**  
✅ **建立了完整的 source assets 生成流程**  
✅ **整合了多视角渲染功能**  
⚠️ **需要调试 UniEdit 方法**  

编辑框架基本可用，核心功能已验证通过。
