# 测试结果总结

## 已完成测试

### 1. ✅ image_prompt_to_prompt
- **状态**: 成功
- **输出**: `outputs/image_prompt_to_prompt/test0/`
- **文件**: GLB (1.3M), PLY (11M)

### 2. ✅ image_slat_xor_fusion  
- **状态**: 成功
- **输出**: `outputs/image_slat_xor_fusion/test0/`
- **文件**: GLB (1.4M), PLY (20M)
- **统计**: fusion_stats.json

### 3. ✅ image_prompt_to_prompt_rf_inversion
- **状态**: 成功
- **输出**: `outputs/image_prompt_to_prompt_rf_inversion/test0/`

### 4. ⚠️ image_uniedit_rf_inversion (preserve_uniedit)
- **状态**: 部分完成（预处理完成，但未生成最终模型）
- **输出**: `outputs/image_uniedit_rf_inversion/test0/`

### 5. ❓ image_uniedit_rf_inversion (free_target)
- **状态**: 未运行

### 6. ❓ image_uniedit_rf_inversion (latent_replace_union)
- **状态**: 未运行

## Source Assets

使用多视角渲染生成（正确格式）:
- **目录**: `outputs/source_assets_test0_multiview/`
- **文件**: 
  - voxels.ply (84K)
  - features.npz (13M) - 格式: ['indices', 'patchtokens']
  - mesh.ply (205K)
  - transforms.json (130K)
  - 150 个渲染图像

## 测试进度

- **完成**: 3/6 (50%)
- **部分完成**: 1/6
- **未运行**: 2/6
