# Image P2P Latent Blend 测试说明

## 方法已实现并注册

✅ 方法实现：`editing/methods/image_p2p_latent_blend.py`
✅ 已注册到 registry：`image_p2p_latent_blend`
✅ 测试脚本已创建

## 测试脚本

### 快速测试（单个配置）
```bash
./test_p2p_blend_quick.sh
```

配置：Soft mask (kernel=5), blend_strength=1.0

### 完整测试（4个消融实验）
```bash
./test_p2p_latent_blend_real.sh
```

包含：
1. Soft mask (kernel=5) - 推荐配置
2. Hard mask - 消融实验
3. Soft mask (kernel=7) - 更平滑
4. Blend strength=0.5 - 强度消融

## 环境要求

⚠️ **注意**：需要安装 flash_attn 或 xformers

如果环境中没有安装，需要：
```bash
# 选项 1: 安装 flash_attn
pip install flash-attn

# 选项 2: 安装 xformers
pip install xformers

# 然后在脚本中指定 backend
--attn-backend flash_attn  # 或 xformers
```

## 手动测试命令

如果脚本无法运行，可以手动执行：

```bash
# 确保环境变量设置
export CUDA_VISIBLE_DEVICES=3
export SPCONV_ALGO=native

# 运行测试
python run_edit_experiment.py \
  --method image_p2p_latent_blend \
  --source-image assets/edit_example/images/2d_render.png \
  --edit-image assets/edit_example/images/2d_edit.png \
  --mask-image assets/edit_example/images/2d_mask.png \
  --case-name p2p_blend_test \
  --seed 1 \
  --preprocess \
  --ss-steps 12 \
  --slat-steps 12 \
  --extra-param blend_slat_enabled=true \
  --extra-param slat_blend_mode=soft \
  --extra-param slat_soft_kernel_size=5 \
  --extra-param blend_strength=1.0
```

## 预期输出

成功运行后，输出目录：
```
outputs/image_p2p_latent_blend/<case_name>/
├── sample.glb              # 最终 3D 模型
├── sample.ply              # 点云
├── source_comparison/      # 源模型（用于对比）
│   ├── sample.glb
│   └── sample.ply
├── edit_preprocessed.png   # 预处理后的编辑图像
├── mask_preprocessed.png   # 预处理后的 mask
└── ...
```

## 核心功能验证

测试应该验证：
1. ✅ 找到重合体素
2. ✅ 3D 坐标投影到 2D mask
3. ✅ 根据 mask 值混合特征
4. ✅ Soft mask 平滑边界
5. ✅ 生成最终 GLB 文件

## 预期日志输出

```
Generating source 3D...
Generating edit 3D...
Blending SLAT features (strength=1.0)...
Found X overlapping voxels out of Y source and Z edit voxels
Blended X voxel features (strength=1.0)
Decoding blended result...
Saving outputs...
```

## 故障排查

### 问题 1: attention backend 未安装
```
Error: Neither flash_attn nor xformers is installed
```
**解决**：安装 flash_attn 或 xformers

### 问题 2: 方法未找到
```
Error: Unknown method 'image_p2p_latent_blend'
```
**解决**：确保已提交 registry 更改

### 问题 3: 没有重合体素
```
Found 0 overlapping voxels
```
**原因**：源和编辑的 3D 结构完全不同
**解决**：检查输入图像是否合理

### 问题 4: OOM (显存不足)
**解决**：
- 减少 steps：`--ss-steps 8 --slat-steps 8`
- 只解码 mesh：`--extra-param decode_modes=mesh`

## 下一步

1. 在有 attention backend 的环境中运行测试
2. 比较不同配置的结果：
   - Hard vs Soft mask
   - 不同 kernel size (3, 5, 7)
   - 不同 blend_strength (0.5, 1.0)
3. 与其他方法对比：
   - vs Image P2P (无混合)
   - vs UniEdit (RF inversion + 去噪中混合)
   - vs Hybrid (UniEdit + P2P)

## 参考

- 方法文档：`docs/P2P_LATENT_REPLACE_METHOD.md`
- VoxHammer 参考：`temp/edit_pipeline.py`
- 实现文件：`editing/methods/image_p2p_latent_blend.py`
