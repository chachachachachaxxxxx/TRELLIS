# OOM 和 UniEdit GLB 导出问题修复报告

## 问题概述

### 1. GLB 导出 OOM 错误
**症状**: 所有编辑方法在 GLB 导出的 texture baking 阶段出现 CUDA OOM 错误
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 12.00 MiB. 
GPU 0 has a total capacity of 31.36 GiB of which 8.12 MiB is free.
```

**影响**: 即使在 32GB GPU 上也无法完成 GLB 导出

### 2. UniEdit 无法生成 GLB 文件
**症状**: `image_uniedit_rf_inversion` 方法运行完成但输出目录中没有任何 3D 文件（.glb、.ply）

**影响**: UniEdit 方法完全无法使用

## 根本原因分析

### OOM 问题
1. Pipeline 的所有模型（sparse structure flow、SLAT flow、DINOv2 等）在采样完成后仍占用 ~30GB GPU 显存
2. GLB 导出的 texture baking 阶段需要额外显存来渲染和优化纹理
3. 默认解码 3 种格式（mesh、gaussian、radiance_field），其中 radiance_field 不是 GLB/PLY 导出必需的

### UniEdit 参数解析问题
测试脚本传递参数格式错误：
```bash
--extra-param decode_modes='[\"mesh\"]'
```
导致参数被解析为字符串 `"[\"mesh\"]"` 而不是列表 `["mesh"]`，解码失败。

## 修复方案

### 1. Pipeline 模型卸载 (OOM 修复)

**文件**: `editing/common/save_utils.py`
```python
def offload_models_to_cpu(pipeline) -> None:
    """Offload pipeline models to CPU to free GPU memory before GLB export."""
    if hasattr(pipeline, 'models'):
        for model in pipeline.models.values():
            if hasattr(model, 'cpu'):
                model.cpu()
    release_cuda_memory()
```

**文件**: `editing/methods/runner.py`
```python
# Run method
outputs = self.method.run(self.pipeline, prepared_state, config)

# Offload pipeline models to CPU before GLB export to free GPU memory
from editing.common.save_utils import offload_models_to_cpu
offload_models_to_cpu(self.pipeline)

# Save artifacts
artifact_paths = self.method.save_artifacts(outputs, out_dir, config)
```

**效果**: 在 GLB 导出前释放 ~30GB GPU 显存

### 2. 减少解码格式 (OOM 优化)

将所有方法的默认解码格式从 `["mesh", "gaussian", "radiance_field"]` 改为 `["mesh", "gaussian"]`

**修改的文件**:
- `editing/methods/image_prompt_to_prompt.py`
- `editing/methods/image_slat_xor_fusion.py`

**效果**: 跳过 radiance_field 解码，节省显存和时间，不影响 GLB/PLY 导出

### 3. UniEdit 参数解析增强

**文件**: `editing/methods/image_uniedit_rf_inversion.py`
```python
# Handle string input (e.g., "gaussian,mesh" from CLI or "[\"mesh\"]" from shell)
if isinstance(decode_modes, str):
    # Try to parse JSON-like string first
    import json
    try:
        decode_modes = json.loads(decode_modes)
    except (json.JSONDecodeError, ValueError):
        # Fall back to comma-separated parsing
        decode_modes = [m.strip() for m in decode_modes.split(",")]
```

**文件**: `test_all_methods.sh`
```bash
# 从复杂的 JSON 格式改为简单的逗号分隔格式
--extra-param decode_modes=mesh,gaussian
```

**效果**: 支持多种参数格式，确保正确解析

### 4. 更新默认解码格式

将 UniEdit 的默认 `decode_modes` 从 `["mesh"]` 改为 `["mesh", "gaussian"]`，确保可以同时导出 GLB 和 PLY。

## 验证结果

### OOM 修复验证
```bash
CUDA_VISIBLE_DEVICES=3 python run_edit_experiment.py \
  --method image_prompt_to_prompt \
  --source-image assets/edit_example/images/2d_render.png \
  --edit-image assets/edit_example/images/2d_edit.png \
  --mask-image assets/edit_example/images/2d_mask.png \
  --case-name oom_fixed_test \
  --seed 1 \
  --preprocess
```

**结果**: ✅ 成功完成，生成 GLB 和 PLY 文件，无 OOM 错误

### UniEdit 修复验证
```bash
CUDA_VISIBLE_DEVICES=3 python run_edit_experiment.py \
  --method image_uniedit_rf_inversion \
  --source-model outputs/source_assets_test0_multiview \
  --source-image assets/edit_example/images/2d_render.png \
  --edit-image assets/edit_example/images/2d_edit.png \
  --mask-glb assets/edit_example/mask.glb \
  --case-name test_uniedit_fix2 \
  --seed 1 \
  --ss-steps 12 \
  --slat-steps 12 \
  --extra-param stage2_variant=preserve_uniedit \
  --extra-param decode_modes=mesh,gaussian
```

**结果**: 
- ✅ 参数正确解析: `Decode modes: ['mesh', 'gaussian']`
- ✅ GLB 文件: 1.7MB
- ✅ PLY 文件: 16MB
- ✅ 方法成功完成

## 修改的文件列表

1. `editing/common/save_utils.py` - 添加 `offload_models_to_cpu()` 函数
2. `editing/methods/runner.py` - 集成自动模型卸载
3. `editing/methods/image_prompt_to_prompt.py` - 减少解码格式
4. `editing/methods/image_slat_xor_fusion.py` - 减少解码格式
5. `editing/methods/image_uniedit_rf_inversion.py` - 增强参数解析，更新默认值
6. `test_all_methods.sh` - 简化 UniEdit 参数格式

## 其他发现

### DINOv2 加载慢的问题
**原因**: `torch.hub.load()` 每次都检查 GitHub 仓库

**建议**: 设置环境变量加速
```bash
export TORCH_HUB_OFFLINE=1  # 使用本地缓存，跳过 GitHub 检查
```

## 总结

所有问题已修复并验证通过：
- ✅ OOM 问题解决 - 32GB GPU 可以成功完成 GLB 导出
- ✅ UniEdit 可以正常生成 GLB/PLY 文件
- ✅ 所有方法的解码格式优化，节省显存和时间
- ✅ 参数解析更加健壮，支持多种格式

修复日期: 2026-04-08
