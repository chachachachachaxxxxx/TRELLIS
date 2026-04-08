# UniEdit GLB/PLY 导出问题修复

## 问题描述

UniEdit 消融测试完成后，没有生成 GLB 和 PLY 文件，只有预处理图像和元数据。

## 根本原因

### 1. decode_modes 配置问题

**原因**: 默认配置只解码 `["mesh"]`，但 GLB/PLY 导出需要 `gaussian`

```python
# 原始默认配置
"decode_modes": ["mesh"]  # ❌ 只有 mesh，无法生成 GLB/PLY
```

**GLB 导出条件** (在 `editing/common/save_utils.py:155`):
```python
if not skip_glb and "gaussian" in outputs and "mesh" in outputs:
    # 需要同时有 gaussian 和 mesh
```

**PLY 导出条件** (在 `editing/common/save_utils.py:167`):
```python
if not skip_ply and "gaussian" in outputs:
    # 需要 gaussian
```

### 2. CLI 参数解析问题

**原因**: CLI 传入的 `decode_modes=gaussian,mesh` 被解析为字符串而不是列表

```python
# CLI 输入
decode_modes=gaussian,mesh

# 解析结果
extra_params["decode_modes"] = "gaussian,mesh"  # ❌ 字符串

# 期望结果
extra_params["decode_modes"] = ["gaussian", "mesh"]  # ✅ 列表
```

## 解决方案

### 修复 1: 更新默认配置

**文件**: `editing/methods/image_uniedit_rf_inversion.py:493`

```python
def get_default_config(self) -> Dict:
    return {
        # ...
        "decode_modes": ["gaussian", "mesh"],  # ✅ 同时解码 gaussian 和 mesh
    }
```

### 修复 2: 处理字符串输入

**文件**: `editing/methods/image_uniedit_rf_inversion.py:280-287`

```python
# Decode
print("Decoding final result...")
extra = config.extra_params or {}
decode_modes = extra.get("decode_modes", ["mesh"])

# Handle string input (e.g., "gaussian,mesh" from CLI)
if isinstance(decode_modes, str):
    decode_modes = [m.strip() for m in decode_modes.split(",")]

print(f"Decode modes: {decode_modes}")
outputs = pipeline.decode_slat(slat_tgt, decode_modes)
```

### 修复 3: 添加调试日志

**文件**: `editing/common/save_utils.py`

添加日志以便调试：
```python
print(f"Saving outputs: {list(outputs.keys())}, num_samples={num_samples}")
print(f"skip_glb={skip_glb}, skip_ply={skip_ply}, skip_render={skip_render}")

# ...

print(f"Exporting GLB for sample {sample_idx}...")
# ...
print(f"✓ Saved GLB: {glb_path}")

print(f"Exporting PLY for sample {sample_idx}...")
# ...
print(f"✓ Saved PLY: {ply_path}")
```

## 验证

### 重新运行测试

```bash
./rerun_uniedit_with_export.sh
```

测试脚本会：
1. 使用 `decode_modes=gaussian,mesh` 参数
2. 字符串会被正确解析为列表
3. 同时解码 gaussian 和 mesh
4. 生成 GLB 和 PLY 文件

### 预期输出

```
outputs/image_uniedit_rf_inversion/test0_preserve_uniedit/edit/
├── sample_00.glb  # ✅ GLB 文件
├── sample_00.ply  # ✅ PLY 文件
├── config.json
├── edit_preprocessed.png
├── source_preprocessed.png
├── mask_preprocessed.png
└── uniedit_metadata.json

outputs/image_uniedit_rf_inversion/test0_free_target/edit/
├── sample_00.glb  # ✅ GLB 文件
├── sample_00.ply  # ✅ PLY 文件
├── config.json
├── edit_preprocessed.png
├── source_preprocessed.png
├── mask_preprocessed.png
└── uniedit_metadata.json
```

## 相关文件

- `editing/methods/image_uniedit_rf_inversion.py` - UniEdit 方法实现
- `editing/common/save_utils.py` - 输出保存工具
- `run_edit_experiment.py` - 统一入口脚本
- `rerun_uniedit_with_export.sh` - 重新运行测试脚本

## 经验教训

1. **默认配置要完整**: 默认配置应该支持所有常用功能（GLB/PLY 导出）
2. **参数类型要灵活**: CLI 参数解析需要处理字符串和列表两种输入
3. **添加调试日志**: 关键步骤添加日志，便于排查问题
4. **文档要清晰**: 明确说明各个输出格式的依赖关系
