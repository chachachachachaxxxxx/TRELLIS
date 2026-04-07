# 预处理模块实现总结

## 完成的工作

### 1. 创建 `editing/preprocess/asset_3d.py` ✅

抽离了 3D 资产预处理逻辑，包括：

#### PLY 文件处理

- **`load_ply_positions()`** - 从 PLY 加载顶点坐标（支持 utils3d 和 trimesh）
- **`ply_to_coords()`** - PLY 坐标转换为 64³ 体素坐标
- **`coords_to_voxel()`** - 稀疏坐标转换为密集体素张量

#### SLAT 特征处理

- **`feats_to_slat()`** - 从 features.npz 加载并编码 SLAT 特征

#### 3D Mask 处理

- **`load_source_voxel_normalization()`** - 加载 transforms.json 归一化参数
- **`load_mask_glb_coords()`** - 从 GLB 或预生成 PLY 加载 3D mask 坐标
  - 策略 1: 读取 voxels_delete.ply
  - 策略 2: 使用 VoxHammer 过滤生成
- **`_load_mask_mesh()`** - 加载并验证 mask mesh

#### RF Inversion 工具

- **`coords_to_flat_indices()`** - 3D/4D 坐标转换为扁平索引
- **`sparse_batch_slice()`** - 提取 batch 的坐标和特征
- **`project_sparse_terminal_noise()`** - 将源噪声投影到目标坐标

#### 数据类

- **`VoxelNormalization`** - 体素归一化元数据
- **`MaskGLBResult`** - 3D mask 加载结果

### 2. 增强 `editing/io/path_resolver.py` ✅

添加了资产路径解析功能：

- **`SOURCE_RENDER_CANDIDATES`** - 常见 source render 文件名列表
- **`ensure_path_exists()`** - 验证路径存在
- **`resolve_asset_dir()`** - 解析资产目录
- **`resolve_source_image_path()`** - 自动查找 source render 图像
- **`validate_required_asset_files()`** - 验证 RF inversion 所需文件
- **`candidate_file()`** - 检查路径是否为文件
- **`resolve_image_dir()`** - 解析并验证图像目录

### 3. 更新模块导出 ✅

- 更新 `editing/preprocess/__init__.py` 导出所有 3D 预处理函数
- 更新 `editing/io/__init__.py` 导出所有路径解析函数

### 4. 验证实现 ✅

#### 语法验证

```bash
python -m compileall editing/preprocess/asset_3d.py editing/preprocess/__init__.py \
  editing/io/path_resolver.py editing/io/__init__.py
```

✓ 所有文件编译通过

#### 导入验证

```python
from editing.preprocess import (
    load_ply_positions, ply_to_coords, coords_to_voxel,
    feats_to_slat, load_source_voxel_normalization,
    load_mask_glb_coords, VoxelNormalization, MaskGLBResult,
)
from editing.io import (
    ensure_path_exists, resolve_asset_dir, resolve_source_image_path,
    validate_required_asset_files, candidate_file, resolve_image_dir,
    SOURCE_RENDER_CANDIDATES,
)
```

✓ 所有导入成功

#### 功能验证

创建并运行 `test_preprocess_validation.py`：

- ✓ 2D 图像预处理（mask 工具、对齐输入）
- ✓ 3D 资产预处理（坐标转换、归一化）
- ✓ 路径解析（文件查找、目录解析）

#### 实际脚本验证

```python
import example_image_prompt_to_prompt
```

✓ 现有脚本能正常导入和使用新模块

## 预处理模块架构

```
editing/preprocess/
├── __init__.py           # 统一导出
├── image_alignment.py    # 2D 图像对齐、裁剪、resize
├── mask_utils.py         # 2D mask 处理
├── asset_3d.py          # 3D 资产预处理（新增）
└── artifacts.py         # 预处理结果保存

editing/io/
├── __init__.py          # 统一导出
├── case_loader.py       # Case 加载
└── path_resolver.py     # 路径解析（增强）
```

## 已有实现（之前完成）

### 2D 图像预处理

- `extract_foreground_rgba()` - 前景提取（RGBA/rembg）
- `build_union_crop_context()` - 联合裁剪上下文
- `prepare_aligned_inputs()` - 主预处理入口
- `prepare_edit_condition_image()` - 单图编辑条件
- `build_auto_mask()` - 自动生成 mask
- `build_blank_mask()` - 空白 mask
- `extract_mask_channel()` - 提取 mask 通道
- `binarize_mask_image()` - 二值化 mask

### 输出保存

- `save_preprocessed_inputs()` - 保存预处理结果
- `save_edit_condition_artifacts()` - 保存编辑条件

## 下一步建议

### 阶段 B：方法实现拆离

现在预处理层已经完成，可以开始：

1. **定义统一方法接口** - `editing/methods/base.py`

   ```python
   class EditMethod:
       method_name: str
       def prepare(self, pipeline, case, config): ...
       def run(self, pipeline, case, config): ...
       def save_artifacts(self, outputs, out_dir): ...
   ```

2. **抽离方法实现**
   - `editing/methods/image_prompt_to_prompt.py`
   - `editing/methods/image_prompt_to_prompt_rf_inversion.py`
   - `editing/methods/image_uniedit_rf_inversion.py`

3. **抽离公共 hook/patch 工具**
   - `editing/hooks/attention_patch.py`
   - `editing/hooks/cross_attention_trace.py`

4. **抽离 inversion 公共层**
   - `editing/inversion/rf_inversion.py`
   - `editing/inversion/noise_projection.py`

## 验证方式

所有改动都经过：

1. Python 语法检查（`compileall`）
2. 导入测试
3. 功能单元测试
4. 现有脚本兼容性测试

没有修改 TRELLIS 核心代码，所有改动都在 `editing/` 模块中。
