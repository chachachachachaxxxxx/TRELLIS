# Edit3D-Bench 评测输出格式

## 概述

本文档说明如何将 TRELLIS_EDIT 的编辑结果转换为 Edit3D-Bench 评测系统所需的格式。

## 输出目录结构

Edit3D-Bench 评测系统要求以下目录结构：

```
exp_comparison/
└── {method_name}/                    # 方法名称（例如：image_prompt_to_prompt）
    └── {dataset}/                    # 数据集名称（例如：GSO, PartObjaverse-Tiny）
        └── {object_name}/            # 物体名称（例如：Toy_Poodle）
            └── prompt_{1,2,3}/       # 编辑提示 ID（1, 2, 或 3）
                └── edit.glb          # 编辑后的 3D 模型（必需）
```

**注意**：
- 评测系统会自动渲染 16 个视角的图像和视频
- 只需要提供 `edit.glb` 文件即可
- 目录结构必须与 Ground Truth 数据完全匹配

## 使用批处理脚本

### 基本用法

```bash
python batch_prepare_eval.py \
    --gt-root /path/to/Edit3D-Bench/data \
    --outputs-root outputs \
    --method-name image_prompt_to_prompt \
    --eval-root exp_comparison
```

### 参数说明

**必需参数：**
- `--gt-root`: Edit3D-Bench Ground Truth 数据根目录（包含 metadata.json）
- `--method-name`: 方法名称（例如：`image_prompt_to_prompt`）

**可选参数：**
- `--outputs-root`: TRELLIS_EDIT 输出目录（默认：`outputs`）
- `--eval-root`: 评测格式输出根目录（默认：`exp_comparison`）
- `--dataset`: 按数据集过滤（例如：`GSO`）
- `--object`: 按物体名称过滤（例如：`Toy_Poodle`）
- `--prompt-id`: 按提示 ID 过滤（1, 2, 或 3）
- `--dry-run`: 试运行，不实际复制文件
- `--max-cases`: 限制处理的案例数量

### 使用示例

**处理所有案例：**
```bash
python batch_prepare_eval.py \
    --gt-root /home/wangxinxing/code/Edit3Dpp/data \
    --method-name image_prompt_to_prompt \
    --eval-root exp_comparison
```

**仅处理 GSO 数据集：**
```bash
python batch_prepare_eval.py \
    --gt-root /home/wangxinxing/code/Edit3Dpp/data \
    --method-name image_prompt_to_prompt \
    --dataset GSO \
    --eval-root exp_comparison
```

**试运行（不实际复制）：**
```bash
python batch_prepare_eval.py \
    --gt-root /home/wangxinxing/code/Edit3Dpp/data \
    --method-name image_prompt_to_prompt \
    --dry-run
```

**处理前 10 个案例：**
```bash
python batch_prepare_eval.py \
    --gt-root /home/wangxinxing/code/Edit3Dpp/data \
    --method-name image_prompt_to_prompt \
    --max-cases 10
```

## 文件查找逻辑

脚本会在 `outputs/{method_name}/` 下查找编辑结果，尝试以下命名模式：

1. `{object_name}_prompt_{prompt_id}/edit/sample_00.glb`
2. `{dataset}_{object_name}_prompt_{prompt_id}/edit/sample_00.glb`
3. `{object_name}_p{prompt_id}/edit/sample_00.glb`
4. `{object_name}/edit/sample_00.glb`

如果以上模式都不匹配，会在所有子目录中搜索包含物体名称的目录。

## 输出示例

运行脚本后，会生成如下结构：

```
exp_comparison/
└── image_prompt_to_prompt/
    ├── GSO/
    │   ├── Toy_Poodle/
    │   │   ├── prompt_1/
    │   │   │   └── edit.glb
    │   │   ├── prompt_2/
    │   │   │   └── edit.glb
    │   │   └── prompt_3/
    │   │       └── edit.glb
    │   └── ...
    └── PartObjaverse-Tiny/
        └── ...
```

## 后续步骤

生成评测格式后，使用 Edit3D-Bench 评测系统进行评测：

```bash
cd VoxHammer/Edit3D-Bench

# 渲染多视角图像和视频
python render.py --base_dir /path/to/exp_comparison/image_prompt_to_prompt

# 运行评测
python eval_main.py \
    --gt_root /path/to/Edit3D-Bench/data \
    --pred_root /path/to/exp_comparison/image_prompt_to_prompt \
    --metrics psnr ssim lpips fid dino_if fvd chamfer clip_t \
    --device cuda:0 \
    --output_dir evaluation_results
```

详见 `EVALUATION_GUIDE.md`。

## 代码接口

也可以在 Python 代码中使用：

```python
from pathlib import Path
from editing.common.eval_output_layout import build_eval_output_layout

# 构建输出布局
layout = build_eval_output_layout(
    method_name="image_prompt_to_prompt",
    dataset="GSO",
    object_name="Toy_Poodle",
    prompt_id=1,
    eval_root="exp_comparison"
)

# 访问路径
print(layout.edit_glb)        # exp_comparison/image_prompt_to_prompt/GSO/Toy_Poodle/prompt_1/edit.glb
print(layout.images_dir)      # exp_comparison/image_prompt_to_prompt/GSO/Toy_Poodle/prompt_1/images
print(layout.videos_dir)      # exp_comparison/image_prompt_to_prompt/GSO/Toy_Poodle/prompt_1/videos
```

## 常见问题

### 1. 找不到编辑结果

**问题**：脚本报告 `[SKIP] No editing result found`

**解决方案**：
- 检查 `outputs/{method_name}/` 目录是否存在
- 确认案例名称是否包含物体名称
- 使用 `--dry-run` 查看脚本尝试查找的路径

### 2. 目录结构不匹配

**问题**：评测系统无法找到文件

**解决方案**：
- 确保 `{dataset}/{object_name}/prompt_{prompt_id}/` 结构完全匹配 Ground Truth
- 检查 `metadata.json` 中的数据集和物体名称
- 使用绝对路径而非相对路径

### 3. 批量处理失败

**问题**：部分案例处理失败

**解决方案**：
- 查看日志中的 `[SKIP]` 和 `[ERROR]` 消息
- 使用 `--object` 和 `--prompt-id` 单独处理失败的案例
- 检查 GLB 文件是否存在且完整

## 相关文档

- `EVALUATION_GUIDE.md` - Edit3D-Bench 评测系统完整指南
- `editing/common/eval_output_layout.py` - 输出布局代码实现
- `batch_prepare_eval.py` - 批处理脚本源码
