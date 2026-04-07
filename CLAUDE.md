# TRELLIS 项目说明

这是一个基于 Python/PyTorch 的 3D 生成项目（text-to-3D、image-to-3D）。在此仓库工作时请遵循以下指南。

## 环境

- 除非明确要求，始终使用 `hammer` conda 环境
- 运行命令前默认已执行：`conda activate hammer`
- 项目需要 Linux + NVIDIA GPU，至少 16GB 显存

## 关键规则

### 环境变量必须尽早设置

在 import trellis 或相关模块之前设置：

- `ATTN_BACKEND`
- `SPARSE_ATTN_BACKEND`
- `SPCONV_ALGO`

对于一次性示例，优先使用 `SPCONV_ALGO=native` 而非 `auto`。

### 禁止提交的内容

永远不要提交这些到 git：

- `outputs/` 目录
- `tmp/` 目录
- 模型权重
- Checkpoint
- 大体积导出文件（`.glb`、`.ply`、视频、可视化结果）

### 修改策略

- 做小而准的改动，不要大面积重构
- 研究实验（attention tracing、prompt-to-prompt、反演算法）：使用 `example_*` 脚本、wrapper 或 monkey patch。除非绝对必要，不要直接修改 `trellis/` 模块
- 跟随每个文件现有代码风格，不要做全仓格式化
- 修改共享模块（`trellis/modules/`、`trellis/models/`）时，注意会同时影响推理、训练和编辑脚本
- 不要动 `trellis/representations/mesh/flexicubes`（git submodule），除非明确要求

## 代码定位

| 任务 | 入口文件 |
|------|----------|
| 图像到 3D 主链路 | `trellis/pipelines/trellis_image_to_3d.py` |
| 文本到 3D 主链路 | `trellis/pipelines/trellis_text_to_3d.py` |
| 采样、CFG、guidance | `trellis/pipelines/samplers/` |
| 训练流程 | `train.py`、`trellis/trainers/` |
| 数据读取 | `trellis/datasets/`、`dataset_toolkits/` |
| 渲染、GLB 导出、mesh | `trellis/utils/render_utils.py`、`trellis/utils/postprocessing_utils.py`、`trellis/renderers/` |
| 网络、attention | `trellis/models/`、`trellis/modules/` |

## 验证方式

本仓库没有正式测试套件，使用分层验证：

### Python 结构改动

```bash
python -m compileall trellis app.py app_text.py train.py
```

### 推理链路改动

运行最小相关示例，减少步数做 smoke test：

```bash
python example.py
python example_text.py
python example_multi_image.py
```

### Demo 改动

```bash
python app.py
python app_text.py
```

### 训练改动

使用 dry run 模式：

```bash
python train.py --config configs/vae/slat_vae_dec_mesh_swin8_B_64l8_fp16.json --output_dir outputs/tryrun --data_dir /path/to/data --tryrun
```

## 输出目录组织

遵循现有约定：

```
outputs/<method_name>/<case_name>/
```

新建脚本时复用 `output_layout.py` 的约定。

## 常见陷阱

- 图像预处理可能使用 `rembg` 去背景，优先使用带透明通道的 RGBA 输入
- "生成成功" ≠ "导出成功"，要区分：模型采样失败、渲染失败、mesh 后处理失败、GLB 导出失败
- GLB 导出显存压力大，依赖较重

## 改动后

明确说明：

- 运行了什么验证
- 没有运行什么验证
