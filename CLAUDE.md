# TRELLIS 项目说明

这是一个基于 Python/PyTorch 的 3D 生成项目（text-to-3D、image-to-3D）。在此仓库工作时请遵循以下指南。

## 📋 重要文档

- **重构状态**: `docs/REFACTORING_STATUS.md` - 编辑实验框架重构的完整状态报告
- **迁移指南**: `docs/METHOD_MIGRATION_GUIDE.md` - 如何迁移和添加新方法
- **测试报告**: `docs/REFACTORING_TEST_REPORT.md` - 测试结果和验证状态
- **反演测试**: `docs/INVERSION_TESTING.md` - 反演质量测试框架说明
- **一键式评测**: `docs/ONE_CLICK_EVALUATION.md` - 一键式评测完整指南
- **评测输出格式**: `docs/EVAL_OUTPUT_FORMAT.md` - Edit3D-Bench 评测输出格式说明

## 环境

- **默认环境**: 除非明确要求，始终使用 `hammer` conda 环境
- **环境激活**: 所有脚本和命令都应在 `hammer` 环境下运行
- **脚本模板**: 创建 bash 脚本时，始终包含以下环境激活代码：
  ```bash
  #!/bin/bash
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate hammer
  export ATTN_BACKEND='flash-attn'
  export SPCONV_ALGO='native'
  ```
- **系统要求**: Linux + NVIDIA GPU，至少 16GB 显存

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

### 代码风格

- 保持代码简洁，不要提前抽象
- 不要写异常处理，不使用 `raise`、`try`、`catch`、`finally` 进行异常处理
- 让代码在遇到问题时自然失败，依赖 Python 的默认错误机制

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
| **编辑实验框架** | `editing/` - 统一的编辑方法框架 |
| **编辑方法运行** | `run_edit_experiment.py` - 统一入口 |
| **反演质量测试** | `test_inversion_quality.py` - 反演测试入口 |

## 编辑实验框架（新）

重构后的编辑实验框架位于 `editing/` 目录：

### 目录结构

```
editing/
├── methods/          # 方法类实现
│   ├── base.py      # EditMethod 抽象基类
│   ├── runner.py    # EditMethodRunner 编排器
│   ├── registry.py  # 方法注册表
│   └── *.py         # 具体方法实现
├── hooks/           # Hook 系统（Prompt-to-Prompt 等）
├── inversion/       # RF inversion 工具
├── preprocess/      # 预处理层
├── utils/           # 工具函数
├── io/              # 输入输出处理
└── common/          # 公共工具
```

### 运行编辑方法

使用统一入口 `run_edit_experiment.py`：

```bash
# Image Prompt-to-Prompt
python run_edit_experiment.py \
  --method image_prompt_to_prompt \
  --source-image path/to/source.png \
  --edit-image path/to/edit.png \
  --mask-image path/to/mask.png \
  --case-name my_edit \
  --seed 1

# Text Prompt-to-Prompt
python run_edit_experiment.py \
  --method text_prompt_to_prompt \
  --source-prompt "a cute cat statue" \
  --edit-prompt "a cute tiger statue" \
  --case-name text_edit \
  --seed 1

# SLAT XOR Fusion
python run_edit_experiment.py \
  --method image_slat_xor_fusion \
  --source-model path/to/source/assets \
  --edit-image path/to/edit.png \
  --case-name fusion_edit \
  --seed 1
```

### 添加新方法

1. 创建方法类继承 `EditMethod`
2. 实现 `prepare()`, `run()`, `save_artifacts()`, `cleanup()`
3. 在 `registry.py` 中注册
4. 自动支持统一的 CLI 和配置管理

详见 `docs/METHOD_MIGRATION_GUIDE.md`

## 验证方式

本仓库没有正式测试套件，使用分层验证：

### Python 结构改动

```bash
python -m compileall trellis editing run_edit_experiment.py
```

### 推理链路改动

运行最小相关示例，减少步数做 smoke test：

```bash
# 基础推理（已移至 trellis_inference/）
python trellis_inference/example.py
python trellis_inference/example_text.py

# 编辑方法
python run_edit_experiment.py --method image_prompt_to_prompt --case test_case --seed 1
```

### 编辑方法验证

```bash
# 测试特定方法
python run_edit_experiment.py \
  --method <method_name> \
  --source-image assets/edit_example/images/2d_render.png \
  --edit-image assets/edit_example/images/2d_edit.png \
  --mask-image assets/edit_example/images/2d_mask.png \
  --case-name test \
  --seed 1 \
  --preprocess
```

## 输出目录组织

遵循现有约定：

```
outputs/<method_name>/<case_name>/
```

新建脚本时复用 `output_layout.py` 的约定。

## Edit3D-Bench 评测

### 一键式评测（推荐）

使用 `run_batch_edit_and_eval.py` 完成：数据准备 → 运行编辑 → 渲染 → 评测

```bash
# 基本用法
python run_batch_edit_and_eval.py \
  --gt-root /home/wangxinxing/code/Edit3Dpp/data \
  --method-name image_p2p_latent_blend \
  --config-name test \
  --max-cases 1 \
  --method-args --ss-steps 25 --slat-steps 25

# 使用测试脚本
./test_batch_dollhouse_no_p2p.sh
```

**输出位置**:
- edit.glb: `/cache/wangxinxing/data/temp/{method_name}_{config_name}_{timestamp}/`
- 评测结果: `outputs/results/{method_name}_{config_name}_{timestamp}/`

详见：
- `docs/EVALUATION_QUICK_START.md` - 快速开始指南
- `docs/BATCH_EDIT_AND_EVAL.md` - 详细使用文档

## 常见陷阱

- 图像预处理可能使用 `rembg` 去背景，优先使用带透明通道的 RGBA 输入
- "生成成功" ≠ "导出成功"，要区分：模型采样失败、渲染失败、mesh 后处理失败、GLB 导出失败
- GLB 导出显存压力大，依赖较重

## 改动后

明确说明：

- 运行了什么验证
- 没有运行什么验证

## 📊 当前重构状态

### 编辑实验框架重构进度: 71% 完成

- ✅ 核心框架: 100% (预处理、方法接口、Hook、Inversion)
- ✅ 方法迁移: 5/7 (71%)
  - ✅ image_prompt_to_prompt - 测试通过
  - ⚠️ text_prompt_to_prompt - 核心逻辑验证
  - ✅ image_slat_xor_fusion - 待测试
  - ✅ image_prompt_to_prompt_rf_inversion - 待测试
  - ⚠️ image_uniedit_rf_inversion - 占位符
- ✅ 问题修复: 6/6 (100%)
- ✅ 文档更新: 100%

### 待完成工作
1. 完整实现 UniEdit 方法
2. 测试 SLAT fusion 和 RF inversion 方法
3. 优化 text 方法显存使用

详见 `docs/REFACTORING_STATUS.md`
