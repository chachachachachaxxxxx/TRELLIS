# AGENTS.md

本文件是 TRELLIS 仓库内给代码代理使用的项目说明草案。目标是帮助代理快速定位改动位置、选择合适的验证方式，并避开这个项目里最容易踩的坑。

## 1. 项目是什么

- TRELLIS 是一个基于 Python / PyTorch 的 3D 生成项目，支持 text-to-3D、image-to-3D、多图条件生成，以及若干编辑/分析脚本。
- 主要运行模式有三类：
  - 推理与导出：`example*.py`、`trellis/pipelines/`、`trellis/utils/`
  - Demo：`app.py`、`app_text.py`
  - 训练与数据：`train.py`、`trellis/trainers/`、`trellis/datasets/`、`dataset_toolkits/`
- 预训练模型默认从 Hugging Face 加载，不在仓库内。

## 2. 先看哪里，再改哪里

- 改图像到 3D 主链路：先看 `trellis/pipelines/trellis_image_to_3d.py`
- 改文本到 3D 主链路：先看 `trellis/pipelines/trellis_text_to_3d.py`
- 改采样逻辑、CFG、interval guidance：先看 `trellis/pipelines/samplers/`
- 改训练流程：先看 `train.py`、`trellis/trainers/`
- 改数据读取或离线处理：先看 `trellis/datasets/`、`dataset_toolkits/`
- 改渲染、GLB 导出、mesh 后处理：先看 `trellis/utils/render_utils.py`、`trellis/utils/postprocessing_utils.py`、`trellis/renderers/`
- 改底层网络、attention、sparse module：先看 `trellis/models/`、`trellis/modules/`

如果只是做研究实验、attention tracing、prompt-to-prompt 或运行时 patch，优先复用现有 `example_*` 脚本的 monkey patch 方式，不要第一反应就把实验逻辑写进 `trellis/` 正式模块。

## 3. 仓库地图

- `trellis/pipelines/`: 推理入口与高层编排
- `trellis/models/`: 模型主体
- `trellis/modules/`: 更底层的 transformer / attention / sparse 组件
- `trellis/representations/`: Gaussian / Radiance Field / Mesh / Octree 表示
- `trellis/renderers/`: 渲染后端
- `trellis/utils/`: 渲染、后处理、通用工具
- `trellis/trainers/`: 训练实现
- `trellis/datasets/`: 训练期数据集封装
- `dataset_toolkits/`: 数据准备与离线处理脚本
- `configs/generation/`: 生成配置
- `configs/vae/`: VAE 训练配置
- `example*.py`: 推理、编辑、调试、分析示例

注意：`trellis/representations/mesh/flexicubes` 是 git submodule。除非任务明确要求修改 submodule，否则不要随手改这里。

## 4. 环境与依赖假设

- 官方 README 以 Linux + NVIDIA GPU 为主，通常需要至少 16GB 显存。
- 当前仓库协作默认使用已有 conda 环境 `hammer`。除非任务明确要求切换环境，否则代理应优先在 `hammer` 环境下运行命令、验证脚本和复现实验。
- 如果需要给出命令示例，默认可假设先执行：

```bash
conda activate hammer
```

- 推荐安装路径以 `README.md` 和 `setup.sh` 为准；仓库里虽然有 `environment.yml`，但看起来不是唯一或最权威的入口。
- 常用安装命令：

```bash
. ./setup.sh --new-env --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
```

- 如果只做 Demo，可优先使用：

```bash
. ./setup.sh --demo
```

- 对不支持 `flash-attn` 的 GPU，可改用 `xformers`，但要在导入 TRELLIS 之前设置环境变量。

## 5. 项目特有约束

### 5.1 环境变量必须尽早设置

下面这些变量应当在 `import trellis` 或导入相关模块之前设置：

- `ATTN_BACKEND`
- `SPARSE_ATTN_BACKEND`
- `SPCONV_ALGO`

原因是部分后端选择在 import 阶段就会读取环境变量。先 import 再设置，通常太晚。

### 5.2 一次性示例更适合 `SPCONV_ALGO=native`

很多示例都显式设置了：

```python
os.environ["SPCONV_ALGO"] = "native"
```

`auto` 可能更快，但会先做 benchmark。对一次性运行或 smoke test，`native` 更稳。

### 5.3 图像预处理可能依赖去背景

- `trellis/pipelines/trellis_image_to_3d.py` 的 `preprocess_image()` 会在没有有效 alpha 时走 `rembg`
- 它还会裁剪前景并 resize 到固定尺寸
- 如果在调图像链路，优先使用带透明背景的 RGBA 输入，能少掉很多环境和结果不确定性

### 5.4 “生成成功”不等于“导出成功”

GLB 导出依赖较重，且显存压力更大。出现问题时，要先区分：

- 模型采样失败
- 渲染失败
- mesh 后处理失败
- 贴图烘焙 / GLB 导出失败

不要把这几类问题混成一个问题处理。

### 5.5 输出目录尽量保持统一

现有示例倾向于使用：

```text
outputs/<method_name>/<case_name>/
```

新增示例或分析脚本时，优先沿用这套结构，并复用 `output_layout.py` 的约定。

### 5.6 不要把生成产物当源码提交

仓库当前的 `.gitignore` 主要不是围绕 Python 训练产物写的，所以代理要主动避免提交这些内容：

- `outputs/`
- `tmp/`
- 下载的模型权重
- checkpoint
- 大体积导出文件，如 `.glb`、`.ply`、视频和中间可视化结果

## 6. 修改策略

- 尽量做小而准的改动，不要顺手大面积重构。
- 这个仓库没有看到成型的 `tests/` 目录，也没有明显的统一格式化配置；改代码时要跟随目标文件现有风格，不要顺手做全仓格式清洗。
- 做反演算法（inversion）相关实验时，默认优先在 `example_*`、独立实验脚本、wrapper 或 monkey patch 中实现，尽量不要直接修改 `trellis/` 下的正式实现。
- 只有在外部封装、继承、组合或运行时 patch 都无法满足需求时，才考虑最小化修改 `trellis/`，并在说明中写清原因与影响面。
- 如果改的是共享底层模块，如 `trellis/modules/` 或 `trellis/models/`，要意识到它可能同时影响推理、训练和编辑脚本。
- 如果改的是 Demo，注意 `app.py` / `app_text.py` 会持有 GPU 资源，并会在仓库根目录下使用 `tmp/` 作为会话目录。
- 如果改的是示例脚本，尽量保留现有输出路径约定和命令行接口风格。
- 文档语言尽量跟随被修改文件本身：README 和大部分公开文档偏英文，项目笔记和局部分析文档允许中文。

## 7. 推荐验证方式

这个仓库目前更适合做“分层验证”，而不是默认寻找单元测试。

### 7.1 纯 Python 结构改动

先做轻量检查：

```bash
python -m compileall trellis app.py app_text.py train.py
```

### 7.2 推理链路改动

按影响范围选择最小示例：

```bash
conda activate hammer
python example.py
python example_text.py
python example_multi_image.py
python example_variant.py
```

如果只是验证 import、对象装配或参数流转，可以优先减少采样步数，先做短链路 smoke test。

### 7.3 Demo 改动

```bash
conda activate hammer
python app.py
python app_text.py
```

至少确认界面能启动、核心回调没有明显参数错误。

### 7.4 训练改动

优先使用 dry run：

```bash
conda activate hammer
python train.py \
  --config configs/vae/slat_vae_dec_mesh_swin8_B_64l8_fp16.json \
  --output_dir outputs/tryrun \
  --data_dir /path/to/data \
  --tryrun
```

如果任务只改了配置装配、trainer 初始化、分布式入口或 checkpoint 恢复逻辑，`--tryrun` 是首选验证手段。

## 8. 给代理的工作原则

- 先定位到正确入口，再动手改代码。
- 先判断问题属于推理、训练、导出、渲染还是实验脚本，不要混改。
- 优先复用现有脚本和工具函数，不要重复造一套并行工作流。
- 未经明确需要，不修改 submodule，不引入新的重量级依赖，不提交生成产物。
- 做完改动后，明确写出“运行了什么验证”和“没有运行什么验证”。
