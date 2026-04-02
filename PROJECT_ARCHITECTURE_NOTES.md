# TRELLIS 项目架构与注意事项

这份笔记面向后续查阅，目标不是完整覆盖全部实现细节，而是帮助快速回答这几个问题：

- 入口在哪里
- 图像 / 文本 / 编辑链路分别怎么走
- 改功能时该先看哪个目录
- 哪些依赖、环境变量、导出步骤最容易踩坑

## 1. 仓库总览

### 核心运行时目录

- `trellis/pipelines/`
  - 顶层推理入口。
  - `base.py` 负责从 `pipeline.json` 加载各子模型。
  - `trellis_image_to_3d.py` 串起图像预处理、DINO 条件编码、稀疏结构采样、SLat 采样和多格式解码。
  - `trellis_text_to_3d.py` 串起文本条件编码、稀疏结构采样、SLat 采样、变体生成等。

- `trellis/pipelines/samplers/`
  - 采样器实现。
  - 当前主线是 Flow Matching 的 Euler sampler。
  - `FlowEulerGuidanceIntervalSampler` 是大多数示例里实际走到的 CFG + interval 版本。

- `trellis/models/`
  - 模型主体定义。
  - 主要包括 sparse structure flow、structured latent flow、以及 structured latent VAE 的各类 encoder / decoder。

- `trellis/modules/`
  - 更底层的网络积木。
  - 包括 dense transformer、sparse transformer、dense/sparse attention、稀疏张量基础结构等。
  - 做 cross-attention patch、attention tracing、prompt-to-prompt 时，经常会直接碰这里的模块类型。

- `trellis/representations/`
  - 最终 3D 表示。
  - 包括 `gaussian/`、`radiance_field/`、`mesh/`、`octree/`。
  - Mesh 分支里还带了 FlexiCubes 相关实现。

- `trellis/renderers/`
  - 渲染后端。
  - 主要被 `trellis/utils/render_utils.py` 调用，用于 snapshot / video 导出。

- `trellis/utils/`
  - 高层工具函数。
  - `render_utils.py` 负责预览图和视频渲染。
  - `postprocessing_utils.py` 负责 mesh 后处理、贴图烘焙、GLB 导出，是重依赖最集中的地方。

### 训练与数据目录

- `configs/`
  - 训练 / 生成配置。
  - `configs/generation/` 对应推理期 flow sampler 和模型配置。
  - `configs/vae/` 对应各类 VAE 训练配置。

- `trellis/trainers/`
  - 训练端逻辑，分 VAE 和 flow matching 两大块。

- `trellis/datasets/` 与 `dataset_toolkits/`
  - 数据集封装与离线处理脚本。
  - 需要做数据准备、体素化、特征抽取、latent 编码时从这里切入。

### 根目录下的重要脚本

- `example*.py`
  - 面向推理、调试、分析、编辑的示例脚本。

- `visualize_variant_intermediates.py`
  - 变体编辑链路的中间结果导出。

- `app.py` / `app_text.py`
  - Gradio Demo。

- `cross_attention_viewer.html`
  - 浏览文本 cross-attention trace 的静态页面。

## 2. 两条主推理链路

### 图像到 3D

入口：`trellis/pipelines/trellis_image_to_3d.py`

主流程：

1. `Pipeline.from_pretrained()` 读取 `pipeline.json`，装配子模型。
2. `_init_image_cond_model()` 通过 `torch.hub` 加载 DINOv2。
3. `preprocess_image()`
   - 若图片已有有效 alpha，直接用 alpha。
   - 否则调用 `rembg` 去背景。
   - 裁剪前景后 resize 到 `518x518`。
4. `get_cond()`
   - 编出 `cond` 和全零 `neg_cond`。
5. `sample_sparse_structure()`
   - 从 dense 3D latent 噪声开始采样。
   - 经过 occupancy decoder 变成 `64^3` 稀疏坐标。
6. `sample_slat()`
   - 在稀疏坐标上采样 structured latent。
   - 再做均值/方差反归一化。
7. `decode_slat()`
   - 输出 mesh / gaussian / radiance_field 三类表示。

关键 shape：

- 图像条件 token：通常是 `[B, 1370, 1024]`
  - `1370 = 1 + 37 * 37`
- 稀疏结构 latent：底层 dense 分辨率通常是 `16^3`
- occupancy 解码后坐标：落到 `64^3` 体素空间

### 文本到 3D

入口：`trellis/pipelines/trellis_text_to_3d.py`

主流程和图像版基本一致，主要差别在条件编码：

1. `_init_text_cond_model()` 加载 CLIP Text Model 和 tokenizer。
2. `encode_text()` 编出长度上限 77 的 token 序列嵌入。
3. `get_cond()` 返回 `cond` 和 `null_cond`。
4. 后续仍然是：
   - sparse structure 采样
   - SLat 采样
   - 多格式解码

额外接口：

- `run_variant()`
  - 从已有 mesh 出发做变体生成。
- `voxelize()`
  - 把 mesh 归一化后体素化，用于 variant / inversion 一类流程。

## 3. 编辑与分析脚本是怎么挂进去的

这类脚本的共同特点是：尽量不改 TRELLIS 核心源码，而是走“运行时 patch”。

### Prompt-to-Prompt / Attention Trace

相关脚本：

- `example_text_prompt_to_prompt.py`
- `example_image_prompt_to_prompt.py`
- `example_image_prompt_to_prompt_rf_inversion.py`
- `example_image_uniedit_rf_inversion.py`
- `example_text_cross_attention.py`
- `example_image_cross_attention.py`

常见做法：

- 在导入 TRELLIS 前先设置 `ATTN_BACKEND`
- monkey patch `MultiHeadAttention` / `SparseMultiHeadAttention` 或各 flow model 的 `forward`
- 在 `try/finally` 里挂 patch / 恢复 patch
- 只改 cross-attention 注入和 tracing，不改主模型权重

如果后面还要继续做编辑实验，优先从这些脚本里复用逻辑，不建议直接把试验代码埋到 `trellis/` 正式模块里。

## 4. 当前示例输出约定

默认示例导出现在统一按：

`outputs/<method_name>/<case_name>/...`

例如：

- `outputs/image_to_3d/<case>/`
- `outputs/text_cross_attention/<case>/`
- `outputs/text_prompt_to_prompt/<case>/edit/`
- `outputs/text_prompt_to_prompt/<case>/source_original/`

通用 helper 在：

- `output_layout.py`

如果后面再加新示例，优先沿用这套目录结构，不要再回到混用：

- 当前目录直出
- `output/<case>/...`
- `output/<case>/<method>/...`

## 5. 高概率踩坑点

### 1. `ATTN_BACKEND` 必须在 import TRELLIS 前决定

尤其是这些脚本：

- prompt-to-prompt
- cross-attention trace
- 一些需要 sparse attention 的文本 / 图像编辑脚本

原因：

- TRELLIS 会在 import 阶段读取后端相关环境变量。
- 先 import 再改 `ATTN_BACKEND`，通常已经晚了。

### 2. `SPCONV_ALGO=native` 是为了避免首次 benchmark

很多示例脚本都主动设了这个环境变量。

影响：

- `auto` 有时更快
- 但只跑一次示例时，`native` 更稳、更省等待

### 3. 图像预处理默认依赖 `rembg`

如果输入图没有有效 alpha，`preprocess_image()` 会走 `rembg` 去背景。

因此：

- 最稳的是直接给带透明背景的 RGBA 图
- 否则就要保证环境里有 `rembg`

### 4. GLB 导出比“只生成结果”更吃依赖、更容易爆显存

`trellis/utils/postprocessing_utils.py` 里依赖比较重，包括：

- `nvdiffrast`
- `xatlas`
- `pyvista`
- `pymeshfix`
- `igraph`
- `opencv`
- `utils3d`

而且 GLB 导出涉及：

- mesh 后处理
- 多视角渲染
- 纹理烘焙

所以如果“生成成功但导出挂了”，通常要先分清：

- 模型采样挂了
- 还是后处理 / 导出挂了

### 5. Radiance Field 预览比 mesh / gaussian 更容易在渲染阶段出问题

常见现象：

- OOM
- 特定 CUDA / rasterizer 组合不兼容

所以很多编辑脚本都预留了：

- `--skip-render`
- `--skip-radiance-field-render`

### 6. 文本条件和图像条件的长度、语义不一样

- 文本侧一般是 CLIP token，长度上限 77
- 图像侧一般是 DINO patch token，常见为 `1 + 37 * 37`

做 attention patch / trace 时不要把两者当成同一套 token 体系。

### 7. 变体 / inversion / UniEdit 脚本往往需要“资产目录”而不是普通导出文件

尤其是 RF inversion 相关流程，经常需要：

- `voxels.ply`
- `features.npz`

仅有：

- `sample.glb`
- `sample.ply`

通常不够恢复完整的 inversion 初始化。

## 6. 后续排查问题时的建议路线

### 想看主推理链路

按这个顺序看：

1. `trellis/pipelines/base.py`
2. `trellis/pipelines/trellis_image_to_3d.py`
3. `trellis/pipelines/trellis_text_to_3d.py`
4. `trellis/pipelines/samplers/flow_euler.py`

### 想看导出和渲染

按这个顺序看：

1. `trellis/utils/render_utils.py`
2. `trellis/utils/postprocessing_utils.py`
3. `trellis/renderers/`
4. `trellis/representations/`

### 想看编辑实验

按这个顺序看：

1. `example_text_prompt_to_prompt.py`
2. `example_image_prompt_to_prompt.py`
3. `example_image_prompt_to_prompt_rf_inversion.py`
4. `example_image_uniedit_rf_inversion.py`

### 想看分析与可视化

按这个顺序看：

1. `example_annotated.py`
2. `example_visualize_stages.py`
3. `example_text_cross_attention.py`
4. `example_image_cross_attention.py`
5. `visualize_variant_intermediates.py`

## 7. 一句话总结

TRELLIS 的核心不是“直接从 prompt 出 mesh”，而是：

先生成 sparse structure，再生成 structured latent，最后把同一个 SLat 解码成多种 3D 表示。

后续如果要改功能，先判断自己动的是哪一层：

- 条件编码
- sparse structure 采样
- SLat 采样
- 解码 / 导出
- 运行时 patch 的编辑或分析逻辑
