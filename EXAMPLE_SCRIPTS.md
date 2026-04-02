# TRELLIS Example 脚本说明

这份文档按用途整理了仓库根目录下的 `example*.py` 脚本，重点说明：

- 这个脚本是干什么的
- 什么时候该用它
- 怎么运行
- 会产出什么文件

如果你只是想先跑通一次，建议先看 [example.py](example.py) 和 [example_text.py](example_text.py)。

## 先看这个：怎么选脚本

- 只想从一张图片生成 3D：`example.py`
- 只想从文本直接生成 3D：`example_text.py`
- 想用多张图片共同约束一个 3D：`example_multi_image.py`
- 已经有一个 mesh，想做“同一物体的风格/材质变体”：`example_variant.py`
- 想理解 TRELLIS 每个阶段到底做了什么：`example_annotated.py` 或 `example_visualize_stages.py`
- 想先用 FLUX 出参考图，再交给 TRELLIS 转 3D：`example_flux_krea_to_3d.py`
- 想看文本 token 是怎么影响 3D 空间的：`example_text_cross_attention.py`
- 想看图像 patch 是怎么影响 3D 空间的：`example_image_cross_attention.py`
- 想做文本 Prompt-to-Prompt 3D 编辑：`example_text_prompt_to_prompt.py`
- 想做图像局部编辑后再转 3D：`example_image_prompt_to_prompt.py`
- 想基于已有 3D 资产做更稳的图像编辑，并用 RF inversion 初始化：`example_image_prompt_to_prompt_rf_inversion.py`

## 通用运行说明

- 默认都在仓库根目录运行：

```bash
cd TRELLIS
```

- 先按 [README.md](README.md) 安装依赖并激活环境。大多数脚本都要求 CUDA GPU。
- 大部分脚本内部都会设置 `SPCONV_ALGO=native`，避免首次 benchmark。
- 一部分高级脚本依赖 sparse attention。如果你的 GPU 不适合 `flash-attn`，可以改用 `xformers`。
  - 支持 `--attn-backend` 的脚本，直接传 `--attn-backend xformers`
  - 不支持该参数的简单脚本，可以在运行前手动：

```bash
export ATTN_BACKEND=xformers
```

- 带 `argparse` 的脚本都可以先看帮助：

```bash
python xxx.py --help
```

## 一览表

- 默认示例导出现在统一按 `outputs/<method_name>/<case_name>/...` 组织；部分脚本仍保留 `--output-dir` 作为显式覆盖。

| 脚本 | 类型 | 主要用途 | 输入方式 | 主要输出 |
| --- | --- | --- | --- | --- |
| `example.py` | 基础推理 | 单张图片转 3D | 脚本内写死图片路径 | `outputs/image_to_3d/<case>/...` |
| `example_text.py` | 基础推理 | 文本转 3D | 脚本内写死 prompt | `outputs/text_to_3d/<case>/...` |
| `example_multi_image.py` | 基础推理 | 多图条件生成 | 脚本内写死图片列表 | `outputs/multi_image_to_3d/<case>/...` |
| `example_variant.py` | 基础推理 | 从已有 mesh 生成变体 | 脚本内写死 mesh + prompt | `outputs/variant_to_3d/<case>/...` |
| `example_annotated.py` | 教学脚本 | 手动拆解整条 image-to-3D 流程 | 脚本内写死图片路径 | `outputs/annotated_image_to_3d/<case>/...` |
| `example_visualize_stages.py` | 可视化/调试 | 导出每个阶段的中间结果 | 命令行位置参数 | `outputs/visualize_stages/<case_name>/...` |
| `example_flux_krea_to_3d.py` | 串联推理 | 先文生图，再图生 3D | 命令行参数 | `outputs/flux_krea_to_3d/<case>/...` |
| `example_text_cross_attention.py` | 分析脚本 | 追踪文本 token 的 cross-attention | 命令行参数 | `outputs/text_cross_attention/<case>/...` |
| `example_image_cross_attention.py` | 分析脚本 | 追踪图像 patch 的 cross-attention | 命令行参数 | `outputs/image_cross_attention/<case>/...` |
| `example_text_prompt_to_prompt.py` | 编辑脚本 | 文本 Prompt-to-Prompt 3D 编辑 | 命令行参数 | `outputs/text_prompt_to_prompt/<case>/...` |
| `example_image_prompt_to_prompt.py` | 编辑脚本 | 基于 source/edit/mask 的图像局部编辑 | 命令行参数 | `outputs/image_prompt_to_prompt/<case>/...` |
| `example_image_prompt_to_prompt_rf_inversion.py` | 编辑脚本 | 基于已有 3D 资产 + 图像编辑的稳态编辑 | 命令行参数 | `outputs/image_prompt_to_prompt_rf_inversion/<case>/...` |

## 1. 基础生成脚本

### `example.py`

- 用途：最小的 image-to-3D 示例，最适合确认环境和模型能不能正常跑通。
- 默认输入：`assets/3D_Dollhouse_Happy_Brother_p1.png`
- 最小运行：

```bash
python example.py
```

- 输出：
  - `outputs/image_to_3d/3d_dollhouse_happy_brother_p1/sample_gs.mp4`
  - `outputs/image_to_3d/3d_dollhouse_happy_brother_p1/sample_rf.mp4`
  - `outputs/image_to_3d/3d_dollhouse_happy_brother_p1/sample_mesh.mp4`
  - `outputs/image_to_3d/3d_dollhouse_happy_brother_p1/sample.glb`
  - `outputs/image_to_3d/3d_dollhouse_happy_brother_p1/sample.ply`
- 适合什么时候用：
  - 第一次验证安装是否成功
  - 想看最小调用范式：`from_pretrained -> run -> render/export`
- 备注：
  - 如果要换图片，直接改脚本里的 `Image.open("assets/example_image/T.png")`
  - 这是“最少代码”的示例，不是通用 CLI 工具

### `example_text.py`

- 用途：最小的 text-to-3D 示例。
- 默认模型：`microsoft/TRELLIS-text-xlarge`
- 最小运行：

```bash
python example_text.py
```

- 当前脚本里 prompt 是写死的，输出目录名也是写死的：
  - prompt 在 `pipeline.run(...)` 的第一个参数
  - 样例名在 `sample_name = "blue_blueberries"`
- 输出：
  - `outputs/text_to_3d/blue_blueberries/sample_gs.mp4`
  - `outputs/text_to_3d/blue_blueberries/sample_rf.mp4`
  - `outputs/text_to_3d/blue_blueberries/sample_mesh.mp4`
  - `outputs/text_to_3d/blue_blueberries/sample.glb`
  - `outputs/text_to_3d/blue_blueberries/sample.ply`
- 适合什么时候用：
  - 想看 TRELLIS 文本模型的最小调用方式
  - 想快速试验少量 prompt
- 备注：
  - 文本模型更适合做研究或功能验证；README 也明确建议高质量产出优先走“先文生图，再图生 3D”

### `example_multi_image.py`

- 用途：多张参考图共同约束一个 3D 结果。
- 默认输入：
  - `assets/example_multi_image/character_1.png`
  - `assets/example_multi_image/character_2.png`
  - `assets/example_multi_image/character_3.png`
- 最小运行：

```bash
python example_multi_image.py
```

- 输出：
  - `outputs/multi_image_to_3d/character_triplet/sample_multi.mp4`
- 适合什么时候用：
  - 单张图视角信息不够时
  - 多张输入图已经是同一个物体的不同视角/不同参考图
- 备注：
  - 核心接口是 `pipeline.run_multi_image(...)`
  - 如果要换图片，改 `images = [...]` 列表即可

### `example_variant.py`

- 用途：从已有 mesh 出发，生成同一物体的风格/外观变体。
- 默认输入：
  - mesh：`assets/T.ply`
  - 文本描述：`"Rugged, metallic texture with orange and white paint finish, ..."`
- 最小运行：

```bash
python example_variant.py
```

- 输出：
  - `outputs/variant_to_3d/t_variant/sample_variant.mp4`
- 适合什么时候用：
  - 你已经有一个基础几何
  - 想保留结构，改材质、表面观感、风格
- 备注：
  - 核心接口是 `pipeline.run_variant(...)`
  - 依赖 `open3d`

## 2. 教学与阶段可视化脚本

### `example_annotated.py`

- 用途：把 image-to-3D 整条链路拆成“预处理 -> 条件编码 -> 稀疏结构采样 -> SLat 采样 -> 多格式解码 -> 导出”逐步手写出来，并写了大量中文注释。
- 默认输入：`assets/example_image/T.png`
- 最小运行：

```bash
python example_annotated.py
```

- 输出：
  - `outputs/annotated_image_to_3d/t/sample_gs.mp4`
  - `outputs/annotated_image_to_3d/t/sample_rf.mp4`
  - `outputs/annotated_image_to_3d/t/sample_mesh.mp4`
  - `outputs/annotated_image_to_3d/t/sample.glb`
  - `outputs/annotated_image_to_3d/t/sample.ply`
- 适合什么时候用：
  - 想学习 TRELLIS 推理流程
  - 想自己改 pipeline，而不是只会调 `pipeline.run(...)`
- 备注：
  - 这是“教学版代码”，不是为了做成 CLI
  - 你可以把它和 `trellis/pipelines/trellis_image_to_3d.py` 对照看

### `example_visualize_stages.py`

- 用途：把每个阶段的输入、中间结果、统计信息和最终导出全部保存下来，适合调试和论文流程理解。
- 最小运行：

```bash
python example_visualize_stages.py
```

- 也可以指定输入、case 和 seed：

```bash
python example_visualize_stages.py assets/example_image/2d_edit.png demo_case 1
```

- 参数方式：
  - 第 1 个位置参数：`image_path`
  - 第 2 个位置参数：`case_name`
  - 第 3 个位置参数：`seed`
- 默认输入：`assets/example_image/2d_edit.png`
- 输出目录：
  - `outputs/visualize_stages/<case_name>/`
- 典型输出：
  - `1_output_preprocessed_518x518.png`
  - `2_output_cond_features_pca.png`
  - `3_output_sparse_coords_3d.png`
  - `4_output_slat_features_pca.png`
  - `5_output_mesh_snapshot.png`
  - `6_output_video_mesh.mp4`
  - `6_output_textured_mesh.glb`
  - `README.txt`
- 适合什么时候用：
  - 想定位哪个阶段出了问题
  - 想观察特征、occupancy、SLat、解码结果
  - 想给别人展示 TRELLIS 每一步都在做什么
- 备注：
  - 相比 `example_annotated.py`，这个脚本更偏“自动导出调试资料”

## 3. 串联推理脚本

### `example_flux_krea_to_3d.py`

- 用途：先用 `FLUX.1 Krea [dev]` 从文本生成参考图，再把这张图喂给 `TRELLIS-image-large` 转 3D。
- 最小运行：

```bash
python example_flux_krea_to_3d.py "a plate of strawberries with whipped cream"
```

- 常用写法：

```bash
python example_flux_krea_to_3d.py \
  "a plate of strawberries with whipped cream" \
  --output-dir outputs/flux_krea_to_3d/strawberries
```

- 输出目录默认是：
  - `outputs/flux_krea_to_3d/strawberries/`
- 主要输出：
  - `reference.png`
  - `reference_preprocessed.png`
  - `sample_gs.mp4`
  - `sample_rf.mp4`
  - `sample_mesh.mp4`
  - `sample.glb`
  - `sample.ply`
  - `prompt.txt`
- 适合什么时候用：
  - 想遵循 README 推荐路线：先文生图，再图生 3D
  - 想把文本的创造力交给 2D 生成模型，把 3D 重建交给 TRELLIS
- 额外前置要求：
  - 先接受 FLUX 模型 license
  - 先登录 Hugging Face
  - 安装额外依赖：

```bash
pip install -U diffusers accelerate sentencepiece
```

## 4. Cross-Attention 追踪脚本

### `example_text_cross_attention.py`

- 用途：追踪文本 token 在 TRELLIS 生成过程中是如何通过 cross-attention 影响 3D 空间的。
- 最小运行：

```bash
python example_text_cross_attention.py --prompt "a red chair"
```

- 常用写法：

```bash
python example_text_cross_attention.py \
  --prompt "a red chair" \
  --focus-words chair,red
```

- 输出目录：
  - `outputs/text_cross_attention/<case_name>/`
- 主要输出：
  - `prompt_tokens.json`
  - `trace_manifest.json`
  - `sparse_structure_coords.npy`
  - `slat_coords.npy`
  - 各 stage 下的 heatmap、curve、raw map、token 局部分析图
  - `README.txt`
- 适合什么时候用：
  - 想研究文本 token 和空间结构之间的对应关系
  - 想比较不同词在 sparse_structure / slat 阶段的影响
- 常用参数：
  - `--trace-stages sparse_structure,slat`
  - `--focus-words chair,red`
  - `--topk-focus-tokens 4`
  - `--topk-spatial 4`
- 备注：
  - 这个脚本以“分析”为主，不是为了导出最终 mesh/glb

### `example_image_cross_attention.py`

- 用途：追踪图像 patch token 与 3D 空间 token 的 cross-attention，对应查看 mask 选中的图像区域主要影响了哪里。
- 默认输入目录：`assets/example_edit`
- 最小运行：

```bash
python example_image_cross_attention.py
```

- 也可以显式指定：

```bash
python example_image_cross_attention.py \
  --images assets/example_edit/2d_render.png,assets/example_edit/2d_edit.png \
  --mask assets/example_edit/2d_mask.png
```

- 输出目录：
  - `outputs/image_cross_attention/<case_name>/`
- 主要输出：
  - `case_manifest.json`
  - `crop_context.json`
  - `index.html`
  - 每个子 case 目录下的：
    - `preprocessed_image.png`
    - `preprocessed_mask.png`
    - `mask_patch_overlay.png`
    - `image_tokens.json`
    - `trace_manifest.json`
    - 各 stage 的 masked-region 平均激活图、step/block 进度图
- 适合什么时候用：
  - 想分析局部图像编辑到底影响了 3D 哪些区域
  - 想确认 mask 选中的 patch 是否合理
- 备注：
  - 默认会把 `input-dir` 中名字带 `mask` 的图当作掩码，其余图当作条件图
  - 脚本最后会生成一个静态 `index.html`，建议用本地 HTTP 服务打开

```bash
python -m http.server 8000
```

## 5. Prompt-to-Prompt 编辑脚本

- 这几种编辑脚本现在统一按 `outputs/<method_name>/<case_name>/...` 组织输出，方便不同编辑方法并排对比。

### `example_text_prompt_to_prompt.py`

- 用途：对文本 prompt 做 Prompt-to-Prompt 编辑，在不改 TRELLIS 核心源码的前提下，通过运行时 patch cross-attention 实现结构保留式编辑。
- 最小运行：

```bash
python example_text_prompt_to_prompt.py \
  --source-prompt "a cute cat statue" \
  --edit-prompt "a cute tiger statue" \
  --case-name cat_to_tiger
```

- 输出目录：
  - 编辑结果：`outputs/text_prompt_to_prompt/<case_name>/edit/`
  - 如果不加 `--skip-source`，还会有：`outputs/text_prompt_to_prompt/<case_name>/source_original/`
- 主要输出：
  - `config.json`
  - `source_tokens.json`
  - `edit_tokens.json`
  - `token_alignment.json`
  - `sample_00_gs.mp4`
  - `sample_00_rf.mp4`
  - `sample_00_mesh.mp4`
  - `sample_00.glb`
  - `sample_00.ply`
- 适合什么时候用：
  - 想把“红椅子”改成“蓝椅子”
  - 想把“猫雕像”改成“虎雕像”
  - 想观察只替换部分 token 时结构保持得如何
- 最常用参数：
  - `--inject-stages st,slat`
  - `--ss-t-start / --ss-t-end`
  - `--slat-t-start / --slat-t-end`
  - `--ss-strength / --slat-strength`
  - `--skip-source`
  - `--skip-render`
  - `--skip-radiance-field-render`
- 备注：
  - 更深入的内部说明可以看仓库里的 [text_prompt_to_prompt.md](text_prompt_to_prompt.md)

### `example_image_prompt_to_prompt.py`

- 用途：对 source 图做局部图像编辑，再借助 Prompt-to-Prompt 的 cross-attention 注入，让 3D 结果尽量保留未编辑区域。
- 最小运行：

```bash
python example_image_prompt_to_prompt.py \
  --source-image assets/example_edit/2d_render.png \
  --edit-image assets/example_edit/2d_edit.png \
  --mask-image assets/example_edit/2d_mask.png \
  --case-name image_local_edit
```

- 输出目录：
  - 编辑结果：`outputs/image_prompt_to_prompt/<case_name>/edit/`
  - 如果不加 `--skip-source`，还会有：`outputs/image_prompt_to_prompt/<case_name>/source_original/`
- 主要输出：
  - `config.json`
  - `input_preprocess.json`
  - `token_metadata.json`
  - `source_preprocessed.png`
  - `edit_preprocessed.png`
  - `mask_preprocessed.png`
  - `mask_token_overlay.png`
  - `edited_patch_grid.png`
  - `sample_00_gs.mp4`
  - `sample_00_rf.mp4`
  - `sample_00_mesh.mp4`
  - `sample_00.glb`
  - `sample_00.ply`
- 适合什么时候用：
  - 已经有 source 图和 edit 图
  - 只希望 mask 内部发生变化，mask 外尽量继承原结果
- 最常用参数：
  - `--preprocess` / `--no-preprocess`
  - `--mask-threshold`
  - `--patch-coverage-threshold`
  - `--inject-stages`
  - `--skip-source`
- 备注：
  - 默认会基于 source 前景、edit 前景和 mask 做共享裁剪，保证 token 网格对齐
  - 如果三张图本来就已经严格对齐，可以加 `--no-preprocess`

### `example_image_prompt_to_prompt_rf_inversion.py`

- 用途：在 `example_image_prompt_to_prompt.py` 的基础上进一步使用 RF inversion，把一个已有 3D 资产反演到 terminal noise，再做前向去噪编辑，通常比“只靠 source/edit 图”更稳。
- 适合什么时候用：
  - 你已经有一个 TRELLIS 资产目录，而不是只有一张 source 图
  - 希望编辑结果更强地继承原始 3D 结构
  - 需要和 VoxHammer 风格流程兼容
- 关键输入要求：
  - `source-model` 或 `render_dir` 指向的目录里必须有：
    - `voxels.ply`
    - `features.npz`
  - 仅有普通导出的 `sample.glb` / `sample.ply` 不够
- 最小运行示例：

```bash
python example_image_prompt_to_prompt_rf_inversion.py \
  --source-model /path/to/source_asset_dir \
  --source-image /path/to/2d_render.png \
  --edit-image /path/to/2d_edit.png \
  --mask-image /path/to/2d_mask.png \
  --output_path /path/to/output.glb
```

- 如果你已经把图片放在同一个目录里，也可以：

```bash
python example_image_prompt_to_prompt_rf_inversion.py \
  --render_dir /path/to/render_dir \
  --image_dir /path/to/image_dir \
  --output_path /path/to/output.glb
```

- 输出目录规则：
  - 不传 `--output_path` 时，默认工作目录是 `outputs/image_prompt_to_prompt_rf_inversion/<case_name>/edit/`
  - 如果给了 `--output_path /a/b/result.glb`
  - 实际工作目录会是 `outputs/image_prompt_to_prompt_rf_inversion/result/edit/`
  - 并且最终 `sample_00.glb` 会复制一份到 `/a/b/result.glb`
- 主要输出：
  - `config.json`
  - `input_preprocess.json`
  - `token_metadata.json`
  - `rf_inversion_stats.json`
  - `coords_target.npy`
  - `source_preprocessed.png`
  - `edit_preprocessed.png`
  - `mask_preprocessed.png`
  - `sample_00_gs.mp4`
  - `sample_00_rf.mp4`
  - `sample_00_mesh.mp4`
  - `sample_00.glb`
  - `sample_00.ply`
- 最常用参数：
  - `--ss-inverse-cfg`
  - `--slat-inverse-cfg`
  - `--cfg-interval-start`
  - `--cfg-interval-end`
  - `--auto-mask-threshold`
  - `--auto-mask-max-filter`
  - `--quiet`
- 备注：
  - 如果不传 `--mask-image`，脚本会根据 source/edit 的像素差自动生成 mask
  - 这是当前 examples 里最偏研究和编辑实验的脚本之一

## 一个建议的阅读/使用顺序

如果你第一次接触这些 examples，建议按下面顺序：

1. 先跑 `example.py`，确认环境没问题。
2. 再看 `example_annotated.py`，理解 TRELLIS 的主流程。
3. 想看中间结果时，用 `example_visualize_stages.py`。
4. 想做纯生成实验时，再看 `example_text.py`、`example_multi_image.py`、`example_variant.py`。
5. 想做可解释性分析时，用两个 `cross_attention` 脚本。
6. 想做编辑时，先用 `example_text_prompt_to_prompt.py` / `example_image_prompt_to_prompt.py`。
7. 想让编辑更贴近已有 3D 资产时，再上 `example_image_prompt_to_prompt_rf_inversion.py`。
