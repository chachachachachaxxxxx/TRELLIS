# Editing Experiment TODO

这份文档面向 TRELLIS 的免训练 3D 编辑 / 反演实验整理。目标不是重写模型，而是把当前“分散在多个研究脚本里的方法实现”收敛成一个更稳定、可比较、可复现的实验平台。

## 1. 适用范围

- 只关注免训练方法
- 优先关注 image-conditioned 编辑链路
- 包括但不限于：
  - Prompt-to-Prompt
  - RF inversion
  - UniEdit 风格方法
  - cross-attention tracing / 可视化
- 暂不覆盖训练框架重构

## 2. 当前主要问题

### P0. 缺少统一实验入口

现状：

- 方法分散在多个顶层脚本里
- 脚本之间直接互相 import，当作“半成品库”使用
- 新增方法通常只能复制脚本再改一份

影响：

- 不同方法难以共用同一套输入、预处理、seed、输出约定
- 横向比较时容易混入脚本差异

对应文件：

- `example_image_prompt_to_prompt.py`
- `example_image_prompt_to_prompt_rf_inversion.py`
- `example_image_uniedit_rf_inversion.py`
- `example_image_cross_attention.py`
- `example_text_cross_attention.py`

### P0. 预处理与方法逻辑耦合太紧

现状：

- source / edit / mask 对齐、裁剪、resize、rembg 前景提取都直接写在方法脚本中
- 同一 case 在不同脚本里不一定走完全相同的预处理路径

影响：

- 方法比较时，结果里混入预处理差异
- 很难回答“效果差异来自方法，还是来自 crop / mask / alpha 处理”

### P0. 输入资产契约不统一

现状：

- RF inversion 需要 `voxels.ply`、`features.npz`、source render
- UniEdit 又额外依赖 `mask_glb`
- 普通 TRELLIS 导出物和 inversion 所需资产并不统一

影响：

- 复现实验前要先手工猜文件组织方式
- 很难批量跑 case

### P1. 实验配置虽然有保存，但还没形成 benchmark 协议

现状：

- 已经有 `outputs/<method>/<case>/...`
- 已经有 `config.json`、`input_preprocess.json`、部分中间结果导出
- 但没有统一的 case manifest、结果总表、失败日志和批量 runner

影响：

- 单次实验可复查，多方法批量对比仍然费手工

### P1. 运行时后端选择太脆弱

现状：

- `ATTN_BACKEND` / `SPARSE_ATTN_BACKEND` / `SPCONV_ALGO` 依赖 import 前环境变量
- 不同脚本通过“先 peek CLI 参数，再设环境变量，再 import”来规避

影响：

- 代码行为容易受 import 顺序影响
- 做方法比较时，后端差异可能被忽略

### P2. 研究代码与公共工具层没有分层

现状：

- CLI、预处理、hook、导出、可视化、方法实现混在单文件里
- 部分脚本已超过 1000 行

影响：

- 可维护性差
- 新方法越多，重复代码越多

## 3. 目标状态

理想中的免训练编辑实验框架应满足：

- 同一个 case 可以被多种方法复用
- 同一个 seed、预处理、输入资产可以在不同方法间共享
- 方法实现和实验 I/O 解耦
- 每次运行都自动落盘配置、输入快照、中间结果、最终结果
- 能方便地做单 case 调试和批量 case 对比

## 4. 重构优先级

## 阶段 A：先统一协议，不急着做大重构

### A1. 定义统一 case 格式

新增一个面向编辑实验的标准 case 目录约定，例如：

```text
cases/<case_name>/
  manifest.json
  source/
    voxels.ply
    features.npz
    2d_render.png
  edit/
    2d_edit.png
    2d_mask.png
    mask.glb
```

`manifest.json` 建议至少记录：

- case 名称
- source asset 路径
- edit image 路径
- 2D mask 路径
- 3D mask 路径
- 推荐 seed
- 备注

### A2. 抽离统一输入解析层

目标：

- 不再让每个脚本自己猜 `render_dir` / `source_model` / `image_dir`
- 不再让每个脚本自己写一遍“自动找 `2d_render.png` / `2d_edit.png`”

建议新增模块：

- `editing/io/case_loader.py`
- `editing/io/path_resolver.py`

### A3. 抽离统一预处理层

目标：

- 把 source/edit/mask 对齐逻辑从方法脚本里拆出来
- 把“共享 crop + resize”变成明确、可记录、可复现的策略

建议新增模块：

- `editing/preprocess/image_alignment.py`
- `editing/preprocess/mask_utils.py`

输出应固定保存：

- `input_preprocess.json`
- `source_preprocessed.png`
- `edit_preprocessed.png`
- `mask_preprocessed.png`

## 阶段 B：把方法实现从脚本里拆出来

### B1. 定义统一方法接口

建议设计一个统一抽象，例如：

```python
class EditMethod:
    method_name: str

    def prepare(self, pipeline, case, config): ...
    def run(self, pipeline, case, config): ...
    def save_artifacts(self, outputs, out_dir): ...
```

先收拢三类方法：

- `image_prompt_to_prompt`
- `image_prompt_to_prompt_rf_inversion`
- `image_uniedit_rf_inversion`

### B2. 抽离公共 hook / patch 工具

建议新增模块：

- `editing/hooks/attention_patch.py`
- `editing/hooks/cross_attention_trace.py`
- `editing/hooks/runtime_patch_utils.py`

目标：

- 把 monkey patch 细节从业务脚本拿走
- 让方法只描述“何时注入、注入什么”

### B3. 抽离 inversion 公共层

建议新增模块：

- `editing/inversion/rf_inversion.py`
- `editing/inversion/noise_projection.py`
- `editing/inversion/asset_features.py`

目标：

- RF inversion 相关逻辑不要继续散落在单一脚本中
- 便于后续增加新的 inversion baseline

## 阶段 C：建立可比较的实验 runner

### C1. 提供统一 CLI

建议新增：

- `run_edit_experiment.py`

支持类似命令：

```bash
conda activate hammer
python run_edit_experiment.py \
  --method image_prompt_to_prompt_rf_inversion \
  --case cases/cat_to_tiger \
  --seed 1
```

### C2. 提供批量对比 runner

建议新增：

- `run_edit_benchmark.py`

能力至少包括：

- 一组 case 跑多种方法
- 自动保存失败日志
- 自动汇总输出目录
- 生成一个总表，例如 `summary.jsonl` 或 `summary.csv`

### C3. 固定结果目录结构

在已有 `output_layout.py` 基础上继续收敛为：

```text
outputs/<method>/<case>/
  config.json
  input_preprocess.json
  artifacts/
  edit/
  source_original/
  metrics.json
  status.json
```

## 5. 建议优先抽离的公共模块

第一批建议抽离：

- `editing/common/output_layout.py`
- `editing/common/save_utils.py`
- `editing/common/seed_utils.py`
- `editing/common/device_utils.py`
- `editing/common/backend_config.py`
- `editing/preprocess/image_alignment.py`
- `editing/preprocess/mask_utils.py`
- `editing/io/case_loader.py`

第二批建议抽离：

- `editing/methods/image_prompt_to_prompt.py`
- `editing/methods/image_prompt_to_prompt_rf_inversion.py`
- `editing/methods/image_uniedit_rf_inversion.py`
- `editing/inversion/rf_inversion.py`
- `editing/hooks/attention_patch.py`

## 6. 最小可落地 TODO

如果只做最少改造，建议按这个顺序：

1. 新增 case manifest 规范，并准备 2 到 3 个标准测试 case。
2. 抽离统一的输入路径解析函数。
3. 抽离统一的 source/edit/mask 预处理函数。
4. 抽离统一的输出保存函数和结果目录约定。
5. 把 `image_prompt_to_prompt_rf_inversion` 和 `image_uniedit_rf_inversion` 先接到同一个 runner 上。
6. 再考虑把 cross-attention trace / 可视化接入同一框架。

## 7. 非目标

当前阶段不优先做：

- 重写 TRELLIS 核心模型
- 改训练框架
- 做端到端 GUI 平台
- 追求一次性统一 text 和 image 全部编辑路径

先把 image-conditioned 的免训练编辑实验框架收稳，比追求“大而全”更重要。

## 8. 建议的近期验收标准

当下面几件事都成立时，可以认为第一阶段整理完成：

- 同一个 case 可以无改动切换至少 2 种编辑方法
- 预处理输出在不同方法间一致且可落盘检查
- 每次运行都会自动保存完整配置和输入快照
- 可以用一条命令在 `hammer` 环境下复现实验
- 新增一个方法时，不需要复制上千行旧脚本

