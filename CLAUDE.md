# TRELLIS 项目说明

本仓库当前以 `wiki/` 作为 editing 重构和配置说明的持久化入口；`docs/` 已不再作为事实来源。

## 先看哪里

- `wiki/home.md`
- `wiki/index.md`
- `wiki/apis/API - editing-cli.md`
- `wiki/apis/API - composable-config.md`
- `wiki/modules/Module - editing-runner.md`
- `wiki/modules/Module - batch-edit-and-eval.md`
- `wiki/architecture/editing-refactor-plan.md`

## 环境

- 默认使用 `hammer` conda 环境
- 优先使用 `microsoft/TRELLIS-image-large`
- 在 import `trellis` 前设置：
  - `ATTN_BACKEND`
  - `SPARSE_ATTN_BACKEND`
  - `SPCONV_ALGO`
- 一次性示例默认用 `SPCONV_ALGO=native`

## 当前 trellis_edit 结构

编辑框架现在以 composable 为主线：

- `trellis_edit/composable/`
  - composable 原生执行路径
  - `preprocess -> ss -> slat` 的显式 stage 编排
  - `entrypoint + runtime/inputs/preprocess/ss/slat` 结构化配置
- `trellis_edit/preprocess/`
  - 输入对齐、mask、source asset 生成
- `trellis_edit/hooks/` / `trellis_edit/inversion/` / `trellis_edit/samplers/`
  - stage 算法组件
- `trellis_edit/common/` / `trellis_edit/utils/`
  - 通用输出布局与工具函数

统一入口：

- `run_edit_experiment.py`
  - 单次 composable 实验 CLI
- `run_batch_edit_and_eval.py`
  - 批量评测只接受 composable 配置

## 当前推荐用法

单次实验：

```bash
python run_edit_experiment.py --config edit_configs/template.config --dry-run
python run_edit_experiment.py --config path/to/your_case.config
```

批量评测：

```bash
python run_batch_edit_and_eval.py --config edit_configs/quick_test_batch.yaml --dry-run
python run_batch_edit_and_eval.py --config edit_configs/batch/full_benchmark.yaml
```

列出可运行入口：

```bash
python run_edit_experiment.py --list-entrypoints
```

## 配置约定

Composable YAML 使用这些顶层键：

- `entrypoint`
- `runtime`
- `inputs`
- `preprocess`
- `ss`
- `slat`
- `batch`

说明：

- `entrypoint` 决定 stage 组合，例如 `uniedit_full`、`p2p_latent_blend_full`
- `runtime` 放模型、设备、seed、输出控制
- `inputs` 放显式图像和 3D 路径
- `preprocess`、`ss`、`slat` 只接受结构化字段，不接受 `method_args`
- `batch` 只给 `run_batch_edit_and_eval.py` 用，`run_edit_experiment.py` 会忽略它

## 输出目录

统一输出布局：

```text
outputs/<entrypoint>/<case_name>/
├── edit/
│   ├── preprocess/
│   ├── ss/
│   └── slat/
├── source_original/
└── artifacts/
```

批量评测预测树默认输出到：

```text
/cache/wangxinxing/data/temp/<entrypoint>_<config_name>/
```

## 代码修改规则

- 不要修改 `trellis/representations/mesh/flexicubes/`
- 不要提交 `outputs/`、模型权重、大型导出文件
- 尽量把 durable 说明写进 `wiki/`，不要重新堆回 `docs/`
- 当前阶段分离和 P2P cleanup 不做旧命名兼容层

## 验证

最小验证命令：

```bash
python -m compileall trellis trellis_edit run_edit_experiment.py run_batch_edit_and_eval.py
python run_edit_experiment.py --config edit_configs/template.config --dry-run
python run_batch_edit_and_eval.py --config edit_configs/quick_test_batch.yaml --dry-run
```

## 常见坑

- composable 路径不接受 `--case`、`asset_dir`、`render_dir`、`method_args`
- stage-only `ss` entrypoint 不会产出最终 `edit.glb`，不能直接拿去做 Edit3D-Bench 评测
- batch runner 依赖显式 3D 输入模板；如果没有 `source_voxels` / `source_features`，full/slat 运行会直接失败
