# SIGGRAPH Asia 2026 投稿作战文档

## 目的
这份文档把当前仓库的本地 3D 编辑主线，收敛成一个可执行的 SIGGRAPH Asia 2026 Technical Papers 投稿节奏。

当前默认前提：
- 当前时间：`2026-04-20`，时区按北京时间 / 上海时间（`UTC+8`）记
- 当前主故事：`hard cases`，尤其是 boundary contact、support replacement、top-mounted accessory、surface contact
- 当前主要候选方法：`anchorflowpp` 风格的两阶段编辑主线
- 当前协作方式：基本单人推进
- 当前风险判断：方法还未完全冻结，因此 `2026-04-26` 前必须完成方法收敛

## 官方 deadline 与内部冻结线

| 阶段 | 官方时间（AoE） | 北京时间 / 上海时间 | 内部冻结线 |
| --- | --- | --- | --- |
| Abstract Deadline / Submission Form | `2026-05-05 23:59 AoE` | `2026-05-06 19:59 CST` | `2026-05-05 12:00 CST` |
| Paper Deadline / Full Submission | `2026-05-12 23:59 AoE` | `2026-05-13 19:59 CST` | `2026-05-12 12:00 CST` |
| Upload Deadline | `2026-05-13 23:59 AoE` | `2026-05-14 19:59 CST` | `2026-05-14 12:00 CST` |

补充约束：
- `Stage 1` 前必须先建 submission form，并锁定标题、摘要、topic areas、完整作者列表。
- `Stage 2` 前必须交完整论文 PDF，以及 supplemental 文件或对应 `MD5 checksum`。
- `Upload Deadline` 前如果前一天走的是 MD5 占位，只能上传与前一天 MD5 一致的文件。
- 到最终上传截止前，所有作者都必须填写 `conflict of interest` 和 `expertise keywords`，否则有 automatic desk-reject 风险。

## 当前故事与证据结构

### 核心叙事
- 主打“局部 3D 编辑里最难的不是普通风格变化，而是接触关系、支撑关系、壳层附着和局部拓扑重组”。
- 主方法卖点不是单阶段替换，而是显式 `preprocess -> ss -> slat` 两阶段编辑。
- 重点证明两件事：
  - Stage 1 决定结构怎么改，尤其是边界与接触区域怎么变。
  - Stage 2 决定保留与生成如何平衡，避免结果退化成重建漂移或局部塌陷。

### 当前优先证据

| 证据目标 | 当前仓库落点 | 通过标准 |
| --- | --- | --- |
| hardest-case 主故事 | [`edit_configs/case_sets/boundary_contact_full300_worst_union.yaml`](../edit_configs/case_sets/boundary_contact_full300_worst_union.yaml) | 12 个 hardest cases 上形成稳定、可讲清楚的优势 |
| 当前主候选正式跑法 | [`edit_configs/batch/anchorflowpp_boundary_contact_full300_worst_union.yaml`](../edit_configs/batch/anchorflowpp_boundary_contact_full300_worst_union.yaml) | 产出完整 `pred_root`、daily 页面和可挑选 hero cases |
| 两阶段必要性 | [`edit_configs/ablation1/`](../edit_configs/ablation1), [`edit_configs/ablation3/`](../edit_configs/ablation3) | 至少拿出 2 到 3 组关键消融，能支持“SS-only 不够，SLAT 约束必要” |
| 非 hardest-case 支撑 | [`edit_configs/batch/full_benchmark.yaml`](../edit_configs/batch/full_benchmark.yaml), [`edit_configs/batch/full_benchmark_2.yaml`](../edit_configs/batch/full_benchmark_2.yaml) | 至少证明方法不是只在挑选案例上成立 |
| 当前 benchmark 现状回顾 | [`temp/benchmark_report_2026-04-15.md`](../temp/benchmark_report_2026-04-15.md) | 可直接转成 paper 的“动机 + gap + why now”材料 |

## 冻结规则

### 时间冻结
- `2026-04-22` 前：冻结 paper scope，不再扩论文问题定义。
- `2026-04-26` 前：冻结唯一主方法，不再新增方法分支。
- `2026-05-05` 前：冻结标题、摘要、作者信息。
- `2026-05-10` 前：冻结主实验结果、主图和主表。
- `2026-05-12` 前：冻结 PDF 与 supplemental 内容。

### 技术冻结
- 不新增新的编辑框架；只允许沿现有 composable 主线推进。
- 单 case 或 smoke 统一走 [`run_edit_experiment.py`](../run_edit_experiment.py)。
- 批量 benchmark 统一走 `run_batch_edit_and_eval.py`。
- 当前 GPU 运行策略默认沿用 `2026-04-19` 结论：最多同时压满 3 张卡，且暂时不使用 `GPU1`。
- `2026-04-26` 后禁止“为了翻盘”再开新方法，只允许补稳定性、补实验、补图表。

## 关键路径

### Phase 0：今天到 `2026-04-22`
目标：把 scope cut 做完。

必须完成：
- 明确一句话题眼，写进 [`abstract_packet.md`](./abstract_packet.md)。
- 确认当前主候选是否就是 `anchorflowpp`；如果不是，要把替代配置写进本文档。
- 列出 paper 只保留的 3 条 claim，以及每条 claim 的证据来源。
- 把“明确不做的内容”写下来：
  - 不以多视图编辑为主故事
  - 不以全面 SOTA benchmark 为主故事
  - 不在这轮投稿里继续扩方法空间

建议命令：

```bash
conda activate hammer
python run_edit_experiment.py --list-entrypoints
python run_edit_experiment.py --config edit_configs/template.config --dry-run
python -m compileall trellis trellis_edit run_edit_experiment.py run_batch_edit_and_eval.py
```

### Phase 1：`2026-04-23` 到 `2026-04-26`
目标：方法冻结。

必须完成：
- 对当前主候选跑 hardest-case 正式结果。
- 用最强的 2 到 3 个 baseline 做同口径对照。
- 做一次失败模式复盘，决定论文主图和 hero cases。
- 如果主候选仍不稳定，降级为“当前最稳版本 + hard-case failure analysis”，不要继续追新想法。

主命令：

```bash
conda activate hammer
python run_batch_edit_and_eval.py --config edit_configs/batch/anchorflowpp_boundary_contact_full300_worst_union.yaml
```

当前该命令对应：
- 主配置：[`edit_configs/batch/anchorflowpp_boundary_contact_full300_worst_union.yaml`](../edit_configs/batch/anchorflowpp_boundary_contact_full300_worst_union.yaml)
- case set：[`edit_configs/case_sets/boundary_contact_full300_worst_union.yaml`](../edit_configs/case_sets/boundary_contact_full300_worst_union.yaml)
- 输出目录：`/cache/wangxinxing/data/trellis_edit_benchmark/pred/anchorflowpp_boundary_contact_full300_worst_union`

### Phase 2：`2026-04-27` 到 `2026-04-30`
目标：摘要包与长跑实验排程。

必须完成：
- 填完 [`abstract_packet.md`](./abstract_packet.md)。
- 生成标题 `v1`、摘要 `v1`、topic areas、作者确认表。
- 启动主 benchmark、ablation、hero case 补跑。
- 固定图表清单：
  - 1 张 teaser
  - 1 张方法总图
  - 1 张 hardest-case 主定性图
  - 1 张失败模式图
  - 1 张主结果表
  - 1 张关键 ablation 表

### Phase 3：`2026-05-01` 到 `2026-05-05`
目标：Stage 1 提交。

必须完成：
- 在 `2026-05-05 12:00 CST` 前冻结 submission form 需要的全部信息。
- 实际在系统中创建 submission form，不拖到官方最后几小时。
- 确认作者顺序、单位、通讯信息、topic areas。

执行标准：
- 标题、摘要、topic areas、作者列表全部有单一版本。
- 仓库里至少已经有主方法 + strongest baseline 的 hardest-case 结果可支撑摘要文字。

### Phase 4：`2026-05-06` 到 `2026-05-10`
目标：正文与主结果收口。

必须完成：
- 写完 Introduction、Related Work、Method、Experiment setup 初稿。
- hardest-case 主图和主表定稿。
- 关键 ablation 定稿。
- 失败案例图定稿。
- 如果 full benchmark 已有能用结果，只做收口与挑图，不再无休止重跑。

建议保留的正文结构：
- 问题定义：局部 3D 编辑在 hard structural contact 下为什么难
- 方法：为什么必须显式拆成 SS / SLAT 两阶段
- 主结果：hardest cases 为主，general benchmark 为辅
- 分析：边界、接触、保留/生成平衡、失败模式

### Phase 5：`2026-05-11` 到 `2026-05-12`
目标：Full Submission 内部冻结。

必须完成：
- PDF 全文通读至少 1 轮。
- 所有图表、表格、caption、数字、引用和匿名性统一核对。
- supplemental 至少包含：
  - 更多 hardest cases
  - 更多 failure cases
  - 额外可视化 / daily 页面截图
  - 复现所需运行细节

执行标准：
- `2026-05-12 12:00 CST` 前内部冻结 PDF 和 supplemental。
- 如果系统切到 checksum 模式，同步生成文件与 `MD5`。

### Phase 6：`2026-05-13` 到 `2026-05-14`
目标：正式提交与最终上传。

必须完成：
- `2026-05-13 12:00 CST` 前完成 Stage 2 提交。
- 如果前一天提交的是 `MD5 checksum`，则 `2026-05-14 12:00 CST` 前上传与 MD5 完全一致的文件。
- 在 `2026-05-14 12:00 CST` 前逐个确认所有作者已经填完 `COI` 与 `expertise keywords`。

## 论文主结果最小包

如果只能完成最小可投稿版本，最低保留以下内容：
- 1 条最终方法线
- 1 个问题导向 hardest-case case set
- 2 到 3 个强基线
- 2 组关键 ablation
- 3 个 hero cases
- 1 组 failure analysis
- 1 份 supplemental

如果到 `2026-04-26` 仍没有稳定超越，则默认降级策略：
- 不再扩方法
- 选择最稳版本
- 把 hardest-case taxonomy、失败模式和两阶段必要性讲透

## 运行纪律
- 所有正式结果都记录：命令、配置、seed、输出路径、GPU。
- 所有要进 paper 的结果都必须能回指到具体配置文件。
- 任何“手工挑图”都要保留来源 run id，避免后期图表失配。
- 任何 nightly 或长跑前先做 `--dry-run` 或最小 smoke。

## 相关文档
- 摘要包模板：[`abstract_packet.md`](./abstract_packet.md)
- 提交检查清单：[`submission_checklist.md`](./submission_checklist.md)
- 当前 benchmark 诊断：[`temp/benchmark_report_2026-04-15.md`](../temp/benchmark_report_2026-04-15.md)
- 当前 GPU 风险记录：[`wiki/changes/Change - 2026-04-19-gpu1-pcie-instability.md`](../wiki/changes/Change%20-%202026-04-19-gpu1-pcie-instability.md)
