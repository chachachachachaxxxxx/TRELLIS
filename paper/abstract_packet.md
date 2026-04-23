# SIGGRAPH Asia 2026 摘要包模板

这份文件只服务 `Stage 1 / Submission Form`。目标是在 `2026-05-05 12:00 CST` 前把所有字段冻结到单一版本。

## 提交信息
- 暂定标题：
- 备用标题 1：
- 备用标题 2：
- 一句话 pitch：
- 投递 track：
- topic areas：
- 当前主方法名：
- 当前 strongest baseline：

## 3 条核心 claim
- Claim 1：
- Claim 2：
- Claim 3：

要求：
- 每条 claim 都必须能回指到至少一组现成实验或配置。
- 摘要只写当前已经能支撑的 claim，不为未来可能跑出来的结果预支信用。

## 摘要骨架

### Problem
- 当前 local 3D editing 的核心难点是什么：
- 为什么普通 case 不足以说明问题：

### Method
- 你的方法一句话定义：
- 为什么要显式拆成 `SS -> SLAT` 两阶段：

### Evidence
- 主要 hardest-case 集合：
- 主要定量证据：
- 主要定性证据：

### Result
- 当前最稳的一句结果描述：
- 当前最稳的一句分析描述：

## 作者与管理信息
- 通讯作者：
- 作者顺序：
- 单位列表：
- 邮箱核对：
- OpenReview / PCS / 官方系统账号状态：

## Stage 1 checklist
- [ ] 标题冻结到单一版本
- [ ] 摘要冻结到单一版本
- [ ] topic areas 已确定
- [ ] 完整作者列表已确认
- [ ] 作者顺序已确认
- [ ] 单位与邮箱已核对
- [ ] submission form 已创建
- [ ] 系统里能看到正确作者列表
- [ ] 不再新增或替换作者

## 当前默认引用材料
- hardest-case case set：[`edit_configs/case_sets/boundary_contact_full300_worst_union.yaml`](../edit_configs/case_sets/boundary_contact_full300_worst_union.yaml)
- 当前主候选配置：[`edit_configs/batch/anchorflowpp_boundary_contact_full300_worst_union.yaml`](../edit_configs/batch/anchorflowpp_boundary_contact_full300_worst_union.yaml)
- benchmark 回顾：[`temp/benchmark_report_2026-04-15.md`](../temp/benchmark_report_2026-04-15.md)
