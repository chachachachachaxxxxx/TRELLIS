# SIGGRAPH Asia 2026 提交检查清单

## 时间换算
- `Abstract Deadline`: `2026-05-05 23:59 AoE` = `2026-05-06 19:59 CST`
- `Paper Deadline`: `2026-05-12 23:59 AoE` = `2026-05-13 19:59 CST`
- `Upload Deadline`: `2026-05-13 23:59 AoE` = `2026-05-14 19:59 CST`

内部规则：
- 所有正式提交都至少提前 `24` 小时完成。
- 真正的内部截止分别按 `2026-05-05 12:00 CST`、`2026-05-12 12:00 CST`、`2026-05-14 12:00 CST` 执行。

## Stage 1：Abstract / Submission Form
- [ ] 已创建 submission form
- [ ] 标题已冻结
- [ ] 摘要已冻结
- [ ] topic areas 已填写
- [ ] 完整作者列表已填写
- [ ] 作者顺序已确认
- [ ] 单位与邮箱已核对
- [ ] 系统截图或导出信息已留档
- [ ] 不再计划新增作者

## Stage 2：Paper / Full Submission
- [ ] 论文 PDF 已冻结
- [ ] 主结果表数字全部与最终 run 对齐
- [ ] 所有图 caption、表 caption 已核对
- [ ] 方法名、baseline 名、ablation 名全篇一致
- [ ] 匿名性已检查
- [ ] 引用、参考文献、附录交叉引用已检查
- [ ] supplemental 内容已冻结
- [ ] 如果系统要求 checksum，已生成并保存所有最终文件的 `MD5`
- [ ] 所有要上传的文件名已最终确认

## Upload Deadline
- [ ] 如果前一天提交的是 MD5，占位文件与最终上传文件完全一致
- [ ] 最终上传已完成
- [ ] 重新下载或预览平台上的最终文件，确认未损坏
- [ ] 记录最终提交时间与版本号

## Author-side hard requirements
- [ ] 所有作者已填写 `conflict of interest`
- [ ] 所有作者已填写 `expertise keywords`
- [ ] 所有作者账号都能在系统里正常显示

## Paper package 最终核对
- [ ] 标题与 submission form 一致
- [ ] 摘要与 submission form 一致
- [ ] PDF 与 supplemental 的方法名一致
- [ ] 图表中的 case 名与实际输出 run 一致
- [ ] hero cases 都能回溯到具体配置文件和输出目录
- [ ] 论文内所有数字都能回溯到具体评测结果

## 当前 repo 对应材料
- 主作战文档：[`paper/README.md`](./README.md)
- 摘要包模板：[`paper/abstract_packet.md`](./abstract_packet.md)
- hardest-case 配置：[`edit_configs/batch/anchorflowpp_boundary_contact_full300_worst_union.yaml`](../edit_configs/batch/anchorflowpp_boundary_contact_full300_worst_union.yaml)
- hardest-case case set：[`edit_configs/case_sets/boundary_contact_full300_worst_union.yaml`](../edit_configs/case_sets/boundary_contact_full300_worst_union.yaml)
