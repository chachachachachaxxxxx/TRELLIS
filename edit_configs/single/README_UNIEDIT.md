# UniEdit Composable 配置说明

当前仓库里，UniEdit 相关配置已经统一到 composable 入口，不再使用 `method_args`。

可选 entrypoint：
- `uniedit_full`：完整流程，先跑 SS，再跑 SLAT
- `uniedit_ss`：只跑 UniEdit SS，输出在 `edit/ss/`
- `uniedit_slat`：只跑 UniEdit SLAT，要求显式提供 `source_features` 和 `edited_coords`
- `ss_p2p_slat_uniedit`：SS 用 P2P，SLAT 用 UniEdit

最小运行命令：
```bash
python run_edit_experiment.py --config edit_configs/template.config --dry-run
python run_edit_experiment.py --config edit_configs/template.config
```

UniEdit 最关键的参数：
- `ss.sampler.steps`：SS 采样步数
- `ss.omega`：SS 编辑强度
- `ss.cfg_interval`：SS 中 CFG 生效区间
- `slat.sampler.steps`：SLAT 采样步数
- `slat.omega`：SLAT 编辑强度
- `slat.cfg_interval`：SLAT 中 CFG 生效区间
- `slat.stage2_variant`：`preserve_uniedit` / `free_target` / `latent_replace_union`
- `slat.decode_modes`：导出 `gaussian`、`mesh`，或二者都要

输入要求：
- `uniedit_full`：`source_image`、`edit_image`、`mask_image`、`source_voxels`、`source_features`
- `uniedit_ss`：不需要 `source_features`
- `uniedit_slat`：必须额外提供 `edited_coords`
- `slat.stage2_variant=latent_replace_union` 时，还要求 `source_voxels` 和 `mask_glb`

推荐做法：
- 从 `edit_configs/template.config` 复制一份新文件
- 把 `entrypoint` 改成 `uniedit_full` / `uniedit_ss` / `uniedit_slat`
- 只保留 UniEdit 对应的 `ss` 和 `slat` 字段，不要混入 P2P 专属字段

更完整的字段解释见：
- `edit_configs/template.config`
- `wiki/apis/API - composable-config.md`
