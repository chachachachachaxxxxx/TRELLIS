```bash
python run_edit_experiment.py \
  --method image_uniedit_rf_inversion \
  --render-dir outputs/rf_p2p/render \
  --source-image assets/edit_example/images/2d_render.png \
  --edit-image assets/edit_example/images/2d_edit.png \
  --mask-glb assets/edit_example/mask.glb \
  --input-model assets/edit_example/model.glb \
  --seed 1

conda activate hammer
python run_edit_experiment.py \
  --method image_slat_xor_fusion \
  --render-dir outputs/rf_p2p/render \
  --edit-image assets/edit_example/images/2d_edit.png \
  --seed 1

```

这条 `uniedit` 现在默认只会先输出 3D mask 预览并停止；确认没问题后，再补一个继续参数：

```bash
python run_edit_experiment.py \
  --method image_uniedit_rf_inversion \
  --render-dir outputs/image_uniedit_rf_inversion/render \
  --source-image assets/edit_example/images/2d_render.png \
  --edit-image assets/edit_example/images/2d_edit.png \
  --mask-glb assets/edit_example/mask.glb \
  --input-model assets/edit_example/model.glb \
  --seed 1 \
  -- --continue-after-mask-preview


```

也可以先整理成标准 case 目录，再直接用统一入口：

```bash
python run_edit_experiment.py --init-case cases/cat_to_tiger
```

把资产放入 `source/` 和 `edit/` 后运行：

```bash
python run_edit_experiment.py \
  --method image_prompt_to_prompt_rf_inversion \
  --case cases/cat_to_tiger \
  -- --ss-steps 12 --slat-steps 12
```

建议的 case 结构：

```text
cases/<case_name>/
  manifest.json
  source/
    voxels.ply
    features.npz
    2d_render.png
    model.glb
  edit/
    2d_edit.png
    2d_mask.png
    mask.glb
```

`manifest.json` 可以只写需要覆盖默认推断的字段；如果目录遵循上面的约定，`run_edit_experiment.py` 会自动补齐常见路径。`methods.<method_name>.args` 可以填写某个方法专属的额外参数，例如 `["--continue-after-mask-preview"]`。
