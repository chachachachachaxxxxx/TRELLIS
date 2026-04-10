# Edit3D-Bench 评测快速参考

## 一键式评测

```bash
python run_full_evaluation.py \
  --gt-root /home/wangxinxing/code/Edit3Dpp/data \
  --method-name image_prompt_to_prompt
```

## 输出位置

### edit.glb 文件
```
/cache/wangxinxing/data/temp/{method_name}_{timestamp}/{method_name}/{dataset}/{object_name}/prompt_{1,2,3}/edit.glb
```

### 评测结果
```
outputs/results/{method_name}_{timestamp}/
├── evaluation_results.json
├── report.txt
├── detailed_results.json
└── summary.json
```

## 常用命令

```bash
# 快速测试（1个案例）
python run_full_evaluation.py \
  --gt-root /home/wangxinxing/code/Edit3Dpp/data \
  --method-name image_prompt_to_prompt \
  --max-cases 1

# 仅评测 GSO 数据集
python run_full_evaluation.py \
  --gt-root /home/wangxinxing/code/Edit3Dpp/data \
  --method-name image_prompt_to_prompt \
  --dataset GSO

# 查看结果
cat outputs/results/image_prompt_to_prompt_*/report.txt

# 查看 edit.glb 位置
ls /cache/wangxinxing/data/temp/image_prompt_to_prompt_*/image_prompt_to_prompt/

# 清理旧数据
rm -rf /cache/wangxinxing/data/temp/image_prompt_to_prompt_*
```

## 命名规范

案例目录需包含物体名称：
```
outputs/image_prompt_to_prompt/3D_Dollhouse_Happy_Brother_prompt_1/edit/sample_00.glb
outputs/image_prompt_to_prompt/GSO_3D_Dollhouse_Happy_Brother_prompt_1/edit/sample_00.glb
outputs/image_prompt_to_prompt/3D_Dollhouse_Happy_Brother_p1/edit/sample_00.glb
```

## 详细文档

- `docs/ONE_CLICK_EVALUATION.md` - 完整使用指南
- `docs/EVAL_OUTPUT_FORMAT.md` - 输出格式说明
