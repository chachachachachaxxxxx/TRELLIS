# 配置文件组织说明

## 目录结构

```
edit_configs/
├── single/              # 单个样例配置
│   ├── image_p2p_example.yaml
│   └── legacy_example.yaml
├── batch/               # 批处理评测配置
│   ├── gso_no_p2p.yaml
│   ├── legacy_gso_no_p2p.yaml
│   └── legacy_dollhouse_no_p2p.yaml
└── methods/             # 方法默认参数
    └── image_p2p_latent_blend.yaml
```

## 配置类型说明

### 1. single/ - 单个样例配置

用于运行单个编辑实验，适合快速测试和调试。

**使用方式**:
```bash
python run_edit_experiment.py --config edit_configs/single/image_p2p_example.yaml
```

**必需字段**:
- `method`: 方法名称
- `case_name`: 案例名称
- `source_image`: 源图像路径
- `edit_image`: 编辑后图像路径
- `mask_image`: 2D mask 路径

**可选字段**:
- `preprocess`: 是否预处理（默认 true）
- `asset_dir`: 预处理的 assets 目录（跳过预处理）
- `seed`: 随机种子（默认 1）
- `device`: 计算设备（默认 cuda:0）
- `ss_steps`: SS 采样步数（默认 25）
- `slat_steps`: SLAT 采样步数（默认 25）
- `method_args`: 方法特定参数

### 2. batch/ - 批处理评测配置

用于批量运行 Edit3D-Bench 评测。

**使用方式**:
```bash
python run_batch_edit_and_eval.py --config edit_configs/batch/gso_no_p2p.yaml
```

**必需字段**:
- `gt_root`: Edit3D-Bench GT 数据根目录
- `method_name`: 方法名称
- `config_name`: 配置标识符（用于输出目录命名）

**数据过滤字段**:
- `dataset`: 数据集过滤（GSO, PartObjaverse-Tiny）
- `object`: 物体名称过滤
- `prompt_id`: 提示 ID 过滤（1, 2, 3）
- `max_cases`: 限制案例数量

**评测配置**:
- `metrics`: 评测指标列表
- `skip_render`: 是否跳过渲染
- `assets_root`: 预处理的 assets 根目录（可选）

**方法参数**:
- `method_args`: 传递给每个单独编辑实验的参数

### 3. methods/ - 方法默认参数

定义方法的默认参数和预设配置，供参考和复用。

**内容**:
- `default_sampling`: 默认采样参数
- `default_method_args`: 默认方法参数
- `presets`: 预设配置（如 no_p2p, p2p_only, full, fast）

## 参数优先级

命令行参数 > 配置文件参数 > 方法默认参数

## 示例

### 单个样例测试
```bash
# 使用配置文件
python run_edit_experiment.py --config edit_configs/single/image_p2p_example.yaml

# 覆盖部分参数
python run_edit_experiment.py \
  --config edit_configs/single/image_p2p_example.yaml \
  --seed 42 \
  --device cuda:1
```

### 批处理评测
```bash
# 完整评测
python run_batch_edit_and_eval.py --config edit_configs/batch/gso_no_p2p.yaml

# 限制案例数量
python run_batch_edit_and_eval.py \
  --config edit_configs/batch/gso_no_p2p.yaml \
  --max-cases 5

# 指定物体
python run_batch_edit_and_eval.py \
  --config edit_configs/batch/gso_no_p2p.yaml \
  --object 3D_Dollhouse_Happy_Brother
```

## 配置文件最佳实践

1. **分类清晰**: single/ 用于单个样例，batch/ 用于批处理
2. **注释完整**: 每个参数都有注释说明
3. **分段组织**: 使用注释分隔不同功能区域
4. **命名规范**: 
   - single/: `{method}_{variant}.yaml`
   - batch/: `{dataset}_{variant}.yaml`
5. **参数复用**: 常用配置放在 methods/ 供参考

## 输出目录

### 单个样例
```
outputs/{method_name}/{case_name}/
├── config.json          # 运行配置
├── status.json          # 运行状态
├── edit/                # 编辑结果
│   └── sample_00.glb
├── artifacts/           # 中间产物
└── logs/                # 日志
```

### 批处理
```
/cache/wangxinxing/data/temp/{method_name}_{config_name}/
├── {dataset}/
│   └── {object}/
│       └── prompt_{id}/
│           ├── edit.glb
│           └── renders/
└── evaluation_output/   # 评测结果

outputs/results/{method_name}_{config_name}_{timestamp}/
├── evaluation_results.json
└── report.txt
```
