# 方法迁移状态

## 已迁移 ✅

### image_prompt_to_prompt
- **方法类:** `editing/methods/image_prompt_to_prompt.py`
- **Hook:** `editing/hooks/prompt_to_prompt.py`
- **状态:** 完全迁移并测试通过
- **测试:** GLB/PLY 导出成功
- **Commit:** `1c54e51`, `0cd567f`

### text_prompt_to_prompt
- **方法类:** `editing/methods/text_prompt_to_prompt.py`
- **工具:** `editing/utils/text_token_utils.py`
- **状态:** 核心逻辑完成，测试部分成功（解码 OOM）
- **测试:** 采样成功，需要更大显存完成解码
- **Commit:** `5a3a432`, `4b49e9f`

### image_slat_xor_fusion
- **方法类:** `editing/methods/image_slat_xor_fusion.py`
- **状态:** 完全迁移，待测试
- **依赖:** source SLAT assets (features.npz)
- **Commit:** `662de5a`

### image_prompt_to_prompt_rf_inversion
- **方法类:** `editing/methods/image_prompt_to_prompt_rf_inversion.py`
- **Inversion:** `editing/inversion/rf_inversion.py`
- **状态:** 完全迁移，待测试
- **依赖:** voxels.ply + features.npz
- **Commit:** `0e01605`, `a754fa5`

### image_uniedit_rf_inversion
- **方法类:** `editing/methods/image_uniedit_rf_inversion.py`
- **状态:** 占位符实现（NotImplementedError）
- **说明:** UniEdit 两阶段编辑逻辑复杂，需要完整实现
- **Commit:** `ecfbb62`

## 待迁移 ⏳

### image_cross_attention
- **脚本:** `vis/example_image_cross_attention.py` (2096 行)
- **依赖:** Cross-attention tracing 工具
- **优先级:** P3 - 主要用于可视化和调试，后续单独处理

### text_cross_attention
- **脚本:** `vis/example_text_cross_attention.py` (1866 行)
- **依赖:** Text pipeline + Cross-attention tracing
- **优先级:** P3 - 主要用于可视化和调试，后续单独处理

## 核心基础设施 ✅

### 预处理层
- ✅ `editing/preprocess/image.py` - 图像预处理
- ✅ `editing/preprocess/asset_3d.py` - 3D 资产预处理
- ✅ `editing/io/path_resolver.py` - 路径解析
- ✅ `editing/io/case_loader.py` - Case 加载（支持 image 和 text）

### 方法接口
- ✅ `editing/methods/base.py` - EditMethod 抽象基类
- ✅ `editing/methods/runner.py` - EditMethodRunner 编排器
- ✅ `editing/methods/registry.py` - 方法注册系统

### Hook 系统
- ✅ `editing/hooks/prompt_to_prompt.py` - PromptToPromptHook
- 支持 dense 和 sparse attention 注入
- 支持分阶段控制 (sparse_structure, slat)

### Inversion 层
- ✅ `editing/inversion/rf_inversion.py` - RF inversion 工具
- `invert_sparse_structure` / `denoise_sparse_structure`
- `invert_slat` / `denoise_slat`
- `get_slat_norm_tensors`

### 工具模块
- ✅ `editing/utils/patch_utils.py` - Patch 元数据
- ✅ `editing/utils/text_token_utils.py` - Text token alignment
- ✅ `editing/common/save_utils.py` - 输出保存

## 使用方式

### Image Prompt-to-Prompt
```bash
python run_edit_experiment.py \
  --method image_prompt_to_prompt \
  --source-image assets/edit_example/images/2d_render.png \
  --edit-image assets/edit_example/images/2d_edit.png \
  --mask-image assets/edit_example/images/2d_mask.png \
  --case-name my_edit \
  --seed 1 \
  --preprocess
```

### Text Prompt-to-Prompt
```bash
python run_edit_experiment.py \
  --method text_prompt_to_prompt \
  --source-prompt "a cute cat statue" \
  --edit-prompt "a cute tiger statue" \
  --case-name text_edit \
  --seed 1
```

### SLAT XOR Fusion
```bash
python run_edit_experiment.py \
  --method image_slat_xor_fusion \
  --source-model path/to/source/assets \
  --edit-image path/to/edit.png \
  --case-name fusion_edit \
  --seed 1
```

### RF Inversion P2P
```bash
python run_edit_experiment.py \
  --method image_prompt_to_prompt_rf_inversion \
  --source-model path/to/source/assets \
  --source-image path/to/source.png \
  --edit-image path/to/edit.png \
  --mask-image path/to/mask.png \
  --case-name rf_edit \
  --seed 1
```

## 验证状态

### 测试完成
- ✅ image_prompt_to_prompt - GLB/PLY 导出成功
- ⚠️ text_prompt_to_prompt - 采样成功，解码 OOM

### 待测试
- ⏳ image_slat_xor_fusion - 需要 source assets
- ⏳ image_prompt_to_prompt_rf_inversion - 需要 source assets
- ⏳ image_uniedit_rf_inversion - 需要完整实现

### 功能验证
- ✅ 预处理正确
- ✅ Hook 注入正确
- ✅ 输出保存正确
- ✅ Cleanup 正确
- ✅ 方法自动检测和切换
- ✅ Pipeline 自动选择（image/text）

## 统计

- **总方法数:** 7
- **已迁移:** 5 (71%)
  - 完整实现: 4
  - 占位符: 1
- **待迁移:** 2 (29%) - 可视化方法
- **删除代码:** ~6000 行
- **新增代码:** ~3000 行（框架 + 工具 + 方法类）
- **总提交数:** 45+

## 修复的问题

1. ✅ 梯度追踪问题 - postprocessing_utils.py 添加 .detach()
2. ✅ 模型自动选择 - text 方法使用 TRELLIS-text-large
3. ✅ Tokenizer 访问 - 使用 text_cond_model['tokenizer']
4. ✅ 语法错误 - 修复 registry.py 多余括号
5. ✅ 默认配置 - 方法默认配置正确应用
6. ✅ extra_inputs 支持 - EditMethodInputs 支持额外输入

## 下一步

### 高优先级
1. 完整实现 UniEdit 方法（两阶段编辑逻辑）
2. 测试 image_slat_xor_fusion
3. 测试 image_prompt_to_prompt_rf_inversion

### 中优先级
4. 优化 text 方法显存使用
5. 添加更多测试用例
6. 完善错误处理

### 低优先级
7. 迁移可视化方法（cross_attention）
8. 性能优化
9. 文档完善

## 架构优势

✅ 统一的 EditMethod 接口
✅ 自动方法检测和切换
✅ 模块化设计，易于扩展
✅ 支持 image 和 text pipeline
✅ 完整的配置管理
✅ 优化的用户体验（默认跳过视频渲染）
