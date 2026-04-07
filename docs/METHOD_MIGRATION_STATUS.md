# 方法迁移状态

## 已迁移 ✅

### image_prompt_to_prompt
- **方法类:** `editing/methods/image_prompt_to_prompt.py`
- **Hook:** `editing/hooks/prompt_to_prompt.py`
- **状态:** 完全迁移，旧脚本已删除
- **Commit:** `1c54e51`

## 待迁移 ⏳

### image_prompt_to_prompt_rf_inversion
- **脚本:** `example_image_prompt_to_prompt_rf_inversion.py` (983 行)
- **依赖:** RF inversion 工具
- **优先级:** P1 - 需要先抽离 RF inversion 公共层

### image_uniedit_rf_inversion
- **脚本:** `example_image_uniedit_rf_inversion.py` (1489 行)
- **依赖:** RF inversion + UniEdit 逻辑
- **优先级:** P1 - 需要先抽离 RF inversion 公共层

### image_slat_xor_fusion
- **脚本:** `example_image_slat_xor_fusion.py` (387 行)
- **依赖:** RF inversion 资产
- **优先级:** P2 - 相对简单，但依赖 RF inversion

### image_cross_attention
- **脚本:** `example_image_cross_attention.py` (2096 行)
- **依赖:** Cross-attention tracing 工具
- **优先级:** P2 - 主要用于可视化和调试

### text_prompt_to_prompt
- **脚本:** `example_text_prompt_to_prompt.py` (893 行)
- **依赖:** Text pipeline + Prompt-to-Prompt hook
- **优先级:** P1 - 结构类似 image 版本，可复用 hook

### text_cross_attention
- **脚本:** `example_text_cross_attention.py` (1866 行)
- **依赖:** Text pipeline + Cross-attention tracing
- **优先级:** P2 - 主要用于可视化和调试

## 迁移策略

### 短期（本次会话）
1. ✅ 实现基础框架
2. ✅ 迁移 image_prompt_to_prompt
3. ✅ 更新 run_edit_experiment.py 支持方法类
4. ✅ 删除已迁移的脚本

### 中期（下次会话）
1. 抽离 RF inversion 公共工具
   - `editing/inversion/rf_inversion.py`
   - `editing/inversion/noise_projection.py`
2. 迁移 image_prompt_to_prompt_rf_inversion
3. 迁移 text_prompt_to_prompt

### 长期
1. 迁移 image_uniedit_rf_inversion
2. 迁移其他方法
3. 完善测试和文档

## 使用方式

### 已迁移方法（自动使用方法类）
```bash
python run_edit_experiment.py \
  --method image_prompt_to_prompt \
  --case cases/cat_to_tiger \
  --seed 1
```
→ 自动使用 `ImagePromptToPromptMethod`

### 未迁移方法（使用脚本）
```bash
python run_edit_experiment.py \
  --method image_prompt_to_prompt_rf_inversion \
  --case cases/cat_to_tiger \
  --seed 1
```
→ 调用 `example_image_prompt_to_prompt_rf_inversion.py`

## 验证

### 输出一致性
- [ ] 验证新方法类与旧脚本输出一致（相同 seed）
- [ ] 验证配置保存格式
- [ ] 验证中间结果保存

### 功能完整性
- [x] 预处理正确
- [x] Hook 注入正确
- [x] 输出保存正确
- [x] Cleanup 正确

## 统计

- **总方法数:** 7
- **已迁移:** 1 (14%)
- **待迁移:** 6 (86%)
- **删除代码:** 1112 行
- **新增代码:** ~2500 行（框架 + 工具 + 方法类）

## 下一步

1. 抽离 RF inversion 工具到 `editing/inversion/`
2. 迁移 `text_prompt_to_prompt`（结构简单，可复用 hook）
3. 迁移 `image_prompt_to_prompt_rf_inversion`
4. 继续迁移其他方法
