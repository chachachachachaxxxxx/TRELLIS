# 方法实现拆离指南

## 已完成

### 基础框架

- ✅ `editing/methods/base.py` - 统一方法基类和接口
- ✅ `editing/methods/runner.py` - 方法运行器
- ✅ `editing/methods/__init__.py` - 模块导出

### 核心抽象

**EditMethod** - 方法基类

```python
class EditMethod(ABC):
    def prepare(pipeline, inputs, config) -> prepared_state
    def run(pipeline, prepared_state, config) -> outputs
    def save_artifacts(outputs, out_dir, config) -> artifact_paths
    def cleanup()
```

**EditMethodRunner** - 运行器

- 协调完整流程：预处理 → 准备 → 运行 → 保存 → 清理
- 统一输出目录管理
- 统一配置保存

**数据类**

- `EditMethodConfig` - 方法配置
- `EditMethodInputs` - 方法输入
- `EditMethodOutputs` - 方法输出

## 迁移策略

### 阶段 1：保持现有脚本可用

当前策略：

1. 新框架与现有脚本并存
2. 现有脚本继续通过 `run_edit_experiment.py` 调用
3. 逐步将方法实现迁移到 `editing/methods/` 下

### 阶段 2：创建方法实现类

对于每个方法（如 `image_prompt_to_prompt`）：

1. **识别核心组件**
   - Hook/Patch 类（如 `ImagePromptToPromptEditor`）
   - 辅助函数（如 `mask_to_patch_selection`）
   - 配置解析（如 `parse_stage_list`）

2. **创建方法类**

   ```python
   # editing/methods/image_prompt_to_prompt.py
   class ImagePromptToPromptMethod(EditMethod):
       def prepare(self, pipeline, inputs, config):
           # 构建 editor, token_meta, conditions
           ...
       
       def run(self, pipeline, prepared_state, config):
           # 执行 pipeline with hooks
           ...
       
       def save_artifacts(self, outputs, out_dir, config):
           # 保存 GLB, PLY, renders
           ...
   ```

3. **抽离公共工具**
   - `editing/hooks/attention_patch.py` - Attention hook 基类
   - `editing/utils/token_utils.py` - Token 处理工具
   - `editing/utils/visualization.py` - 可视化工具

### 阶段 3：更新注册表

```python
# editing/methods/registry.py
METHODS = {
    "image_prompt_to_prompt": MethodSpec(
        name="image_prompt_to_prompt",
        script_path="example_image_prompt_to_prompt.py",  # 旧方式
        method_class=ImagePromptToPromptMethod,  # 新方式
        description="...",
    ),
}
```

支持两种调用方式：

- 旧：通过 `script_path` 调用脚本
- 新：通过 `method_class` 实例化方法类

### 阶段 4：逐步废弃脚本

当方法类稳定后：

1. 移除 `script_path`
2. 将脚本移到 `legacy/` 或删除
3. 更新文档

## 优先级

### P0 - 立即完成

- ✅ 创建基础框架（base, runner）
- ⏳ 创建公共工具模块
  - `editing/hooks/` - Hook/Patch 工具
  - `editing/utils/` - 通用工具函数

### P1 - 近期完成

- ⏳ 迁移第一个方法（建议：`image_prompt_to_prompt`）
  - 最简单，没有 RF inversion 复杂度
  - 验证框架设计是否合理
- ⏳ 迁移第二个方法（建议：`image_prompt_to_prompt_rf_inversion`）
  - 验证 inversion 相关抽象

### P2 - 后续完成

- ⏳ 迁移其他方法
- ⏳ 废弃旧脚本
- ⏳ 完善文档和示例

## 公共工具模块设计

### editing/hooks/

```
editing/hooks/
├── __init__.py
├── base.py                    # Hook 基类
├── attention_patch.py         # Attention patching
└── cross_attention_trace.py   # Cross-attention tracing
```

### editing/utils/

```
editing/utils/
├── __init__.py
├── token_utils.py      # Token 处理（mask_to_patch_selection 等）
├── visualization.py    # 可视化工具（save_patch_grid_preview 等）
└── tensor_utils.py     # Tensor 工具（tensor_signature 等）
```

### editing/inversion/

```
editing/inversion/
├── __init__.py
├── rf_inversion.py     # RF inversion 实现
└── noise_projection.py # Noise projection 工具
```

## 验证标准

方法迁移完成的标准：

1. ✅ 方法类实现了 `EditMethod` 接口
2. ✅ 可以通过 `EditMethodRunner` 运行
3. ✅ 输出与原脚本一致（相同 seed 下）
4. ✅ 配置和中间结果正确保存
5. ✅ 单元测试通过

## 下一步

1. 创建公共工具模块（hooks, utils, inversion）
2. 迁移 `image_prompt_to_prompt` 方法
3. 验证框架设计
4. 继续迁移其他方法
