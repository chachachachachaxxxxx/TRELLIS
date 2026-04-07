# TRELLIS 编辑实验框架文档

本目录包含 TRELLIS 编辑实验框架重构的完整文档。

## 📚 文档索引

### 规划与设计

#### [EDITING_EXPERIMENT_TODO.md](./EDITING_EXPERIMENT_TODO.md)
重构规划文档，定义了整个重构的目标、问题分析和实施路线图。

**内容：**
- 当前主要问题分析
- 目标状态定义
- 重构优先级（阶段 A/B/C）
- 最小可落地 TODO
- 验收标准

**适合：** 了解重构背景和整体规划

---

#### [METHOD_MIGRATION_GUIDE.md](./METHOD_MIGRATION_GUIDE.md)
方法迁移指南，说明如何将现有脚本迁移到新框架。

**内容：**
- 迁移策略（4 个阶段）
- 公共工具模块设计
- 验证标准
- 优先级规划

**适合：** 准备迁移方法的开发者

---

### 实施总结

#### [REFACTORING_SUMMARY.md](./REFACTORING_SUMMARY.md)
重构总结文档，完整记录了重构过程和最终架构。

**内容：**
- 完成的工作（阶段 A/B/C）
- 最终架构设计
- 核心抽象（EditMethod, Hook, Runner）
- 使用方式
- 下一步计划

**适合：** 了解重构成果和当前架构

---

#### [PREPROCESS_IMPLEMENTATION_SUMMARY.md](./PREPROCESS_IMPLEMENTATION_SUMMARY.md)
预处理层实现总结，详细说明预处理模块的设计和实现。

**内容：**
- 2D 图像预处理
- 3D 资产预处理
- 路径解析增强
- 验证测试结果

**适合：** 了解预处理层实现细节

---

### 进度跟踪

#### [METHOD_MIGRATION_STATUS.md](./METHOD_MIGRATION_STATUS.md)
方法迁移状态，跟踪各个方法的迁移进度。

**内容：**
- 已迁移方法列表
- 待迁移方法列表
- 迁移策略和优先级
- 使用方式对比
- 统计数据

**适合：** 查看当前迁移进度

---

## 🗂️ 文档结构

```
docs/
├── README.md                              # 本文件
├── EDITING_EXPERIMENT_TODO.md             # 重构规划
├── METHOD_MIGRATION_GUIDE.md              # 迁移指南
├── REFACTORING_SUMMARY.md                 # 重构总结
├── PREPROCESS_IMPLEMENTATION_SUMMARY.md   # 预处理实现
└── METHOD_MIGRATION_STATUS.md             # 迁移状态
```

## 🚀 快速开始

### 新用户
1. 阅读 [REFACTORING_SUMMARY.md](./REFACTORING_SUMMARY.md) 了解整体架构
2. 查看 [METHOD_MIGRATION_STATUS.md](./METHOD_MIGRATION_STATUS.md) 了解当前状态

### 开发者
1. 阅读 [EDITING_EXPERIMENT_TODO.md](./EDITING_EXPERIMENT_TODO.md) 了解设计思路
2. 参考 [METHOD_MIGRATION_GUIDE.md](./METHOD_MIGRATION_GUIDE.md) 进行方法迁移

### 维护者
1. 定期更新 [METHOD_MIGRATION_STATUS.md](./METHOD_MIGRATION_STATUS.md)
2. 在 [REFACTORING_SUMMARY.md](./REFACTORING_SUMMARY.md) 中记录重要变更

## 📊 重构进度

- **阶段 A（统一协议）:** ✅ 完成
- **阶段 B（方法拆离）:** 🔄 进行中（1/7 方法已迁移）
- **阶段 C（实验 Runner）:** ✅ 基本完成

## 🔗 相关资源

- **代码仓库:** `/editing` 目录
- **测试脚本:** `test_preprocess_validation.py`
- **运行入口:** `run_edit_experiment.py`

## 📝 更新日志

- **2026-04-07:** 完成基础框架和第一个方法迁移
- **2026-04-07:** 整理文档到 docs 目录
