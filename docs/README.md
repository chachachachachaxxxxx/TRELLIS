# TRELLIS 编辑实验框架文档

本目录包含 TRELLIS 编辑实验框架重构的核心文档。

## 📚 核心文档

### [REFACTORING_STATUS.md](./REFACTORING_STATUS.md)
**重构状态报告** - 最权威的重构进度和状态文档

包含：
- 完整的重构进度（71% 完成）
- 已迁移方法列表和测试状态
- 待完成工作清单
- 使用示例

**推荐：** 想了解当前状态时首先查看此文档

---

### [METHOD_MIGRATION_GUIDE.md](./METHOD_MIGRATION_GUIDE.md)
**方法迁移指南** - 如何添加新的编辑方法

包含：
- EditMethod 接口说明
- 迁移步骤详解
- 代码示例
- 最佳实践

**推荐：** 准备添加新方法时参考

---

### [REFACTORING_TEST_REPORT.md](./REFACTORING_TEST_REPORT.md)
**测试报告** - 框架验证和测试结果

包含：
- 测试命令和结果
- 已修复的问题
- 验证结论

---

### [MEMORY_OPTIMIZATION.md](./MEMORY_OPTIMIZATION.md)
**显存优化指南** - 通用的显存优化技术

包含：
- 优化策略和技术
- 适用场景
- 实施建议

---

### [UNIEDIT_MEMORY_OPTIMIZATION.md](./UNIEDIT_MEMORY_OPTIMIZATION.md)
**UniEdit 显存优化** - UniEdit 方法的专项优化

包含：
- UniEdit 特定的优化措施
- 优化效果对比
- 使用建议

---

### [UNIEDIT_COMPLETION_SUMMARY.md](./UNIEDIT_COMPLETION_SUMMARY.md)
**UniEdit 完成总结** - UniEdit 方法实现的完整报告

包含：
- 实现细节
- 测试结果
- 使用方法
- 技术亮点

---

## 🗂️ 文档结构

```
docs/
├── README.md                           # 本文件
├── REFACTORING_STATUS.md               # ⭐ 重构状态（主文档）
├── METHOD_MIGRATION_GUIDE.md           # 方法迁移指南
├── REFACTORING_TEST_REPORT.md          # 测试报告
├── MEMORY_OPTIMIZATION.md              # 通用显存优化
├── UNIEDIT_MEMORY_OPTIMIZATION.md      # UniEdit 显存优化
└── UNIEDIT_COMPLETION_SUMMARY.md       # UniEdit 完成总结
```

## 🚀 快速开始

### 新用户
查看 [REFACTORING_STATUS.md](./REFACTORING_STATUS.md) 了解当前状态和使用方法

### 开发者
参考 [METHOD_MIGRATION_GUIDE.md](./METHOD_MIGRATION_GUIDE.md) 添加新方法

### 优化显存
查看 [MEMORY_OPTIMIZATION.md](./MEMORY_OPTIMIZATION.md) 和 [UNIEDIT_MEMORY_OPTIMIZATION.md](./UNIEDIT_MEMORY_OPTIMIZATION.md)

## 📊 当前状态

**重构进度**: 71% 完成
- ✅ 核心框架: 100%
- ✅ 方法迁移: 5/7 (71%)
- ✅ 测试通过: 3/5 (60%)
- ✅ 文档完善: 100%

详见 [REFACTORING_STATUS.md](./REFACTORING_STATUS.md)

## 🔗 相关资源

- **代码**: `editing/` 目录
- **运行入口**: `run_edit_experiment.py`
- **Wiki**: `wiki/` 目录

## 📝 更新日志

- **2026-04-08:** 完成 UniEdit 方法实现和显存优化
- **2026-04-08:** 清理过时文档，更新文档索引
- **2026-04-07:** 完成基础框架和第一个方法迁移
