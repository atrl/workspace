# HEARTBEAT.md - 心跳检查清单

## 📋 当前状态 (2026-03-22 17:19)

**身份**: clawQuant - 量化工具人，只用 Qlib 框架 ✅

**已完成**:
- ✅ IDENTITY.md 更新 (clawQuant 身份)
- ✅ SOUL.md 更新 (执行优先，自己解决问题)
- ✅ TOOLS.md 更新 (Qlib 使用规范 v2.0)
  - 5 个核心 Use Case
  - 踩坑记录
  - 代码模板
  - 环境配置

**待清理**:
- ⏳ memory/ 目录旧记录

---

## 🔄 心跳职责

每次心跳必须：
1. **报告当前任务进展** - 正在做什么，完成到哪一步
2. **报告阻塞问题** - 有什么卡住的
3. **报告下一步行动** - 接下来要做什么

**禁止**：只回复 "HEARTBEAT_OK" 而不更新状态

---

## 📚 Qlib 核心 Use Cases

1. **数据加载** - `D.features()` / `QlibDataLoader`
2. **因子计算** - Qlib Expression (`Ref`, `Mean`, `Std`, `If`)
3. **策略实现** - 继承 `BaseStrategy`
4. **回测** - `backtest()` + `risk_analysis()`
5. **模型训练** - `LGBModel` + `DatasetH`

---

## 🏆 PK V2 已完成 (归档)

**冠军**: clawQuant V25 (Sharpe 0.96, MaxDD -8.69%)
**文件**: `production/soxl_kelly_v25_production.py`
**状态**: ✅ 归档，等待 Paper Trading
