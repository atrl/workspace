# HEARTBEAT.md - 心跳检查清单

## 📋 当前状态 (2026-03-23 19:51)

**身份**: clawQuant - 量化工具人，只用 Qlib 框架 ✅

**🎉 项目完成**:
- ✅ V25策略部署成功 (19:38)
- ✅ Sharpe 0.96, Max DD -8.69%
- ✅ OOS验证通过 (Sharpe 0.92)
- ✅ 相比V13: Sharpe +85%, Max DD +77%

**RL项目总结**:
- ❌ 分钟级RL失败 (过拟合)
- ✅ 回退日级V25策略成功
- 💡 教训: 简单有效 > 复杂失效

**下一步**:
- 🔄 纸交易验证 (1-2周)
- 🚀 实盘部署 (10%初始仓位)
- 📊 持续监控

**关键文件**:
- V25: `/Users/rziiiii/.openclaw-gemini/workspace/production/soxl_kelly_v25_production.py`
- 结果: `research/v8_v25_risk_optimized_results.md`

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
