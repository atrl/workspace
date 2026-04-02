# HEARTBEAT.md - 心跳检查清单

## 📋 当前状态 (2026-03-23 23:34)

**身份**: clawQuant - 量化工具人，只用 Qlib 框架 ✅

**🎉 项目完成** (已归档):
- ✅ V25策略部署成功
- ✅ Sharpe 0.96, Max DD -8.69%
- ✅ OOS验证通过 (Sharpe 0.92)
- ✅ 完整文档已提交 (21:06)

**RL项目总结**:
- ❌ 分钟级RL失败 (过拟合)
- ✅ 回退日级V25策略成功
- 💡 教训: 简单有效 > 复杂失效

**项目状态**: ✅ **已归档** - 等待纸交易验证
**上次心跳**: 2026-04-02 13:57
**检查结果**: 无新任务，项目持续归档状态，等待纸交易验证

**关键文件**:
- V25: `/Users/rziiiii/.openclaw-gemini/workspace/production/soxl_kelly_v25_production.py`
- 文档: `memory/2026-03-23.md` (49KB完整记录)

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
