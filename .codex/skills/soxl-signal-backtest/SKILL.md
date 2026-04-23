---
name: soxl-signal-backtest
description: SOXL 单信号/多信号组合的独立回测验证层。不依赖 RL 训练流程，直接对时序信号进行历史模拟交易、分体制切片分析、与 rulebase 策略对比。
---

# SOXL 信号独立回测（Signal Backtest）

本 skill 负责对候选特征信号进行独立的历史回测验证，是特征入库前的最后一道关卡。

## 设计原则

1. **完全独立**：不依赖 `src/soxl_quant/policy/` 或 RL 训练流程。
2. **简单直接**：信号 → 仓位规则 → 模拟交易 → 收益统计。
3. **体制感知**：必须分 bull / base / stress 体制报告信号表现。

## 核心脚本

### `signal_backtest.py`

单信号历史回测。核心逻辑：

```
信号值 > 做多阈值 → 满仓 (或加仓)
信号值 < 做空阈值 → 空仓 (或减仓)
中间区域 → 保持上一期仓位
```

支持参数扫描：阈值 × 持有期 × 交易成本。

### `multi_signal_ensemble.py`

多信号组合回测。支持：
- 等权投票
- ICIR 加权
- 逻辑与/或组合

### `regime_slice_analysis.py`

分体制切片分析。按 `execution_regime()` 的输出标签（bull/base/stress）分别统计：
- 信号在每种体制下的 IC
- 信号在每种体制下的胜率
- 信号在每种体制下的盈亏比

### `compare_to_rulebase.py`

将新信号的回测曲线与 `soxl_backtest/` 的 rulebase 策略直接对比：
- 收益率曲线重叠图
- 关键指标对比表
- 交易时机差异分析

## 指标标准

| 指标 | 通过门槛 | 说明 |
|------|---------|------|
| annualized_return | > 0% | 信号策略年化收益为正 |
| max_drawdown | < 40% | 最大回撤控制 |
| sharpe | > 0.3 | 基础风险调整收益 |
| win_rate | > 0.50 | 胜率过半 |
| profit_factor | > 1.2 | 盈亏比 |
| turnover | < 5x/年 | 换手率控制 |

## 输出

- `docs/factor_research/signal_validation_report.md` 模板
- `data/soxl_factor_library.db` 的 `feature_backtests` 表
