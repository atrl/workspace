---
name: leveraged-etf-factor-research
description: 3x杠杆ETF特征假设研究与生成。默认标的SOXL，支持TQQQ/UPRO/SPXL/TECL等。从rulebase策略逻辑、量化研报、历史实验中系统性提取特征构造思路，生成结构化的特征假设卡片供挖掘流程使用。
version: 2.0.0
changelog:
  - 2.0.0: 重命名并适配多ETF，更新数据来源
  - 1.0.0: SOXL专用初始版本
---

# 3x杠杆ETF特征研报研究（Factor Research）

本skill负责从`soxl_backtest/`的rulebase逻辑和历史实验中提取特征假设。
默认标的为SOXL，所有特征逻辑可泛化至其他3x杠杆ETF（TQQQ/UPRO/SPXL/TECL）。

## 核心来源

### 1. Rulebase信号拆解

`soxl_backtest/`的rulebase策略包含丰富的手工构造信号：

- **Panic Score**：sigmoid(VIX) × volume_multiplier + RSI6_adjustment
- **买入条件**：日跌幅/累积跌幅 + 波动率系数 + 冷却期
- **卖出条件**：止盈/止损 + 恐慌指数阈值
- **仓位管理**：MIN_POSITION=0.50, FIFO持仓栈

这些逻辑可以被拆解为独立的时序特征，并用数据验证其预测能力。
不同3xETF的波动率特征不同，阈值需按标的重新校准。

### 2. 历史实验记忆

`runtime/factor/knowledge.md`和`experiments/`中记录的失败/成功案例包含特征改进方向。

### 3. 外部研报

通用量化因子（动量、反转、波动率、流动性）在LETF上的适配。

## 输出

特征假设卡片（YAML）：

```yaml
factor_hypothesis:
  name: "RB_PANIC_SCORE_PROXY"
  expression: "sigmoid(vix_zscore) * relative_volume_zscore + rsi6_adjustment"
  category: "sentiment"
  source: "rulebase_panic_calculator"
  market_logic: "VIX驱动的恐慌情绪 + 成交量放大 + RSI6超卖调节"
  expected_direction: "negative"  # 恐慌指数高 → 未来反弹
  expected_horizon: "5d"
  priority: "high"
  risks: "VIX数据缺失时失效"
  applicable_symbols: ["SOXL", "TQQQ", "UPRO", "SPXL", "TECL"]
```

## 可复用脚本

- `generate_hypothesis.py` — 从rulebase自动提取假设卡片，支持`--symbol`参数
