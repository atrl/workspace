# DATA Knowledge

> 最后更新: 2026-04-24
> 该方向负责数据管道、qlib store、外部数据接入和数据质量监控。

## Data Configuration Comparison Table

| Profile | Return | Sharpe | MaxDD | AvgPos | FS | WF | Feature Note | Verdict |
|---------|--------|--------|-------|--------|----|----|--------------|---------|
| balanced_floor40_sym_velocity | 591 | 0.891 | 60.1 | 0.517 | ❌ | ❌ | Velocity features | 🟡 Good |
| balanced_rule_residual_v19_transition | 519 | 0.824 | 60.9 | 0.568 | ❌ | ❌ | Transition signals | 🟡 Good |
| balanced_floor40_sym_velocity_rules | 386 | 0.800 | 62.3 | 0.456 | ❌ | ❌ | Velocity features | 🟡 Good |

### Verdict Legend

- ✅ **Strong** — Sharpe > 1.5 and MaxDD < 30%
- 🟡 **Good** — Sharpe > 0.8
- 🟡 **Gate-pass** — Passed fixed-split or walk-forward gate
- 🟡 **Moderate** — Sharpe > 0.5
- ❌ **Weak** — Sharpe <= 0.5

## Core Conclusions

1. **SOXL 数据覆盖**: 2010-01-01 至 2025-12-31，包含 OHLCV + VIX。
2. **外部数据**: VIX3M, VIX9D, SKEW, VVIX 已接入（qlib store）。
3. **Feature 稳定性**: 50 core features + 14 rulebase + 7 academic + 7 transition。
4. **VIX-free 替代**: regime_risk_no_vix 在 VIX 缺失时可用。

## 有效方法 ✅

| 方法 | 验证结果 |
|------|---------|
| Qlib store 标准化 | 数据一致性通过 |
| 外部数据对齐 | ffill + 0-fill，避免 NaN 丢行 |

## 待验证方向

- [ ] 多资产联合数据管道（TQQQ/UPRO/SPXL/TECL）
- [ ] 实时数据延迟模拟