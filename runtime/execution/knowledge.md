# EXECUTION Knowledge

> 最后更新: 2026-04-24
> 该方向负责执行层策略：仓位管理、交易频率控制、mode routing 和 overlay。

## Execution Strategy Comparison Table

| Profile | Return | Sharpe | MaxDD | AvgPos | 2022 MaxDD | FS | WF | Verdict |
|---------|--------|--------|-------|--------|------------|----|----|---------|
| balanced_rule_residual_mode_v1 | 886 | 1.1 | 42.3 | 0.450 | — | ❌ | ❌ | 🟡 Good |
| balanced_rule_residual_mode_v2 | 694 | 1.0 | 50.4 | 0.450 | — | ❌ | ❌ | 🟡 Good |
| balanced_floor40_sym_slow_bear | 515 | 0.910 | 46.4 | 0.450 | — | ❌ | ❌ | 🟡 Good |
| balanced_floor40_sym_velocity | 591 | 0.891 | 60.1 | 0.517 | — | ❌ | ❌ | 🟡 Good |
| balanced_floor40_sym_cap25 | 482 | 0.875 | 55.9 | 0.456 | — | ❌ | ❌ | 🟡 Good |
| balanced_rule_residual_learned_mode_v1 | 507 | 0.873 | 50.0 | 0.469 | — | ❌ | ❌ | 🟡 Good |
| balanced_floor40_sym_v2 | 385 | 0.849 | 49.0 | 0.408 | — | ❌ | ❌ | 🟡 Good |
| balanced_floor40_sym_tconv | 542 | 0.846 | 61.6 | 0.550 | — | ❌ | ❌ | 🟡 Good |
| balanced_floor40_sym_crisis30_v2 | 357 | 0.840 | 51.6 | 0.402 | — | ❌ | ❌ | 🟡 Good |
| balanced_floor40 | 614 | 0.839 | 65.7 | 0.642 | — | ❌ | ❌ | 🟡 Good |
| balanced_floor40_sym_velocity_rules | 386 | 0.800 | 62.3 | 0.456 | — | ❌ | ❌ | 🟡 Good |
| balanced_floor40_sym_tight_crash | 397 | 0.797 | 56.5 | 0.478 | — | ❌ | ❌ | 🟡 Moderate |
| balanced_floor40_sym_crisis30 | 375 | 0.795 | 59.9 | 0.468 | — | ❌ | ❌ | 🟡 Moderate |
| balanced_floor40_sym | 444 | 0.794 | 61.5 | 0.555 | — | ❌ | ❌ | 🟡 Moderate |
| balanced_floor40_sym_bear30 | 443 | 0.793 | 61.0 | 0.559 | — | ❌ | ❌ | 🟡 Moderate |
| balanced_floor40_sym_v2 | 447 | 0.775 | 64.8 | 0.605 | — | ❌ | ❌ | 🟡 Moderate |
| balanced_floor40_sym | 455 | 0.763 | 74.4 | 0.682 | — | ❌ | ❌ | 🟡 Moderate |
| balanced_floor40_sym_expectancy | 162 | 0.569 | 65.0 | 0.533 | — | ❌ | ❌ | 🟡 Moderate |
| balanced_floor40_sym_expectancy | 162 | 0.569 | 65.0 | 0.533 | — | ❌ | ❌ | 🟡 Moderate |

### Verdict Legend

- ✅ **Strong** — Sharpe > 1.5 and MaxDD < 30%
- 🟡 **Good** — Sharpe > 0.8
- 🟡 **Gate-pass** — Passed fixed-split or walk-forward gate
- 🟡 **Moderate** — Sharpe > 0.5
- ❌ **Weak** — Sharpe <= 0.5

## Core Conclusions

1. **Position cap 是最有效的风控工具**: drawdown-based 动态仓位上限（min_cap 从 0.40 起步）比固定仓位更优。
2. **Decision frequency = 3 是 sweet spot**: 降低交易频率到每 3 天一次，2022 MaxDD 从 60.56% 降至 38.4%。
3. **Trade lock（3-day cooldown）有效**: 减少微交易摩擦，RL 交易次数从 1000+ 降至 ~100。
4. **Regime-conditional overlay 有方向性**: vol_stress 特征的 regime gating 改善 stress 态表现。
5. **Generic overlay retuning 是重复工作**: stress/bull 阈值微调已被多次验证为边际收益。

## 有效方法 ✅

| 方法 | 关键参数 | 验证结果 |
|------|---------|---------|
| Symmetric floor 40% | min_cap = 0.40 | A+B 突破: 698% return |
| Decision frequency = 3 | freq = 3 | 2022 MaxDD 38.4% |
| 3-day trade lock | lock = 3 | 摩擦从 1000+ 笔降至 ~100 笔 |

## 无效方法 ❌

| 方法 | 失败原因 |
|------|---------|
| Regime gate (balanced_regime_gate) | 未通过 fixed-split 2022 gate |
| Generic overlay retuning | 仅调整 stress/bull 阈值是重复工作 |

## 待验证方向

- [ ] 显式 release/re-entry 状态的 overlay 机制
- [ ] 动态 decision frequency（stress 时低频，bull 时高频）
- [ ] 仓位上限与 VIX term structure 联动