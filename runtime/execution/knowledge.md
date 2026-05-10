# EXECUTION Knowledge

> 最后更新: 2026-05-10
> 该方向负责执行层策略：仓位管理、交易频率控制、mode routing 和 overlay。

## 执行策略索引

| Profile | 类别 | Return | Sharpe | MaxDD | AvgPos | 2022 MaxDD | 状态 | 卡片 |
|---------|------|--------|--------|-------|--------|------------|------|------|
| balanced_floor40 | cap | 614 | 0.84 | 65.68 | 0.64 | — | 🟡 Good | [card](cards/balanced_floor40/card.md) |
| balanced_floor40_sym_bear30 | cap | 443 | 0.79 | 61.05 | 0.56 | — | 🟡 Moderate | [card](cards/balanced_floor40_sym_bear30/card.md) |
| balanced_floor40_sym_cap25 | cap | 482 | 0.87 | 55.88 | 0.46 | — | 🟡 Good | [card](cards/balanced_floor40_sym_cap25/card.md) |
| balanced_floor40_sym_crisis30 | cap | 375 | 0.79 | 59.87 | 0.47 | — | 🟡 Moderate | [card](cards/balanced_floor40_sym_crisis30/card.md) |
| balanced_floor40_sym_crisis30_v2 | cap | 357 | 0.84 | 51.62 | 0.40 | — | 🟡 Good | [card](cards/balanced_floor40_sym_crisis30_v2/card.md) |
| balanced_floor40_sym_expectancy | cap | 162 | 0.57 | 65.05 | 0.53 | — | 🟡 Moderate | [card](cards/balanced_floor40_sym_expectancy/card.md) |
| balanced_floor40_sym_expectancy | cap | 162 | 0.57 | 65.05 | 0.53 | — | 🟡 Moderate | [card](cards/balanced_floor40_sym_expectancy/card.md) |
| balanced_floor40_sym | cap | 455 | 0.76 | 74.39 | 0.68 | — | 🟡 Moderate | [card](cards/balanced_floor40_sym/card.md) |
| balanced_floor40_sym_slow_bear | cap | 515 | 0.91 | 46.45 | 0.45 | — | 🟡 Good | [card](cards/balanced_floor40_sym_slow_bear/card.md) |
| balanced_floor40_sym | cap | 444 | 0.79 | 61.55 | 0.55 | — | 🟡 Moderate | [card](cards/balanced_floor40_sym/card.md) |
| balanced_floor40_sym_tconv | cap | 542 | 0.85 | 61.58 | 0.55 | — | 🟡 Good | [card](cards/balanced_floor40_sym_tconv/card.md) |
| balanced_floor40_sym_tight_crash | cap | 397 | 0.80 | 56.53 | 0.48 | — | 🟡 Moderate | [card](cards/balanced_floor40_sym_tight_crash/card.md) |
| balanced_floor40_sym_v2 | cap | 385 | 0.85 | 49.01 | 0.41 | — | 🟡 Good | [card](cards/balanced_floor40_sym_v2/card.md) |
| balanced_floor40_sym_v2 | cap | 447 | 0.78 | 64.77 | 0.60 | — | 🟡 Moderate | [card](cards/balanced_floor40_sym_v2/card.md) |
| balanced_floor40_sym_velocity_rules | cap | 386 | 0.80 | 62.34 | 0.46 | — | 🟡 Good | [card](cards/balanced_floor40_sym_velocity_rules/card.md) |
| balanced_floor40_sym_velocity | cap | 591 | 0.89 | 60.14 | 0.52 | — | 🟡 Good | [card](cards/balanced_floor40_sym_velocity/card.md) |
| balanced_rule_residual_learned_mode_v1 | mode | 507 | 0.87 | 50.01 | 0.47 | — | 🟡 Good | [card](cards/balanced_rule_residual_learned_mode_v1/card.md) |
| balanced_rule_residual_mode_v1 | mode | 886 | 1.09 | 42.32 | 0.45 | — | 🟡 Good | [card](cards/balanced_rule_residual_mode_v1/card.md) |
| balanced_rule_residual_mode_v2 | mode | 694 | 1.01 | 50.40 | 0.45 | — | 🟡 Good | [card](cards/balanced_rule_residual_mode_v2/card.md) |

## 核心结论

1. **Position cap 是最有效的风控工具**。
2. **Decision frequency = 3 是 sweet spot**。
3. **Trade lock（3-day cooldown）有效**。
4. **Regime-conditional overlay 有方向性**。
5. **Generic overlay retuning 是重复工作**。
6. **Generic `fast_reentry` 不是答案**：它能改善恢复期收益，但在 proxy Sharpe / MaxDD 上被 `dwell_hold_v2` 压制。
7. **当前瓶颈是低换手的 selective release / re-entry classification**：`regime_route_v1` 证明更聪明的放行有价值，但 churn 明显升高。

## 待验证方向

- [ ] 显式 selective release/re-entry overlay（优先 `dwell_hold_v2` / `regime_route_v1` 风格，不再泛化搜索 `fast_reentry`）
- [ ] 动态 decision frequency
- [ ] 仓位上限与 VIX term structure 联动
- [ ] budget-usage actor/runtime mismatch 诊断
