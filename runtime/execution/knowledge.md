# EXECUTION Knowledge

> 最后更新: 2026-05-14
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
| selective_release_overlay_v1 | overlay | 1415 | 1.11 | 54.30 | 0.52 | 40.64 | 🟢 Counterfactual | [card](card/selective_release_overlay_v1/selective_release_overlay_review.md) |

## 核心结论

1. **Position cap 是最有效的风控工具**。
2. **Decision frequency = 3 是 sweet spot**。
3. **Trade lock（3-day cooldown）有效**。
4. **Regime-conditional overlay 有方向性**。
5. **Generic overlay retuning 是重复工作**。
6. **Generic `fast_reentry` 不是答案**：它能改善恢复期收益，但在 proxy Sharpe / MaxDD 上被 `dwell_hold_v2` 压制。
7. **当前瓶颈是低换手的 selective release / re-entry classification**：`regime_route_v1` 证明更聪明的放行有价值，但 churn 明显升高。
8. **固定 v22 actor 的 selective release gate 已证明方向正确**：`selective_release_recovery_v1` 把 `test_2023_2024` 从 `57.12% / 0.719` 提到 `82.10% / 0.877`，`test_2022` 完全不变，`test_proxy` 从 `1132.19% / 1.043 / 56.48%` 提到 `1415.10% / 1.105 / 54.30%`。
9. **walk-forward actor 验证未通过 runtime promotion**：`selective_release_walkforward_v1` 显示三种既有 release gate 在 `wf_2023_2024` 有 `6-18` 个 eligible days，但 `changed_days=0`，说明当前执行映射吸收了 boost；不要直接治理化该 overlay。

## 待验证方向

- [x] 显式 selective release/re-entry overlay（优先 `dwell_hold_v2` / `regime_route_v1` 风格，不再泛化搜索 `fast_reentry`）— fixed v22 actor counterfactual retained as `selective_release_recovery_v1`
- [x] 对 `selective_release_recovery_v1` 做 walk-forward actor validation，再决定是否进入 runtime overlay / governed execution profile — blocked for promotion because WF actor orders do not change
- [ ] 诊断 `wf_2023_2024` eligible release days 为什么被 actor target / rebalance threshold / trade lock 吸收
- [ ] 动态 decision frequency
- [ ] 仓位上限与 VIX term structure 联动
- [ ] budget-usage actor/runtime mismatch 诊断

## 2026-05-14: Selective Release Overlay Counterfactual

Question: can retained release factors improve recovery if they are gated in execution instead of injected directly into PPO state?

Answer: yes on the fixed `balanced_rule_residual_v22` actor.

| Variant | test_proxy | test_2022 | test_2023_2024 | Decision |
|---|---:|---:|---:|---|
| baseline | `1132.19% / 1.043 / 56.48% / avg_pos 0.5226` | `-30.97% / -0.764 / 40.64% / avg_pos 0.3304` | `57.12% / 0.719 / 34.87% / avg_pos 0.5402` | control |
| `selective_release_recovery_v1` | `1415.10% / 1.105 / 54.30% / avg_pos 0.5230` | unchanged | `82.10% / 0.877 / 36.95% / avg_pos 0.5510` | retain counterfactual |

Mechanism:

- Gate requires `RB_COOLDOWN_PROXY` plus either `TRANS_VIX_DIVERGENCE` or `CICC_RETURN_VOLUME_SYNC`.
- The best retained config uses `boost=0.08`, `max_target=0.65`, `max_regime_risk=0.60`, `cooldown_threshold=0.20`, `vix_threshold=0.05`, `volume_sync_threshold=0.70`.
- It does not fire in `2022`, which is why bear-market exposure and drawdown remain unchanged.

Decision:

- Do not retrain another release-pack PPO branch yet.
- Next valid loop is walk-forward actor validation for the same overlay.
- If it survives, encode it as a governed execution overlay/profile rather than expanding the model feature pack.

## 2026-05-15: Selective Release Walk-Forward Actor Validation

Question: does the retained release overlay survive when applied to saved `balanced_rule_residual_v22` walk-forward actors?

Answer: no, not as an executable runtime overlay yet.

| Variant | wf_2020 | wf_2022 | wf_2023_2024 | Decision |
|---|---:|---:|---:|---|
| baseline | `113.17% / 2.015 / 19.47%` | `-34.11% / -0.926 / 38.27%` | `-20.21% / -0.277 / 38.54%` | control |
| `selective_release_recovery_v1` | unchanged, `0` eligible days | unchanged, `0` eligible days | unchanged, `18` eligible days but `0` changed days | block runtime promotion |

Mechanism:

- The release signals do identify 2024 recovery days, so the factor side is not the primary failure.
- The overlay boost is absorbed by actor target / execution mapping, because eligible days do not translate into changed orders.

Decision:

- Keep `selective_release_overlay_v1` as fixed full-window counterfactual evidence only.
- Do not promote it to a governed runtime profile until an explicit target-lift / rebalance-bypass test changes orders while preserving `wf_2022`.
