# MODEL Knowledge

> 最后更新: 2026-04-24
> 该方向负责 checkpoint 选择、promotion gate、walk-forward 验证和模型保留策略。

## Model Comparison Table

| Profile | Return | Sharpe | MaxDD | AvgPos | 2022 MaxDD | FS | WF | Verdict |
|---------|--------|--------|-------|--------|------------|----|----|---------|
| balanced_rule_residual_v19 | 105 | 2.2 | 22.3 | 0.520 | — | ❌ | ❌ | ✅ Strong |
| balanced | 97.0 | 2.2 | 21.1 | 0.518 | — | ❌ | ❌ | ✅ Strong |
| balanced_rule_residual_mode_v1 | 886 | 1.1 | 42.3 | 0.450 | — | ❌ | ❌ | 🟡 Good |
| balanced_rule_residual_mode_v2 | 694 | 1.0 | 50.4 | 0.450 | — | ❌ | ❌ | 🟡 Good |
| balanced_rule_residual_v19 | 830 | 0.976 | 51.5 | 0.532 | — | ❌ | ❌ | 🟡 Good |
| balanced_rule_residual_v24 | 759 | 0.961 | 54.6 | 0.518 | — | ❌ | ❌ | 🟡 Good |
| risk_budget_usage_v1 | 274 | 0.942 | 32.5 | 0.269 | — | ❌ | ❌ | 🟡 Good |
| balanced_rule_residual_v22 | 695 | 0.941 | 54.5 | 0.509 | — | ❌ | ❌ | 🟡 Good |
| balanced_rule_residual_imitation_v6 | 768 | 0.929 | 55.5 | 0.554 | — | ❌ | ❌ | 🟡 Good |
| balanced_rule_residual_imitation_v2_fix | 741 | 0.925 | 59.1 | 0.555 | — | ❌ | ❌ | 🟡 Good |
| balanced_rule_residual_imitation_v3 | 796 | 0.924 | 59.8 | 0.587 | — | ❌ | ❌ | 🟡 Good |
| balanced_rule_residual_imitation_v4 | 586 | 0.924 | 50.9 | 0.470 | — | ❌ | ❌ | 🟡 Good |
| balanced_rule_residual_imitation_v5 | 668 | 0.916 | 56.6 | 0.528 | — | ❌ | ❌ | 🟡 Good |
| risk_budget_bc_warmstart_v1 | 251 | 0.911 | 27.3 | 0.266 | — | ❌ | ❌ | 🟡 Good |
| balanced_kelly_warmstart_v9 | 584 | 0.911 | 53.4 | 0.485 | — | ❌ | ❌ | 🟡 Good |
| balanced_kelly_warmstart_v9b | 584 | 0.911 | 53.4 | 0.485 | — | ❌ | ❌ | 🟡 Good |
| balanced_floor40_sym_slow_bear | 515 | 0.910 | 46.4 | 0.450 | — | ❌ | ❌ | 🟡 Good |
| balanced_rule_residual_imitation_v2 | 525 | 0.906 | 50.5 | 0.456 | — | ❌ | ❌ | 🟡 Good |
| balanced_rule_residual_minimal_v1 | 570 | 0.904 | 52.5 | 0.484 | — | ❌ | ❌ | 🟡 Good |
| balanced_rule_residual_v17 | 570 | 0.904 | 52.5 | 0.484 | — | ❌ | ❌ | 🟡 Good |
| balanced_rule_residual_v18 | 608 | 0.903 | 51.8 | 0.507 | — | ❌ | ❌ | 🟡 Good |
| balanced_rule_residual_simplified_v1 | 730 | 0.903 | 61.6 | 0.586 | — | ❌ | ❌ | 🟡 Good |
| balanced_kelly_dynamic_v6 | 682 | 0.898 | 60.3 | 0.564 | — | ❌ | ❌ | 🟡 Good |
| balanced_kelly_dynamic_v7 | 682 | 0.898 | 60.3 | 0.564 | — | ❌ | ❌ | 🟡 Good |
| balanced_kelly_execution_v8 | 682 | 0.898 | 60.3 | 0.564 | — | ❌ | ❌ | 🟡 Good |
| balanced_kelly_dynamic_v5 | 682 | 0.898 | 60.3 | 0.564 | — | ❌ | ❌ | 🟡 Good |
| balanced_conservative_ft | 660 | 0.897 | 50.8 | 0.553 | — | ❌ | ❌ | 🟡 Good |
| balanced_rule_residual_v20 | 623 | 0.897 | 53.8 | 0.518 | — | ❌ | ❌ | 🟡 Good |
| balanced_rule_residual_refined_v2 | 590 | 0.895 | 50.3 | 0.507 | — | ❌ | ❌ | 🟡 Good |
| balanced_rule_residual_imitation_v7 | 634 | 0.892 | 55.8 | 0.537 | — | ❌ | ❌ | 🟡 Good |
| balanced_rule_residual_refined_v4 | 562 | 0.891 | 51.0 | 0.489 | — | ❌ | ❌ | 🟡 Good |
| balanced_floor40_sym_velocity | 591 | 0.891 | 60.1 | 0.517 | — | ❌ | ❌ | 🟡 Good |
| balanced_rule_residual_v23 | 493 | 0.887 | 51.2 | 0.456 | — | ❌ | ❌ | 🟡 Good |
| balanced_rule_residual_refined_v3 | 575 | 0.877 | 56.0 | 0.528 | — | ❌ | ❌ | 🟡 Good |
| balanced_floor40_sym_cap25 | 482 | 0.875 | 55.9 | 0.456 | — | ❌ | ❌ | 🟡 Good |
| balanced_rule_residual_learned_mode_v1 | 507 | 0.873 | 50.0 | 0.469 | — | ❌ | ❌ | 🟡 Good |
| balanced_rule_residual_refined_v1 | 535 | 0.870 | 50.1 | 0.501 | — | ❌ | ❌ | 🟡 Good |
| balanced_rule_residual_imitation_v1 | 561 | 0.868 | 54.3 | 0.529 | — | ❌ | ❌ | 🟡 Good |
| balanced | 469 | 0.858 | 50.2 | 0.478 | — | ❌ | ❌ | 🟡 Good |
| balanced_floor40_sym_v2 | 385 | 0.849 | 49.0 | 0.408 | — | ❌ | ❌ | 🟡 Good |
| balanced_floor40_sym_tconv | 542 | 0.846 | 61.6 | 0.550 | — | ❌ | ❌ | 🟡 Good |
| balanced_floor40_sym_crisis30_v2 | 357 | 0.840 | 51.6 | 0.402 | — | ❌ | ❌ | 🟡 Good |
| balanced_floor40 | 614 | 0.839 | 65.7 | 0.642 | — | ❌ | ❌ | 🟡 Good |
| balanced_rule_residual_v19_transition | 519 | 0.824 | 60.9 | 0.568 | — | ❌ | ❌ | 🟡 Good |
| balanced | 407 | 0.816 | 52.8 | 0.483 | — | ❌ | ❌ | 🟡 Good |
| balanced_floor40_sym_velocity_rules | 386 | 0.800 | 62.3 | 0.456 | — | ❌ | ❌ | 🟡 Good |
| balanced_floor40_sym_tight_crash | 397 | 0.797 | 56.5 | 0.478 | — | ❌ | ❌ | 🟡 Moderate |
| balanced_floor40_sym_crisis30 | 375 | 0.795 | 59.9 | 0.468 | — | ❌ | ❌ | 🟡 Moderate |
| balanced_floor40_sym | 444 | 0.794 | 61.5 | 0.555 | — | ❌ | ❌ | 🟡 Moderate |
| balanced_floor40_sym_bear30 | 443 | 0.793 | 61.0 | 0.559 | — | ❌ | ❌ | 🟡 Moderate |
| balanced_rule_residual_v16 | 413 | 0.782 | 55.6 | 0.524 | — | ❌ | ❌ | 🟡 Moderate |
| balanced_floor40_sym_v2 | 447 | 0.775 | 64.8 | 0.605 | — | ❌ | ❌ | 🟡 Moderate |
| balanced_checkpoint_review | 371 | 0.765 | 60.3 | 0.521 | — | ❌ | ❌ | 🟡 Moderate |
| balanced_floor40_sym | 455 | 0.763 | 74.4 | 0.682 | — | ❌ | ❌ | 🟡 Moderate |
| balanced_kelly_attention_v10 | 386 | 0.759 | 59.1 | 0.546 | — | ❌ | ❌ | 🟡 Moderate |
| smoke_tide | 307 | 0.693 | 70.6 | 0.615 | — | ❌ | ❌ | 🟡 Moderate |
| smoke | 298 | 0.683 | 76.8 | 0.727 | — | ❌ | ❌ | 🟡 Moderate |
| balanced_floor40_sym_expectancy | 162 | 0.569 | 65.0 | 0.533 | — | ❌ | ❌ | 🟡 Moderate |
| balanced_floor40_sym_expectancy | 162 | 0.569 | 65.0 | 0.533 | — | ❌ | ❌ | 🟡 Moderate |
| balanced | 1.0 | 0.500 | — | — | — | ❌ | ❌ | ❌ Weak |
| balanced | -14.2 | -0.235 | 33.5 | 0.429 | — | ❌ | ❌ | ❌ Weak |
| balanced | -25.8 | -0.877 | 33.0 | 0.283 | — | ❌ | ❌ | ❌ Weak |
| balanced_rule_residual_v19 | -27.3 | -0.920 | 36.3 | 0.298 | — | ❌ | ❌ | ❌ Weak |

### Verdict Legend

- ✅ **Strong** — Sharpe > 1.5 and MaxDD < 30%
- 🟡 **Good** — Sharpe > 0.8
- 🟡 **Gate-pass** — Passed fixed-split or walk-forward gate
- 🟡 **Moderate** — Sharpe > 0.5
- ❌ **Weak** — Sharpe <= 0.5

## Core Conclusions

1. **当前最佳系列**: `balanced_rule_residual_v19`（canonical 基线）和 `v22`（最新迭代）。
2. **Checkpoint 选择**: validation 最佳 checkpoint 常在 iter 40 而非 iter 80。
3. **Promotion Gate 硬约束**: test_2022 MaxDD < 35% 且 avg_pos < 0.35，当前无模型能同时满足。
4. **Walk-forward 有效**: 2020/2022/2023-2024 三个测试窗口的 walk-forward 验证能暴露过拟合。

## 有效方法 ✅

| 方法 | 验证结果 |
|------|---------|
| Checkpoint @60 (v1) | 优于 @80，trade quality 改善 |
| Walk-forward 三窗口 | 2020/2022/2023-2024 验证暴露过拟合 |

## 无效方法 ❌

| 方法 | 失败原因 |
|------|---------|
| Phase-balanced sampling | 扩大 bull/restart 采样未修复 2022 excess |
| Structural model loop (v1) | 更长 context + hard regime routing，MaxDD 仍 39.86% |

## 待验证方向

- [ ] 改变 stress set 或 episode 采样策略
- [ ] 引入外部对冲机制降低 2022 MaxDD
- [ ] v22 的 walk-forward 完整验证结果