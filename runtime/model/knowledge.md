# MODEL Knowledge

> 最后更新: 2026-04-24
> 该方向负责 checkpoint 选择、promotion gate、walk-forward 验证和模型保留策略。

## 模型索引

| Profile | Return | Sharpe | MaxDD | AvgPos | 2022 MaxDD | FS | WF | 状态 | 卡片 |
|---------|--------|--------|-------|--------|------------|----|----|------|------|
| balanced_rule_residual_v19 | 105 | 2.24 | 22.26 | 0.52 | — | ❌ | ❌ | ✅ Strong | [card](cards/balanced_rule_residual_v19/card.md) |
| balanced | 96.96 | 2.16 | 21.12 | 0.52 | — | ❌ | ❌ | ✅ Strong | [card](cards/balanced/card.md) |
| balanced_rule_residual_mode_v1 | 886 | 1.09 | 42.32 | 0.45 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_rule_residual_mode_v1/card.md) |
| balanced_rule_residual_mode_v2 | 694 | 1.01 | 50.40 | 0.45 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_rule_residual_mode_v2/card.md) |
| balanced_rule_residual_v19 | 830 | 0.98 | 51.47 | 0.53 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_rule_residual_v19/card.md) |
| balanced_rule_residual_v24 | 759 | 0.96 | 54.58 | 0.52 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_rule_residual_v24/card.md) |
| risk_budget_usage_v1 | 274 | 0.94 | 32.49 | 0.27 | — | ❌ | ❌ | 🟡 Good | [card](cards/risk_budget_usage_v1/card.md) |
| balanced_rule_residual_v22 | 695 | 0.94 | 54.52 | 0.51 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_rule_residual_v22/card.md) |
| balanced_rule_residual_imitation_v6 | 768 | 0.93 | 55.50 | 0.55 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_rule_residual_imitation_v6/card.md) |
| balanced_rule_residual_imitation_v2_fix | 741 | 0.92 | 59.11 | 0.55 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_rule_residual_imitation_v2_fix/card.md) |
| balanced_rule_residual_imitation_v3 | 796 | 0.92 | 59.83 | 0.59 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_rule_residual_imitation_v3/card.md) |
| balanced_rule_residual_imitation_v4 | 586 | 0.92 | 50.95 | 0.47 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_rule_residual_imitation_v4/card.md) |
| balanced_rule_residual_imitation_v5 | 668 | 0.92 | 56.61 | 0.53 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_rule_residual_imitation_v5/card.md) |
| risk_budget_bc_warmstart_v1 | 251 | 0.91 | 27.27 | 0.27 | — | ❌ | ❌ | 🟡 Good | [card](cards/risk_budget_bc_warmstart_v1/card.md) |
| balanced_kelly_warmstart_v9 | 584 | 0.91 | 53.44 | 0.48 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_kelly_warmstart_v9/card.md) |
| balanced_kelly_warmstart_v9b | 584 | 0.91 | 53.44 | 0.48 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_kelly_warmstart_v9b/card.md) |
| balanced_floor40_sym_slow_bear | 515 | 0.91 | 46.45 | 0.45 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_floor40_sym_slow_bear/card.md) |
| balanced_rule_residual_imitation_v2 | 525 | 0.91 | 50.51 | 0.46 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_rule_residual_imitation_v2/card.md) |
| balanced_rule_residual_minimal_v1 | 570 | 0.90 | 52.52 | 0.48 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_rule_residual_minimal_v1/card.md) |
| balanced_rule_residual_v17 | 570 | 0.90 | 52.52 | 0.48 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_rule_residual_v17/card.md) |
| balanced_rule_residual_v18 | 608 | 0.90 | 51.84 | 0.51 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_rule_residual_v18/card.md) |
| balanced_rule_residual_simplified_v1 | 730 | 0.90 | 61.63 | 0.59 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_rule_residual_simplified_v1/card.md) |
| balanced_kelly_dynamic_v6 | 682 | 0.90 | 60.34 | 0.56 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_kelly_dynamic_v6/card.md) |
| balanced_kelly_dynamic_v7 | 682 | 0.90 | 60.34 | 0.56 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_kelly_dynamic_v7/card.md) |
| balanced_kelly_execution_v8 | 682 | 0.90 | 60.34 | 0.56 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_kelly_execution_v8/card.md) |
| balanced_kelly_dynamic_v5 | 682 | 0.90 | 60.34 | 0.56 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_kelly_dynamic_v5/card.md) |
| balanced_conservative_ft | 660 | 0.90 | 50.79 | 0.55 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_conservative_ft/card.md) |
| balanced_rule_residual_v20 | 623 | 0.90 | 53.85 | 0.52 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_rule_residual_v20/card.md) |
| balanced_rule_residual_refined_v2 | 590 | 0.89 | 50.27 | 0.51 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_rule_residual_refined_v2/card.md) |
| balanced_rule_residual_imitation_v7 | 634 | 0.89 | 55.75 | 0.54 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_rule_residual_imitation_v7/card.md) |
| balanced_rule_residual_refined_v4 | 562 | 0.89 | 50.99 | 0.49 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_rule_residual_refined_v4/card.md) |
| balanced_floor40_sym_velocity | 591 | 0.89 | 60.14 | 0.52 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_floor40_sym_velocity/card.md) |
| balanced_rule_residual_v23 | 493 | 0.89 | 51.20 | 0.46 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_rule_residual_v23/card.md) |
| balanced_rule_residual_refined_v3 | 575 | 0.88 | 55.97 | 0.53 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_rule_residual_refined_v3/card.md) |
| balanced_floor40_sym_cap25 | 482 | 0.87 | 55.88 | 0.46 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_floor40_sym_cap25/card.md) |
| balanced_rule_residual_learned_mode_v1 | 507 | 0.87 | 50.01 | 0.47 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_rule_residual_learned_mode_v1/card.md) |
| balanced_rule_residual_refined_v1 | 535 | 0.87 | 50.05 | 0.50 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_rule_residual_refined_v1/card.md) |
| balanced_rule_residual_imitation_v1 | 561 | 0.87 | 54.34 | 0.53 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_rule_residual_imitation_v1/card.md) |
| balanced | 469 | 0.86 | 50.18 | 0.48 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced/card.md) |
| balanced_floor40_sym_v2 | 385 | 0.85 | 49.01 | 0.41 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_floor40_sym_v2/card.md) |
| balanced_floor40_sym_tconv | 542 | 0.85 | 61.58 | 0.55 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_floor40_sym_tconv/card.md) |
| balanced_floor40_sym_crisis30_v2 | 357 | 0.84 | 51.62 | 0.40 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_floor40_sym_crisis30_v2/card.md) |
| balanced_floor40 | 614 | 0.84 | 65.68 | 0.64 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_floor40/card.md) |
| balanced_rule_residual_v19_transition | 519 | 0.82 | 60.89 | 0.57 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_rule_residual_v19_transition/card.md) |
| balanced | 407 | 0.82 | 52.75 | 0.48 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced/card.md) |
| balanced_floor40_sym_velocity_rules | 386 | 0.80 | 62.34 | 0.46 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_floor40_sym_velocity_rules/card.md) |
| balanced_floor40_sym_tight_crash | 397 | 0.80 | 56.53 | 0.48 | — | ❌ | ❌ | 🟡 Moderate | [card](cards/balanced_floor40_sym_tight_crash/card.md) |
| balanced_floor40_sym_crisis30 | 375 | 0.79 | 59.87 | 0.47 | — | ❌ | ❌ | 🟡 Moderate | [card](cards/balanced_floor40_sym_crisis30/card.md) |
| balanced_floor40_sym | 444 | 0.79 | 61.55 | 0.55 | — | ❌ | ❌ | 🟡 Moderate | [card](cards/balanced_floor40_sym/card.md) |
| balanced_floor40_sym_bear30 | 443 | 0.79 | 61.05 | 0.56 | — | ❌ | ❌ | 🟡 Moderate | [card](cards/balanced_floor40_sym_bear30/card.md) |
| balanced_rule_residual_v16 | 413 | 0.78 | 55.62 | 0.52 | — | ❌ | ❌ | 🟡 Moderate | [card](cards/balanced_rule_residual_v16/card.md) |
| balanced_floor40_sym_v2 | 447 | 0.78 | 64.77 | 0.60 | — | ❌ | ❌ | 🟡 Moderate | [card](cards/balanced_floor40_sym_v2/card.md) |
| balanced_checkpoint_review | 371 | 0.76 | 60.26 | 0.52 | — | ❌ | ❌ | 🟡 Moderate | [card](cards/balanced_checkpoint_review/card.md) |
| balanced_floor40_sym | 455 | 0.76 | 74.39 | 0.68 | — | ❌ | ❌ | 🟡 Moderate | [card](cards/balanced_floor40_sym/card.md) |
| balanced_kelly_attention_v10 | 386 | 0.76 | 59.12 | 0.55 | — | ❌ | ❌ | 🟡 Moderate | [card](cards/balanced_kelly_attention_v10/card.md) |
| smoke_tide | 307 | 0.69 | 70.60 | 0.62 | — | ❌ | ❌ | 🟡 Moderate | [card](cards/smoke_tide/card.md) |
| smoke | 298 | 0.68 | 76.83 | 0.73 | — | ❌ | ❌ | 🟡 Moderate | [card](cards/smoke/card.md) |
| balanced_floor40_sym_expectancy | 162 | 0.57 | 65.05 | 0.53 | — | ❌ | ❌ | 🟡 Moderate | [card](cards/balanced_floor40_sym_expectancy/card.md) |
| balanced_floor40_sym_expectancy | 162 | 0.57 | 65.05 | 0.53 | — | ❌ | ❌ | 🟡 Moderate | [card](cards/balanced_floor40_sym_expectancy/card.md) |
| balanced | 1.00 | 0.50 | — | — | — | ❌ | ❌ | ❌ Weak | [card](cards/balanced/card.md) |
| balanced | -14.20 | -0.23 | 33.51 | 0.43 | — | ❌ | ❌ | ❌ Weak | [card](cards/balanced/card.md) |
| balanced | -25.82 | -0.88 | 33.04 | 0.28 | — | ❌ | ❌ | ❌ Weak | [card](cards/balanced/card.md) |
| balanced_rule_residual_v19 | -27.27 | -0.92 | 36.32 | 0.30 | — | ❌ | ❌ | ❌ Weak | [card](cards/balanced_rule_residual_v19/card.md) |

### 状态图例

- ✅ **Strong** — Sharpe > 1.5 and MaxDD < 30%
- 🟡 **Good** — Sharpe > 0.8
- 🟡 **Gate-pass** — Passed fixed-split or walk-forward gate
- 🟡 **Moderate** — Sharpe > 0.5
- ❌ **Weak** — Sharpe <= 0.5
- ⚪ **Unverified** — No metrics

## 核心结论

1. **当前最佳系列**: （canonical 基线）和 （最新迭代）。
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
## 诊断报告

| 报告 | 内容 | 文件 |
|------|------|------|
| 模型调整路线图 | 硬底仓+RL战术叠加层、特征盲区、微幅调仓问题 | [diagnosis/model_adjustment_roadmap.md](diagnosis/model_adjustment_roadmap.md) |
| 新特征扫描 | VIX期限结构、期权隐含特征、外部数据源 | [diagnosis/new_feature_scan.md](diagnosis/new_feature_scan.md) |
| RL vs Rule-Based 深度复盘 | 交易哲学差距、持仓模式对比 | [diagnosis/deep_dive_rl_vs_rulebased.md](diagnosis/deep_dive_rl_vs_rulebased.md) |
| 时段切片诊断 | 年度/季度表现归因 | [diagnosis/period_slice_comparison.md](diagnosis/period_slice_comparison.md) |
| 交易级归因 | Kelly诊断、分层分析 | [diagnosis/soxl_trade_attribution.md](diagnosis/soxl_trade_attribution.md) |

## 实验包索引

以下 `experiments/` 目录中的实验包保留完整配置、动机和迭代历史，与上表中的运行时卡片一一对应。

| 实验包 | 类型 | 包含 Profile | 状态 |
|--------|------|-------------|------|
| [`balanced`](experiments/canonical/balanced/) | canonical | balanced | 🔵 基线 |
| [`smoke`](experiments/canonical/smoke/) | canonical | smoke | 🔵 基线 |
| [`balanced_rule_residual_v19`](experiments/candidate/balanced_rule_residual_v19/) | candidate | v19 | 🟢 优秀 |
| [`balanced_rule_residual_v22`](experiments/candidate/balanced_rule_residual_v22/) | candidate | v22 系列 | 🟡 迭代 |
| [`balanced_rule_residual_refined_v2`](experiments/candidate/balanced_rule_residual_refined_v2/) | candidate | refined_v2 | 🟡 迭代 |
| [`balanced_rule_residual_refined_v3`](experiments/candidate/balanced_rule_residual_refined_v3/) | candidate | refined_v3 | 🟡 迭代 |
| [`balanced_rule_residual_mode_v1`](experiments/candidate/balanced_rule_residual_mode_v1/) | candidate | mode_v1/v2 | 🟡 迭代 |
| [`balanced_rule_residual_imitation`](experiments/candidate/balanced_rule_residual_imitation/) | candidate | imitation v1–v7 (合并) | 🟡 迭代 |
| [`balanced_kelly`](experiments/candidate/balanced_kelly/) | candidate | kelly v5–v10 (合并) | 🟡 迭代 |
| [`balanced_conservative_ft`](experiments/candidate/balanced_conservative_ft/) | candidate | conservative_ft | 🟡 迭代 |
| [`risk_budget_usage_v1`](experiments/candidate/risk_budget_usage_v1/) | candidate | risk_budget_usage_v1, bc_warmstart_v1 | 🟡 最佳候选 |
| [`risk_budget_usage_v2`](experiments/candidate/risk_budget_usage_v2/) | candidate | risk_budget_usage_v2 | 🟡 扩展中 |
| [`balanced_rule_residual_v19_academic`](experiments/candidate/balanced_rule_residual_v19_academic/) | candidate | v19_academic | ⚪ 未运行 |
| [`balanced_rule_residual_moderate_v1`](experiments/candidate/balanced_rule_residual_moderate_v1/) | candidate | moderate_v1 | ⚪ 未运行 |
| [`smoke_rule_residual_v1`](experiments/candidate/smoke_rule_residual_v1/) | candidate | smoke_rule_residual | ⚪ 未运行 |
