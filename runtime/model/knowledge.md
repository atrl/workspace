# MODEL Knowledge

> 最后更新: 2026-05-21
> 该方向负责 checkpoint 选择、promotion gate、walk-forward 验证和模型保留策略。

## 模型索引

| Profile | Return | Sharpe | MaxDD | AvgPos | 2022 MaxDD | FS | WF | 状态 | 卡片 |
|---------|--------|--------|-------|--------|------------|----|----|------|------|
| balanced_rule_residual_v19 | 105 | 2.24 | 22.26 | 0.52 | — | ❌ | ❌ | ✅ Strong | [card](experiments/candidate/balanced_rule_residual_v19/card.md) |
| balanced | 96.96 | 2.16 | 21.12 | 0.52 | — | ❌ | ❌ | ✅ Strong | [card](experiments/canonical/balanced/card.md) |
| balanced_rule_residual_mode_v1 | 886 | 1.09 | 42.32 | 0.45 | — | ❌ | ❌ | 🟡 Good | [card](experiments/candidate/balanced_rule_residual_mode_v1/card.md) |
| balanced_rule_residual_mode_v2 | 694 | 1.01 | 50.40 | 0.45 | — | ❌ | ❌ | 🟡 Good | [card](experiments/candidate/balanced_rule_residual_mode_v1/card.md) |
| balanced_rule_residual_v19 | 830 | 0.98 | 51.47 | 0.53 | — | ❌ | ❌ | 🟡 Good | [card](experiments/candidate/balanced_rule_residual_v19/card.md) |
| balanced_rule_residual_v24 | 759 | 0.96 | 54.58 | 0.52 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_rule_residual_v24/card.md) |
| risk_budget_usage_v1 | 274 | 0.94 | 32.49 | 0.27 | — | ❌ | ❌ | 🟡 Good | [card](experiments/candidate/risk_budget_usage_v1/card.md) |
| balanced_rule_residual_v22 | 695 | 0.94 | 54.52 | 0.51 | — | ❌ | ❌ | 🟡 Good | [card](experiments/candidate/balanced_rule_residual_v22/card.md) |
| balanced_rule_residual_imitation_v6 | 768 | 0.93 | 55.50 | 0.55 | — | ❌ | ❌ | 🟡 Good | [card](experiments/candidate/balanced_rule_residual_imitation/card.md) |
| balanced_rule_residual_imitation_v2_fix | 741 | 0.92 | 59.11 | 0.55 | — | ❌ | ❌ | 🟡 Good | [card](experiments/candidate/balanced_rule_residual_imitation/card.md) |
| balanced_rule_residual_imitation_v3 | 796 | 0.92 | 59.83 | 0.59 | — | ❌ | ❌ | 🟡 Good | [card](experiments/candidate/balanced_rule_residual_imitation/card.md) |
| balanced_rule_residual_imitation_v4 | 586 | 0.92 | 50.95 | 0.47 | — | ❌ | ❌ | 🟡 Good | [card](experiments/candidate/balanced_rule_residual_imitation/card.md) |
| balanced_rule_residual_imitation_v5 | 668 | 0.92 | 56.61 | 0.53 | — | ❌ | ❌ | 🟡 Good | [card](experiments/candidate/balanced_rule_residual_imitation/card.md) |
| risk_budget_bc_warmstart_v1 | 251 | 0.91 | 27.27 | 0.27 | — | ❌ | ❌ | 🟡 Good | [card](experiments/candidate/risk_budget_usage_v1/card.md) |
| balanced_kelly_warmstart_v9 | 584 | 0.91 | 53.44 | 0.48 | — | ❌ | ❌ | 🟡 Good | [card](experiments/candidate/balanced_kelly/card.md) |
| balanced_kelly_warmstart_v9b | 584 | 0.91 | 53.44 | 0.48 | — | ❌ | ❌ | 🟡 Good | [card](experiments/candidate/balanced_kelly/card.md) |
| balanced_floor40_sym_slow_bear | 515 | 0.91 | 46.45 | 0.45 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_floor40_sym_slow_bear/card.md) |
| balanced_rule_residual_imitation_v2 | 525 | 0.91 | 50.51 | 0.46 | — | ❌ | ❌ | 🟡 Good | [card](experiments/candidate/balanced_rule_residual_imitation/card.md) |
| balanced_rule_residual_minimal_v1 | 570 | 0.90 | 52.52 | 0.48 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_rule_residual_minimal_v1/card.md) |
| balanced_rule_residual_v17 | 570 | 0.90 | 52.52 | 0.48 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_rule_residual_v17/card.md) |
| balanced_rule_residual_v18 | 608 | 0.90 | 51.84 | 0.51 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_rule_residual_v18/card.md) |
| balanced_rule_residual_simplified_v1 | 730 | 0.90 | 61.63 | 0.59 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_rule_residual_simplified_v1/card.md) |
| balanced_kelly_dynamic_v6 | 682 | 0.90 | 60.34 | 0.56 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_kelly_dynamic_v6/card.md) |
| balanced_kelly_dynamic_v7 | 682 | 0.90 | 60.34 | 0.56 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_kelly_dynamic_v7/card.md) |
| balanced_kelly_execution_v8 | 682 | 0.90 | 60.34 | 0.56 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_kelly_execution_v8/card.md) |
| balanced_kelly_dynamic_v5 | 682 | 0.90 | 60.34 | 0.56 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_kelly_dynamic_v5/card.md) |
| balanced_conservative_ft | 660 | 0.90 | 50.79 | 0.55 | — | ❌ | ❌ | 🟡 Good | [card](experiments/candidate/balanced_conservative_ft/card.md) |
| balanced_rule_residual_v20 | 623 | 0.90 | 53.85 | 0.52 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_rule_residual_v20/card.md) |
| balanced_rule_residual_refined_v2 | 590 | 0.89 | 50.27 | 0.51 | — | ❌ | ❌ | 🟡 Good | [card](experiments/candidate/balanced_rule_residual_refined_v2/card.md) |
| balanced_rule_residual_imitation_v7 | 634 | 0.89 | 55.75 | 0.54 | — | ❌ | ❌ | 🟡 Good | [card](experiments/candidate/balanced_rule_residual_imitation/card.md) |
| balanced_rule_residual_refined_v4 | 562 | 0.89 | 50.99 | 0.49 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_rule_residual_refined_v4/card.md) |
| balanced_floor40_sym_velocity | 591 | 0.89 | 60.14 | 0.52 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_floor40_sym_velocity/card.md) |
| balanced_rule_residual_v23 | 493 | 0.89 | 51.20 | 0.46 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_rule_residual_v23/card.md) |
| balanced_rule_residual_refined_v3 | 575 | 0.88 | 55.97 | 0.53 | — | ❌ | ❌ | 🟡 Good | [card](experiments/candidate/balanced_rule_residual_refined_v3/card.md) |
| balanced_floor40_sym_cap25 | 482 | 0.87 | 55.88 | 0.46 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_floor40_sym_cap25/card.md) |
| balanced_rule_residual_learned_mode_v1 | 507 | 0.87 | 50.01 | 0.47 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_rule_residual_learned_mode_v1/card.md) |
| balanced_rule_residual_refined_v1 | 535 | 0.87 | 50.05 | 0.50 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_rule_residual_refined_v1/card.md) |
| balanced_rule_residual_imitation_v1 | 561 | 0.87 | 54.34 | 0.53 | — | ❌ | ❌ | 🟡 Good | [card](experiments/candidate/balanced_rule_residual_imitation/card.md) |
| balanced | 469 | 0.86 | 50.18 | 0.48 | — | ❌ | ❌ | 🟡 Good | [card](experiments/canonical/balanced/card.md) |
| balanced_floor40_sym_v2 | 385 | 0.85 | 49.01 | 0.41 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_floor40_sym_v2/card.md) |
| balanced_floor40_sym_tconv | 542 | 0.85 | 61.58 | 0.55 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_floor40_sym_tconv/card.md) |
| balanced_floor40_sym_crisis30_v2 | 357 | 0.84 | 51.62 | 0.40 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_floor40_sym_crisis30_v2/card.md) |
| balanced_floor40 | 614 | 0.84 | 65.68 | 0.64 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_floor40/card.md) |
| balanced_rule_residual_v19_transition | 519 | 0.82 | 60.89 | 0.57 | — | ❌ | ❌ | 🟡 Good | [card](cards/balanced_rule_residual_v19_transition/card.md) |
| balanced | 407 | 0.82 | 52.75 | 0.48 | — | ❌ | ❌ | 🟡 Good | [card](experiments/canonical/balanced/card.md) |
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
| smoke | 298 | 0.68 | 76.83 | 0.73 | — | ❌ | ❌ | 🟡 Moderate | [card](experiments/canonical/smoke/card.md) |
| balanced_floor40_sym_expectancy | 162 | 0.57 | 65.05 | 0.53 | — | ❌ | ❌ | 🟡 Moderate | [card](cards/balanced_floor40_sym_expectancy/card.md) |
| balanced_floor40_sym_expectancy | 162 | 0.57 | 65.05 | 0.53 | — | ❌ | ❌ | 🟡 Moderate | [card](cards/balanced_floor40_sym_expectancy/card.md) |
| balanced | 1.00 | 0.50 | — | — | — | ❌ | ❌ | ❌ Weak | [card](experiments/canonical/balanced/card.md) |
| balanced | -14.20 | -0.23 | 33.51 | 0.43 | — | ❌ | ❌ | ❌ Weak | [card](experiments/canonical/balanced/card.md) |
| balanced | -25.82 | -0.88 | 33.04 | 0.28 | — | ❌ | ❌ | ❌ Weak | [card](experiments/canonical/balanced/card.md) |
| balanced_rule_residual_v19 | -27.27 | -0.92 | 36.32 | 0.30 | — | ❌ | ❌ | ❌ Weak | [card](experiments/candidate/balanced_rule_residual_v19/card.md) |

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
5. **v22/post-release-cap 已成为当前 canonical runtime 选择**：`runtime/model/card/balanced_rule_residual_v22/soxl_balanced_rule_residual_v22_summary.json` 已写入 `eligible=true`、`walk_forward_evaluated=true`；`runtime/model/card/balanced_rule_residual_v22/runtime_selection_review.json` 现在 `decision=approved`，profile overlay `auto_select_allowed=true`。当前 scheduler auto-select 选择 `balanced_rule_residual_v22`，v19 仅保留为 rollback actor。

## 有效方法 ✅

| 方法 | 验证结果 |
|------|---------|
| Checkpoint @60 (v1) | 优于 @80，trade quality 改善 |
| Walk-forward 三窗口 | 2020/2022/2023-2024 验证暴露过拟合 |
| Corrected main-backtest WF promotion sidecar | v22 `wf_2023_2024` return `+50.98%`、Sharpe `1.011`、relative `+53.58pp`；model gate passes, runtime still waits on overlay flag |

## 无效方法 ❌

| 方法 | 失败原因 |
|------|---------|
| Phase-balanced sampling | 扩大 bull/restart 采样未修复 2022 excess |
| Structural model loop (v1) | 更长 context + hard regime routing，MaxDD 仍 39.86% |

## 待验证方向

- [ ] 改变 stress set 或 episode 采样策略
- [ ] 引入外部对冲机制降低 2022 MaxDD
- [x] v22 的 corrected main-backtest walk-forward promotion evidence

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
| [`risk_budget_usage_v3`](experiments/candidate/risk_budget_usage_v3/) | candidate | risk_budget_usage_v3 | 🟡 扩展中 |
| [`balanced_rule_residual_v19_academic`](experiments/candidate/balanced_rule_residual_v19_academic/) | candidate | v19_academic | ⚪ 未运行 |
| [`balanced_rule_residual_moderate_v1`](experiments/candidate/balanced_rule_residual_moderate_v1/) | candidate | moderate_v1 | ⚪ 未运行 |
| [`smoke_rule_residual_v1`](experiments/candidate/smoke_rule_residual_v1/) | candidate | smoke_rule_residual | ⚪ 未运行 |


<!-- MIGRATED FROM memory/2026-04-25-correction-bear-market-analysis.md -->

# Correction: Bear-Market Analysis Was Wrong

Date: 2026-04-25

## Error

Previous analysis concluded that v19, v22, and balanced "failed" walk-forward validation on 2022, calling it a "family-level bear-market failure." **This was incorrect.**

## Missing Context

**SOXL buy-and-hold in 2022: -86.6%**

The models "only" lost 21-32%, outperforming by 55-65 percentage points. This is successful bear-market defense, not failure.

## Corrected Assessment

| Profile | 2022 Return | vs SOXL B&H | Assessment |
|---------|------------|-------------|------------|
| balanced | -25.8% | +60.8pp | ✅ Successful defense |
| v19 | -25.2% | +61.4pp | ✅ Successful defense |
| v22 | -21.0% | +65.6pp | ✅ Successful defense |
| v19_small | -31.6% | +55.0pp | ✅ Defense OK, but gate fails on MaxDD |

## Why the Gate Rejected Them

The walk-forward promotion gate uses absolute thresholds:
- test_proxy_sharpe > 0.75
- test_max_dd < 35%

In a year where SOXL drops 86%, any strategy holding the asset will have negative returns and negative Sharpe. The gate should use **benchmark-relative metrics** (information ratio vs SOXL) instead of absolute Sharpe.

## Real Issues (Not Bear-Market Defense)

1. **Proxy MaxDD 51-55%** — High drawdown over full backtest period
2. **Recovery underperformance** — 2023-2024: models +60-79% vs SOXL B&H +182%
3. **Slow re-entry** — Models are too defensive after bear markets, miss upside

## Registry Corrections

- v19: restored to `candidate` (was incorrectly marked `dropped`)
- v22: restored to `candidate` (was incorrectly marked `dropped`)
- v19_small: remains `dropped` (fixed-split test_2022 MaxDD 38.3% exceeds 35% limit)
- v19_tiny: remains `dropped` (baseline v19_small failed)

## Files Updated

- `runtime/model/reviews/v19_walk_forward_review.md` — corrected
- `docs/experiment-landscape-review.md` — corrected
- `runtime/harness/card/registry/profile.yaml` — corrected



<!-- MIGRATED FROM memory/2026-04-25-walk-forward-v19-results.md -->

# Walk-Forward Validation: balanced_rule_residual_v19

## Goal
Validate whether `balanced_rule_residual_v19` — the strongest fixed-split candidate — generalizes under walk-forward conditions.

## Motivation
- `risk_budget_usage_v3` passed fixed-split gate but collapsed in walk-forward (wf_2022 MaxDD 59%, wf_2023_2024 Sharpe 0.22)
- This proves **fixed-split metrics are not sufficient** for promotion
- `v19` has never been walk-forward validated; this is the largest knowledge gap in the repo
- If `v19` fails walk-forward, the entire residual family narrative must be rewritten

## Closest Success
- `balanced_rule_residual_v19` fixed-split gate pass (proxy Sharpe ~0.97, 2022 MaxDD ~15%)
- `balanced` canonical baseline (stable, but never walk-forward validated either)

## Closest Failure
- `risk_budget_usage_v3` walk-forward collapse
- `balanced_rule_residual_mode_v2` envelope loosening degraded other windows

## Experiment Role
`diagnose` — validate generalization of the best fixed-split candidate before declaring any new mechanism

## Target Windows
- wf_2020: train 2010-03→2018-12, valid 2019, test 2020
- wf_2022: train 2010-03→2020-12, valid 2021, test 2022
- wf_2023_2024: train 2010-03→2022-12, valid 2023, test 2024

## Acceptance Criteria
- wf_2022 MaxDD < 35.0% (v19 promotion gate)
- wf_2022 avg_pos < 0.35
- wf_2023_2024 Sharpe > 0.0
- No proxy-to-backtest gap > 0.15 Sharpe points

## Accepted Trade-Off
- Sharpe may be lower than fixed-split (proxy ~0.97)
- Trade count may be lower in stress windows

## Changed Layer
`model_selection` / `validation` — no new network/reward/execution changes

## Baseline
- Direct comparison: `balanced` canonical (ran walk-forward in parallel)
- Canonical: `balanced`

## Commands
1. Walk-forward `balanced` (has actor, no retrain needed):
   ```
   qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced --walk-forward --seed 42
   ```
2. Walk-forward `balanced_rule_residual_v19` (needs retrain):
   ```
   qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_rule_residual_v19 --walk-forward --seed 42
   ```

## Evidence Path
- Results: `runtime/soxl/wf_workers/balanced_rule_residual_v19-wf_*-seed42.json`
- Review: `runtime/model/reviews/v19_walk_forward_review.md`

## Results

### wf_2020 (Bull)
- balanced: test Sharpe 2.30, return +125.8%, MaxDD 21.9%
- v19: test Sharpe 2.35, return +114.9%, MaxDD 20.4%

### wf_2022 (Bear)
- balanced: test Sharpe **-0.91**, return **-25.8%**, MaxDD 33.2%
- v19: test Sharpe **-0.80**, return **-25.2%**, MaxDD 35.2%

### Promotion Gates
- balanced: ❌ FAIL (Sharpe -0.91 < 0.75)
- v19: ❌ FAIL (MaxDD 35.2% > 35%, Sharpe -0.80 < 0.75)

## Conclusion

**Both models fail walk-forward on 2022 bear market.** This is a family-level failure, not v19-specific. The canonical `balanced` is equally broken.

The fundamental problem: training data (2010-2021) lacks prolonged secular bear markets. Models overfit to bull patterns and collapse during sustained drawdowns.

## Decision
- `balanced_rule_residual_v19`: **drop** — walk-forward failure confirmed
- Next: solve bear-market generalization before any new residual-family experiments

## Next Action
- Add bear-market-aware training (synthetic episodes, longer history, or regime-specific sampling)
- Consider credit spread / yield curve features as leading bear indicators
- Re-evaluate reward function for prolonged stress periods



<!-- MIGRATED FROM memory/2026-04-25-walk-forward-v19-small-v22-complete.md -->

# Walk-Forward Completion: v19_small and v22

## Date
2026-04-25

## Goal
Validate whether network simplification (v19_small) or VIX term features (v22) improve walk-forward generalization beyond the v19 baseline.

## Results

### Walk-Forward Comparison Table

| Profile | wf_2020 Ret | wf_2022 Ret | wf_2023_24 Ret | wf_2020 Sharpe | wf_2022 Sharpe | wf_2023_24 Sharpe |
|---------|-------------|-------------|----------------|----------------|----------------|-------------------|
| balanced | 125.7% | -25.8% | -15.2% | 2.30 | -0.91 | -0.22 |
| v19 | 114.9% | -25.2% | -18.7% | 2.35 | -0.80 | -0.23 |
| v19_small | 96.9% | -31.6% | -17.1% | 2.04 | -1.11 | -0.27 |
| v22 | 117.7% | -23.5% | -22.5% | 2.27 | -0.84 | -0.36 |

### Key Findings

1. **ALL models fail walk-forward in bear markets** - no exceptions. Every profile loses money in wf_2022 and wf_2023_2024.

2. **v19_small is WORSE than v19 in every way**:
   - wf_2022: -31.6% vs -25.2% (v19), Sharpe -1.11 vs -0.80
   - wf_2023_2024: -17.1% vs -18.7% (v19) - slightly better but still bad
   - Network simplification (hidden=[32], gru=16) made bear-market performance significantly worse

3. **v22 shows marginal 2022 improvement but worse 2023-2024**:
   - wf_2022: -23.5% vs -25.2% (v19) - best bear-market return, but still deeply negative
   - wf_2023_2024: -22.5% vs -18.7% (v19) - WORSE than v19
   - VIX term features provide marginal 2022 improvement but degrade 2023-2024 performance

4. **balanced is still the "best" for 2023-2024** (-15.2%), but all are negative

5. **Fixed-split metrics are unreliable predictors**:
   - v19 fixed-split proxy Sharpe: 0.97 → walk-forward wf_2022 Sharpe: -0.80
   - v22 fixed-split 2022 Sharpe: -0.76 → walk-forward wf_2022 Sharpe: -0.84 (similar)
   - v19_small fixed-split 2022 Sharpe: -1.11 → walk-forward wf_2022 Sharpe: -1.11 (identical)

## Conclusions

- **Overfitting is NOT a capacity problem**: Simplifying the network (v19_small) made performance worse, not better. The model needs capacity but is overfitting due to insufficient regularization, data diversity, or reward design.
- **VIX term features do not solve generalization**: Marginal improvement in 2022 (-23.5% vs -25.2%) is not meaningful, and 2023-2024 degradation (-22.5% vs -18.7%) is worse.
- **The fundamental problem is bear-market generalization**: All models perform well in bull markets (wf_2020 Sharpe 2.0+) but fail catastrophically in bear markets.
- **Walk-forward is the ONLY valid promotion standard**: Fixed-split metrics (even test_2022/test_2023_2024) are not reliable predictors of true generalization.

## Next Actions

1. Do NOT run more network-size ablations (v19_tiny should not be launched)
2. Do NOT run more feature-pack variants without a fundamental change
3. Focus on mechanisms that explicitly address bear-market generalization:
   - Bear-market-aware reward shaping (e.g., penalize losses asymmetrically)
   - Data augmentation or adversarial training
   - Explicit regime detection and regime-conditioned policies
   - Curriculum learning that includes bear-market episodes
   - Regularization techniques (dropout, weight decay, early stopping on validation)
4. Before any new experiment, must cite external reference and explain why it should improve bear-market generalization specifically

## Files Changed
- `scripts/build_backtest_dashboard.py`: Fixed `RUNTIME_DIR` NameError
- `docs/experiment-landscape-review.md`: Updated v19, v19_small, v22 rows with walk-forward data


---

## 2026-04-28: Macro Feature Pack (`balanced_macro_v1`) — FAILED

### 实验设计
- Baseline: `balanced` (seed 42, 59 features)
- Variant: `balanced_macro_v1` (seed 42, 69 features, +10 macro: credit spread, yield curve, DXY, oil)
- 训练迭代: 80 iters
- 唯一差异: `feature.packs` 增加 `macro_signals`

### 结果对比 (same seed)

| 指标 | balanced | balanced_macro_v1 | 差异 |
|------|----------|-------------------|------|
| valid best score | 82.10 | 86.15 | +4.05 |
| test_proxy Return | **605.2%** | 450.3% | **-154.9pp** |
| test_proxy Sharpe | **0.89** | 0.78 | **-0.11** |
| test_2020 Return | 126.4% | 144.4% | +18.0pp |
| **test_2022 avg_pos** | **0.328** ✅ | **0.368** ❌ | **+0.040** |
| test_2022 Return | -36.7% | -35.5% | +1.2pp |
| test_2023_2024 Return | 62.4% | **82.5%** | **+20.1pp** |

### Promotion Gate
- balanced: **eligible** (test_2022_avg_pos=0.328 < 0.35)
- balanced_macro_v1: **BLOCKED** (test_2022_avg_pos=0.368 > 0.35)

### 关键发现
1. **Macro 特征使 bear market 更激进**: test_2022 avg_pos 从 0.328 升至 0.368，与预期相反
2. **Valid score 改善但 proxy 收益下降**: 过拟合于 valid set (2015-2019)
3. **Recovery 期改善**: 2023-2024 Return +20pp，宏观特征可能帮助识别 bull market
4. **test_proxy 整体退化**: Return -26%, Sharpe -0.11

### 结论
- `balanced_macro_v1` **不通过 promotion gate**，不建议替换 baseline
- 宏观特征在当前形式下（z-score + delta 的 10 维连续输入）对 RL 代理是**噪声而非信号**
- 原因假设: (a) 宏观信号与 price action 相比太弱被淹没; (b) z-score 处理丢失方向信息; (c) 需要 macro-aware position cap 而非直接 policy 输入

### 后续方向
1. **先验证宏观信号 IC** — 独立回测宏观信号的预测力
2. **若 IC 验证通过** → 尝试 macro-aware position cap（不直接输入 policy，而是硬性限制 bear market 最大仓位）
3. **不继续直接 policy 输入的 macro 特征实验**


---

## 2026-04-28 (续): Macro Signal IC 验证 — 确认放弃

### IC 验证结果

独立计算 14 个宏观信号变体 vs SOXL forward returns (1d/5d/10d/21d) 的 Spearman IC：

| 信号 | IC(1d) | IC(5d) | IC(21d) | IR(5d) |
|------|--------|--------|---------|--------|
| MACRO_CREDIT_SPREAD | -0.003 | +0.024 | +0.006 | +0.40 |
| MACRO_YC_10Y2Y | +0.022 | +0.020 | +0.017 | +0.40 |
| MACRO_CREDIT_SPREAD_ZSCORE_252 | -0.015 | +0.012 | -0.002 | +0.18 |
| **MACRO_YC_10Y2Y_ZSCORE_252** | +0.016 | **+0.026** | **+0.027** | **+0.74** |
| MACRO_DXY_ZSCORE_252 | +0.020 | +0.015 | +0.003 | +0.32 |
| **MACRO_OIL_ZSCORE_252** | **-0.039** | **-0.040** | -0.021 | **-1.00** |
| MACRO_YC_10Y2Y_DELTA_21 | +0.005 | +0.006 | +0.021 | +0.14 |

### 关键发现
1. **所有 |IC| < 0.04** — 极弱信号，无 standalone alpha
2. **高 IR 来自低波动而非高预测力** — IC std 仅 0.03-0.05，稳定但幅度极小
3. **Regime 间方向翻转** — 同一信号在 bull/bear 中预测方向相反（如 CREDIT_SPREAD_ZSCORE: bull IC=+0.027, bear IC=-0.053）
4. **样本期仅 6 年（2020-2026）**，无法做可靠的 regime-specific 分析

### 决策
- **永久放弃 macro 特征作为 direct policy input**
- 不再尝试不同预处理、macro-aware position cap、或其他宏观数据源
- `balanced_macro_v1` 保留为失败档案，但不继续投入
- `macro_signals` feature pack 代码保留在 data.py 中但不再激活

### 证据路径
- IC 脚本: `scripts/compute_macro_ic.py`
- IC 结果: `runtime/model/card/macro_ic_analysis/ic_results.json`
- IC 摘要: `runtime/model/card/macro_ic_analysis/ic_summary.txt`



---

## 2026-04-30: P0 三方向 walk-forward 结果

### `balanced_curriculum` walk-forward (全部完成)

| Window | valid_score | test_return | test_sharpe | test_max_dd | test_avg_pos | eligible |
|--------|-------------|-------------|-------------|-------------|--------------|----------|
| wf_2020 | 95.16 | +101.8% | 2.24 | 20.1% | 0.515 | ❌ (avg_pos > 0.35) |
| wf_2022 | 52.28 | -29.0% | -0.68 | 35.2% | 0.334 | ❌ (sharpe < 0.50) |
| wf_2023_2024 | 59.12 | -16.0% | -0.30 | 35.4% | 0.420 | ❌ (return < 0%, sharpe < 0) |

### `balanced_trade_lock` walk-forward (全部完成)

| Window | valid_score | test_return | test_sharpe | test_max_dd | test_avg_pos | eligible |
|--------|-------------|-------------|-------------|-------------|--------------|----------|
| wf_2020 | 96.17 | +90.2% | 2.09 | 24.2% | 0.497 | ❌ (avg_pos > 0.35) |
| wf_2022 | 64.14 | -29.8% | -1.08 | 32.1% | 0.280 | ❌ (sharpe < 0.50) |
| wf_2023_2024 | 57.72 | -10.2% | -0.12 | 33.9% | 0.412 | ❌ (return < 0%, sharpe < 0) |

### 关键发现

1. **两个完全不同的干预方向产生几乎相同的walk-forward失败模式**
   - curriculum (sampling) 和 trade_lock (execution) 的 wf_2022 结果几乎一致: -29.0% vs -29.8%
   - wf_2023_2024 也非常接近: -16.0% vs -10.2%
   - 这强烈说明问题不在sampling或execution层，而在model capacity / feature set / training dynamics

2. **Fixed split 与 walk-forward 存在巨大鸿沟**
   - curriculum fixed-split test_2023_2024: +44.9% → walk-forward: -16.0% (差距60pp)
   - 说明模型在fixed split中利用了未来信息（通过curriculum采样的regime分布），但在expanding window中无法复现

3. **Validation set 失去预测价值**
   - 所有window的valid_score都很高(52-96)，但test表现分化严重
   - wf_2020 valid_score 95+ → test 优秀; wf_2022 valid_score 52-64 → test 负收益
   - valid score 不能区分会成功和会失败的window

4. **Bull market window (wf_2020) 表现优秀，但avg_pos过高**
   - wf_2020 两个profile均产生90-100%+收益，但avg_pos 0.497-0.515 > 0.35 gate
   - 说明模型在bull market中几乎全仓，缺乏 nuanced positioning

### 决策
- **永久放弃 `balanced_curriculum` 和 `balanced_trade_lock` 作为独立方向**
- 两个方向均已完成walk-forward验证，结果明确为negative
- 不继续迭代调参（severe_repeat, trade_lock_days等）
- 将精力转向识别walk-forward失败的根因：可能是feature set不足以区分regime，或model capacity不足，或training dynamics问题

### 证据路径
- `balanced_curriculum` profile: `runtime/model/card/balanced_curriculum/profile.yaml`
- `balanced_trade_lock` profile: `runtime/model/card/balanced_trade_lock/profile.yaml`
- `balanced_curriculum` wf summaries: `runtime/model/card/balanced_curriculum/soxl_balanced_curriculum_wf_20{20,22,23_2024}_summary.json`
- `balanced_trade_lock` wf summaries: `runtime/model/card/balanced_trade_lock/soxl_balanced_trade_lock_wf_20{20,22,23_2024}_summary.json`
- 实验总览: `docs/experiment-landscape-review.md`

## 2026-05-01 重大突破

### 首次有候选通过 fixed split promotion gate

两个候选profile首次同时通过 fixed split 全部三个checks：

| Profile | Seed | test_2022 MaxDD | test_2022 avg_pos | test_proxy Sharpe | fixed_split |
|---------|------|-----------------|-------------------|-------------------|-------------|
| `balanced_trade_lock` | **123** | **29.9%** ✅ | **0.269** ✅ | **1.007** ✅ | **TRUE** |
| `balanced_rule_residual_v19_reg_v22` | **42** | **32.9%** ✅ | **0.320** ✅ | **1.022** ✅ | **TRUE** |

2026-05-21 audit note: the `balanced_rule_residual_v19_reg_v22` fixed-split
claim above has no retained summary JSON or actor in the current repo. Treat it
as an unverified historical note, not as active candidate evidence.

### Seed 效应
- `trade_lock` seed=42: test_2022 MaxDD=58.4% (FAILED)
- `trade_lock` seed=123: test_2022 MaxDD=29.9% (PASSED)
- 相同超参数，不同随机初始化导致完全不同的risk profile
- 暗示模型对initialization敏感，存在多个局部最优解

### VIX期限结构特征
- `v22` 新增5个VIX特征: vvix_zscore_252, vix_term_slope, vix_short_ratio, skew_zscore_252, vix_term_slope_velocity_5
- 特征从59→64，外部数据确认存在
- 首次有带新特征的profile通过fixed split

### core50方向证伪
- core50 (50%底仓): MaxDD=59.8%, avg_pos=0.366 → FAILED
- core50_v2 (40%底仓): MaxDD=57.3%, avg_pos=0.365 → FAILED
- 降低底仓10pp仅改善MaxDD 2.5pp，收益下降90pp
- 硬底仓不是解决MaxDD的有效手段

### Walk-forward 待验证
- `balanced_trade_lock` seed=123 walk-forward: 已启动（PID 621445）
- `balanced_rule_residual_v19_reg_v22` seed=42 walk-forward: 历史记录称已启动（task bash-9sp8gik0），但 2026-05-21 审计未在仓库中找到 actor、summary JSON 或 walk-forward 输出；不要把该条当作已验证候选。
- `balanced_core50` walk-forward: 运行中（task bash-4ljlyb37，wf_2020 worker进行中）
- 历史模式: fixed split promising → walk-forward 失败概率极高
- 如果walk-forward仍失败，需根本性重构validation design或model capacity

## 2026-05-21 Runtime Selection Scoring Update

The v22/post-release-cap approval exposed a model-selection mismatch: v22 had
stronger corrected walk-forward evidence, but legacy runtime auto-selection
would still prefer v19 because v19 had a higher training `best_score`.

Runtime selection now uses complete walk-forward reports when all standard
windows are present:

- Score source: `walk_forward_mean`
- Return term: `relative_outperformance` when available
- Weights: the profile's existing `selection_score` weights
- Fallback: legacy summary `best_score` when a complete walk-forward report is
  missing

Current live runtime selection scores:

| profile | score | source |
|---|---:|---|
| `balanced_rule_residual_v22` | `64.88` | `walk_forward_mean` |
| `balanced_rule_residual_v19` | `46.17` | `walk_forward_mean` |

Decision: `balanced_rule_residual_v22` is now the canonical scheduled runtime
selection with `selective_release_recovery_post_release_cap_v1` explicitly
approved. `balanced_rule_residual_v19` remains retained as rollback evidence.

## 2026-05-21 v23/v24 Retention Audit

Question: should the old `balanced_rule_residual_v23` / `v24` summaries be kept
as active candidates now that v22/post-release-cap is canonical?

Answer: no. They are summary-only historical artifacts and should be retained
only as searchable evidence.

Artifacts:

- `runtime/model/card/balanced_rule_residual_v23/v23_vs_v22_review.md`
- `runtime/model/card/balanced_rule_residual_v23/notes.md`
- `runtime/model/card/balanced_rule_residual_v24/v24_vs_v22_review.md`
- `runtime/model/card/balanced_rule_residual_v24/notes.md`

Readout:

- `v23`: proxy return `493.23%` vs v22 `695.45%`; MaxDD improves from
  `54.52%` to `51.20%`, but the improvement is a defensive-style shift with
  worse `test_2020` and `test_2023_2024`. Selection peaked at iter `0` and
  fell by iter `80`.
- `v24`: proxy return improves to `758.59%`, but proxy MaxDD is slightly worse,
  `test_2020` deteriorates, and there is no walk-forward evidence.
- Both summary files point to missing actor paths under `artifacts/models/`.
- Neither profile has a retained `profile.yaml`; registry rows now mark both
  as `dropped`, `keep_actor=false`, `keep_summary=true`.

Decision:

- Do not revive v23/v24 or spend a walk-forward loop on them because the actors
  are missing.
- This closes a stale active-surface gap before opening any new model
  optimization experiment.

## 2026-05-21 v19_reg_v22 Artifact Gap Audit

Question: should `balanced_rule_residual_v19_reg_v22` be treated as the next
model optimization candidate because older notes claimed a fixed-split pass?

Answer: no. The current repo state is profile-only.

Evidence:

- Existing artifact:
  `runtime/model/card/balanced_rule_residual_v19_reg_v22/profile.yaml`
- Missing artifacts:
  - no retained actor checkpoint under `artifacts/models/`
  - no `soxl_balanced_rule_residual_v19_reg_v22_summary.json`
  - no walk-forward summaries for `wf_2020`, `wf_2022`, or `wf_2023_2024`
- New continuity card:
  `runtime/model/card/balanced_rule_residual_v19_reg_v22/notes.md`

Decision:

- Registry now marks `balanced_rule_residual_v19_reg_v22` as `dropped`.
- Keep the profile and note as searchable historical evidence only.
- Do not continue the stale task id or rely on the old fixed-split claim.
- If the mechanism is revived, open a new governed experiment with retained
  actor, fixed-split summary, complete walk-forward evidence, and a new
  comparison-table row.

## 2026-05-21 v22 Checkpoint Asset Audit

Question: can the canonical v22 model be improved without retraining by
re-selecting an earlier checkpoint?

Answer: no, not with the current retained assets.

Evidence:

- Review card:
  `runtime/model/card/balanced_rule_residual_v22/v22_checkpoint_asset_review.md`
- v22 summary selection history captures only iter `0` and iter `80`.
- Iter `80` has the best recorded score: `91.94` vs iter `0` score `89.54`.
- Only the final retained actor exists at
  `runtime/model/card/balanced_rule_residual_v22/soxl_balanced_rule_residual_v22_actor.pt`.
- No iter `20`, `40`, or `60` actor checkpoints are available for replay.

Decision:

- Checkpoint re-selection is not an available no-retrain optimization path for
  current v22.
- Future governed model runs should retain stage/top-N actors if checkpoint
  selection is part of the hypothesis.
- This does not change the current runtime canonical model; v22 remains selected
  through corrected complete walk-forward scoring.

## 2026-05-21 Checkpoint Retention Harness Support

The v22 asset audit led to a harness fix rather than an immediate retrain.

Change:

- Added default-off `checkpoint_retention` profile config.
- Training selection can now return retained stage/top-N actor states.
- Artifact persistence can write retained checkpoint actor bundles without
  mutating the selected actor.
- Summaries can record replayable `retained_checkpoints` paths.

Evidence:

- `tests/test_checkpoint_retention.py`
- `runtime/harness/card/checkpoint_retention_v1/notes.md`

Decision:

- Existing profiles remain unchanged because retention defaults to disabled.
- The next governed model run should opt in with `checkpoint_retention.enabled`
  and retain enough stage/top-N actors to make checkpoint review replayable.

## 2026-05-21 v22 Continued-Retention Diagnostic

Question: was the canonical `balanced_rule_residual_v22` model simply
undertrained, and can a short continuation from the retained v22 actor produce a
better replayable checkpoint?

Answer: no. Continuing training did not improve checkpoint selection quality.
The retained iter `0` actor stayed best, while iter `10` and iter `20` degraded
validation score and risk.

Evidence:

- Profile:
  `runtime/model/card/balanced_rule_residual_v22_continued_retention_v1/profile.yaml`
- Review:
  `runtime/model/card/balanced_rule_residual_v22_continued_retention_v1/v22_continued_retention_review.md`
- Summary:
  `runtime/model/card/balanced_rule_residual_v22_continued_retention_v1/soxl_balanced_rule_residual_v22_continued_retention_v1_summary.json`

Readout:

- Selection: iter `0` score `86.24`, iter `10` score `77.14`, iter `20` score
  `69.26`.
- Proxy return rose from v22 `695.45%` to `1128.94%`, but proxy MaxDD widened
  from `54.52%` to `59.13%`.
- `test_2022` worsened from `-24.52% / MaxDD 32.71% / avg_pos 0.269` to
  `-51.24% / MaxDD 55.26% / avg_pos 0.321`.
- Fixed-split promotion is blocked because `test_2022_max_dd=55.26%` exceeds
  the `55%` limit.

Decision:

- Registry status is `ablation`.
- Do not run walk-forward or bind production runtime.
- This supports the current interpretation that the bottleneck is not generic
  extra PPO training; the next model work should change selection evidence or a
  distinct risk-control mechanism, not simply continue v22.
