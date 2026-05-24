# Recovery Reentry Anchor Confirm v1

Date: 2026-05-24

## Governance

- Baseline: `v19reg_v22_budget_usage_bull_guard_v1`
- Direct comparison: `recovery_reentry_floor_v1`
- Role: `diagnose`
- Changed layer: `execution`
- Target window: strict OOS 2025-2026 recovery poor windows
- Walk-forward checks: `wf_2020`, `wf_2022`, `wf_2023_2024`
- Accepted trade-off: preserve the 2026 reentry target change while removing
  `wf_2022` false positives
- Failure mode: long-anchor confirmation removes too much recovery capture or
  still leaves drawdown above target
- Comparison-table row key: `recovery_reentry_anchor_confirm_v1`

## External Basis

- [Time Series Momentum](https://www.aqr.com/insights/research/journal-article/time-series-momentum):
  trend persistence motivates recovery participation, but the trend state must
  be confirmed rather than inferred from a short bounce alone.
- [Risk Premia Harvesting Through Dual Momentum](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2042750):
  absolute trend/momentum filters are used to reduce volatility and drawdown.
- [Enhancing Time Series Momentum Strategies Using Deep Neural Networks](https://arxiv.org/abs/1904.04912):
  separates trend estimation and position sizing, matching this fixed-actor
  sizing counterfactual.
- [Qlib GitHub](https://github.com/microsoft/qlib): keep the loop in the qlib
  research-to-production harness.

## Root-Cause Readout

`recovery_reentry_floor_v1` created 2022 false positives. Comparing gate days
showed the best separation is long-anchor confirmation:

| feature | gate_2022 mean | gate_2026 mean | readout |
| --- | ---: | ---: | --- |
| `below_anchor_ratio_63` | 0.9095 | 0.0152 | 2022 bounces were still below anchor |
| `anchor_gap_180` | -0.3654 | 1.4738 | 2026 had recovered above long anchor |
| `days_since_high_63` | 0.7333 | 0.0425 | 2026 was much closer to new highs |
| `mom_21` | 0.4341 | 0.6895 | short momentum alone cannot separate enough |

The anchor-confirmed guard therefore adds:

```text
max_below_anchor_ratio_63 = 0.20
min_anchor_gap_180 = 0.75
```

## Command

```bash
PYTHONPATH=src:. qlib_venv/bin/python scripts/compose_budget_usage_counterfactual.py \
  --slow runtime/model/card/balanced_rule_residual_v19_reg/balanced_rule_residual_v19_reg_soxl_daily.csv \
  --fast runtime/model/card/balanced_rule_residual_v22/balanced_rule_residual_v22_soxl_daily.csv \
  --output-dir runtime/execution/card/recovery_reentry_anchor_confirm_v1 \
  --profile recovery_reentry_anchor_confirm_v1 \
  --slow-label balanced_rule_residual_v19_reg_current \
  --fast-label balanced_rule_residual_v22_current \
  --composed-label recovery_reentry_anchor_confirm_v1 \
  --description "Fixed-actor recovery/reentry floor with long-anchor confirmation after v1 2022 false positives." \
  --bull-participation-guard \
  --recovery-reentry-guard \
  --reentry-min-target 0.35 \
  --reentry-max-target 0.55 \
  --reentry-max-base-target 0.25 \
  --reentry-min-mom-10 0.08 \
  --reentry-min-mom-21 0.12 \
  --reentry-min-relative-strength-21 1.0 \
  --reentry-min-vol-of-vol-21 0.10 \
  --reentry-max-vix-percentile-252 0.85 \
  --reentry-max-below-anchor-ratio-63 0.20 \
  --reentry-min-anchor-gap-180 0.75
```

## Results

| window | baseline | floor_v1 | anchor_confirm_v1 |
| --- | ---: | ---: | ---: |
| strict OOS return | 1542.89% | 2091.90% | 1894.61% |
| strict OOS Sharpe | 1.109 | 1.194 | 1.165 |
| strict OOS MaxDD | 41.59% | 41.59% | 41.59% |
| strict OOS avg_pos | 0.462 | 0.468 | 0.467 |
| strict OOS gate_days | n/a | 35 | 22 |
| strict OOS changed_days | n/a | 73 | 21 |
| `wf_2020` return | 98.27% | 108.00% | 113.89% |
| `wf_2020` MaxDD | 26.14% | 26.14% | 26.14% |
| `wf_2022` return | -20.68% | -22.79% | -20.68% |
| `wf_2022` avg_pos | 0.184 | 0.212 | 0.184 |
| `wf_2023_2024` return | 58.19% | 58.19% | 58.19% |

Worst 2026 poor-window read-through:

| window | baseline strategy | floor_v1 strategy | anchor_confirm strategy | buy-hold |
| --- | ---: | ---: | ---: | ---: |
| 2026-02-10..2026-05-11 | 1.74% | 33.78% | 25.60% | 200.63% |
| 2026-02-05..2026-05-06 | 17.69% | 46.84% | 37.86% | 211.46% |

## Decision

Keep `recovery_reentry_anchor_confirm_v1` as the better diagnostic branch, but
do not promote to runtime.

It preserves the execution-addressable 2026 recovery improvement while removing
the `wf_2022` false positives from `recovery_reentry_floor_v1`. It also improves
`wf_2020` more than floor v1. The remaining blocker is risk: strict OOS MaxDD is
still `41.59%`, far above the `15%` target, and the 2026 poor-window excess is
still materially negative versus buy-and-hold.

Next valid loop: pair anchor-confirmed reentry with post-reentry de-risk or
profit-lock logic, then require unchanged `wf_2022` and lower strict-OOS / recent
drawdown before considering runtime promotion.

## Artifacts

```text
runtime/execution/card/recovery_reentry_anchor_confirm_v1/recovery_reentry_anchor_confirm_v1.json
runtime/execution/card/recovery_reentry_anchor_confirm_v1/recovery_reentry_anchor_confirm_v1.md
runtime/execution/card/recovery_reentry_anchor_confirm_v1/recovery_reentry_anchor_confirm_v1_soxl_daily.csv
runtime/execution/card/recovery_reentry_anchor_confirm_v1/oos_poor_window_diagnostics/
runtime/execution/card/recovery_reentry_anchor_confirm_v1/walkforward/
```
