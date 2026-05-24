# OOS Poor-Window Feature Diagnostics v1

Date: 2026-05-24

## Goal

Answer whether the current Futu paper model has post-training OOS risk beyond the
headline full-period return, and create a reusable harness script to quickly
screen bad intervals and feature differences before proposing another training
branch.

## External Basis

- [Deflated Sharpe Ratio](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551):
  repeated trials need selection-bias discipline.
- [Probability of Backtest Overfitting](https://doi.org/10.21314/JCF.2016.322):
  OOS and walk-forward checks should drive model-selection decisions.
- [Qlib GitHub](https://github.com/microsoft/qlib) and
  [Qlib arXiv](https://arxiv.org/abs/2009.11189): keep the diagnostic inside
  the qlib research-to-production workflow.

## Inputs

```text
profile: v19reg_v22_budget_usage_bull_guard_v1_current
paper daily replay: runtime/production/card/paper_model_oos_risk_check_v1/paper_model_oos_risk_check_v1_soxl_daily.csv
train: 2010-03-11 .. 2018-12-31
valid: 2019-01-01 .. 2019-12-31
strict OOS: 2020-01-02 .. 2026-05-21
targets: excess >= 30%, max_dd <= 15%
```

## Command

```bash
PYTHONPATH=src:. qlib_venv/bin/python scripts/diagnose_oos_poor_windows.py \
  --daily-csv runtime/production/card/paper_model_oos_risk_check_v1/paper_model_oos_risk_check_v1_soxl_daily.csv \
  --profile v19reg_v22_budget_usage_bull_guard_v1_current \
  --output-dir runtime/production/card/oos_poor_window_feature_diagnostics_v1 \
  --oos-start 2020-01-02 \
  --oos-end 2026-05-21 \
  --window-size 63 \
  --top-windows 12 \
  --top-features 30 \
  --target-excess 30 \
  --target-max-dd 15
```

## Key Results

The worst 63-trading-day OOS windows are concentrated in the 2026 rebound:

| window | strategy | buy-hold | excess | max_dd | avg_pos |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026-02-10 .. 2026-05-11 | 1.74% | 200.63% | -198.89pp | 6.42% | 0.140 |
| 2026-02-05 .. 2026-05-06 | 17.69% | 211.46% | -193.77pp | 6.42% | 0.179 |
| 2026-02-12 .. 2026-05-13 | 1.99% | 191.98% | -190.00pp | 6.42% | 0.113 |
| 2025-04-08 .. 2025-07-09 | 82.77% | 223.03% | -140.26pp | 16.32% | 0.552 |

Top bad-window feature deltas:

| feature | delta_z | spearman_to_excess |
| --- | ---: | ---: |
| vol_of_vol_21 | 1.194 | -0.022 |
| jump_variation_5 | 1.194 | 0.009 |
| realized_vol_63 | 1.093 | -0.011 |
| mom_21 | 0.813 | -0.190 |
| relative_strength_21 | 0.803 | -0.188 |
| expectancy_proxy_21 | 0.795 | -0.213 |

## Decision

The current paper model still has strong full-period strict-OOS excess, but the
recent risk is a recovery/upside-capture failure: high-volatility rebound windows
show very low average position while buy-and-hold SOXL compounds rapidly.

Do not open another broad PPO retraining branch from this evidence alone. The
next valid loop should be a fixed-actor recovery/re-entry counterfactual that
targets high `vol_of_vol_21` / `jump_variation_5` plus positive momentum or
relative-strength states, then checks whether the order/position mapping changes
without breaking 2022 defense.

The feature deltas are diagnostic hypotheses only. They are not promoted factors
until they pass factor IC, single-signal backtest, and walk-forward gates.

## Artifacts

```text
runtime/production/card/oos_poor_window_feature_diagnostics_v1/oos_diagnostic_review.md
runtime/production/card/oos_poor_window_feature_diagnostics_v1/poor_windows.csv
runtime/production/card/oos_poor_window_feature_diagnostics_v1/feature_differences.csv
runtime/production/card/oos_poor_window_feature_diagnostics_v1/manifest.json
```
