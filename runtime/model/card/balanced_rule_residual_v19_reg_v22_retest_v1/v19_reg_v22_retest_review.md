# v19_reg + v22 Feature Retest Review

Date: 2026-05-21

Profile: `balanced_rule_residual_v19_reg_v22_retest_v1`

Command:

```bash
PYTHONPATH=src:. qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_rule_residual_v19_reg_v22_retest_v1 --skip-data-refresh --seed 20260521
```

## Baselines

| Role | Profile | Summary |
| --- | --- | --- |
| Canonical baseline | `balanced_rule_residual_v22` | `runtime/model/card/balanced_rule_residual_v22/soxl_balanced_rule_residual_v22_summary.json` |
| Direct comparison | `balanced_rule_residual_v19_reg` | `runtime/model/card/balanced_rule_residual_v19_reg/soxl_balanced_rule_residual_v19_reg_summary.json` |
| Retest | `balanced_rule_residual_v19_reg_v22_retest_v1` | `runtime/model/card/balanced_rule_residual_v19_reg_v22_retest_v1/soxl_balanced_rule_residual_v19_reg_v22_retest_v1_summary.json` |

## Fixed-Split Readout

| Profile | Proxy Ret | Proxy Sharpe | Proxy MaxDD | 2022 Ret | 2022 Sharpe | 2022 MaxDD | 2022 Avg Pos | 2023-2024 Ret | 2023-2024 MaxDD | Promotion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `balanced_rule_residual_v22` | 695.5% | 0.941 | 54.5% | -24.5% | -0.911 | 32.7% | 0.269 | 37.9% | 32.4% | pass |
| `balanced_rule_residual_v19_reg` | 559.6% | 0.913 | 47.7% | -21.9% | -0.864 | 30.2% | 0.250 | 58.7% | 35.1% | blocked by WF |
| `balanced_rule_residual_v19_reg_v22_retest_v1` | 1280.4% | 1.020 | 59.6% | -54.1% | -1.428 | 58.3% | 0.381 | 44.5% | 40.8% | blocked |

## Promotion Checks

| Check | Actual | Limit | Result |
| --- | ---: | ---: | --- |
| `test_2022_max_dd` | 58.30% | 55.00% | fail |
| `test_2022_avg_pos` | 0.381 | 0.350 | fail |
| `test_proxy_sharpe` | 1.020 | 0.500 | pass |

## Checkpoint Retention

Retained validation checkpoints:

| Iter | Score | Valid Ret | Valid Sharpe | Valid MaxDD | Reason |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 89.48 | 61.8% | 1.571 | 31.2% | stage, top_n |
| 20 | 88.62 | 58.1% | 1.686 | 26.6% | stage |
| 40 | 89.32 | 59.0% | 1.674 | 26.6% | stage, top_n |
| 60 | 88.62 | 58.1% | 1.686 | 26.6% | stage |
| 80 | 70.83 | 44.6% | 1.468 | 26.4% | stage |

The selected actor is still not promotable because the fixed-split 2022 risk
checks fail. The retained checkpoints are kept as replay evidence for this
ablation only.

## Decision

Do not run walk-forward. The combination improves proxy return and 2020 return,
but it breaks the bear-market risk budget versus both baselines:

- Versus `balanced_rule_residual_v22`, 2022 MaxDD widens from `32.7%` to
  `58.3%`, and avg position rises from `0.269` to `0.381`.
- Versus `balanced_rule_residual_v19_reg`, 2022 MaxDD widens from `30.2%` to
  `58.3%`, while 2023-2024 return falls from `58.7%` to `44.5%`.

Interpretation: injecting the v22 VIX term-structure pack into the v19_reg
regularized network creates a higher-exposure recovery/bull-market profile, not
a better risk-controlled successor to canonical v22.
