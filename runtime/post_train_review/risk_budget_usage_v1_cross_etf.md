# risk_budget_usage_v1 cross-ETF diagnostic

This diagnostic freezes the merged risk-budget usage mechanism and tests two separable capabilities:

- `fixed_signal_transfer`: use the existing SOXL fixed replay position path on each ETF's own price series.
- `actor_budget_runtime`: run the retained `imitation_v4` actor under the `risk_budget_usage_v1` budget-usage mapping on each ETF.

The retained `balanced_rule_residual_mode_v1` actor bundle is absent, so this is not a full two-actor zero-shot composer.

| ETF | fixed proxy ret/sh/dd/pos | fixed 2022 ret/sh/dd/pos | actor proxy ret/sh/dd/pos | actor 2022 ret/sh/dd/pos | buyhold proxy ret/sh/dd | decision |
| --- | --- | --- | --- | --- | --- | --- |
| SOXL | 2030.3% / 1.252 / 44.9% / pos 0.391 | -16.9% / -0.351 / 25.5% / pos 0.214 | 144.4% / 0.827 / 22.1% / pos 0.187 | -18.1% / -1.318 / 22.0% / pos 0.121 | 154.3% / 0.694 / 90.5% / pos 1.000 | risk transfers, return weak |
| TQQQ | 823.4% / 1.287 / 29.6% / pos 0.391 | -25.4% / -1.078 / 29.2% / pos 0.214 | 51.2% / 0.473 / 22.7% / pos 0.269 | -13.3% / -1.508 / 13.7% / pos 0.115 | 284.3% / 0.667 / 81.7% / pos 1.000 | risk transfers, return weak |
| UPRO | 427.4% / 1.148 / 22.8% / pos 0.391 | -12.6% / -0.643 / 15.8% / pos 0.214 | 35.2% / 0.436 / 22.3% / pos 0.299 | -17.9% / -1.793 / 18.8% / pos 0.142 | 180.5% / 0.583 / 76.8% / pos 1.000 | risk transfers, return weak |
| SPXL | 429.7% / 1.154 / 22.9% / pos 0.391 | -12.5% / -0.637 / 15.7% / pos 0.214 | 28.1% / 0.372 / 22.9% / pos 0.296 | -17.9% / -1.801 / 19.1% / pos 0.149 | 183.2% / 0.585 / 76.9% / pos 1.000 | risk transfers, return weak |
| TECL | 965.5% / 1.260 / 30.6% / pos 0.391 | -19.9% / -0.700 / 24.1% / pos 0.214 | 46.0% / 0.429 / 23.7% / pos 0.238 | -19.0% / -1.609 / 20.7% / pos 0.131 | 270.1% / 0.669 / 78.0% / pos 1.000 | risk transfers, return weak |

## Interpretation

A result only counts as promotable if actor_budget_runtime, not just fixed_signal_transfer, keeps drawdown controlled and preserves enough participation across ETFs. Fixed signal transfer is a timing-path stress test and remains exposed to SOXL test-path selection bias.
