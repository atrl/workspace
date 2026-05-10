# v22 Release v1 Backtest Review

- Profile: `balanced_rule_residual_v22_release_v1`
- Baseline: `balanced_rule_residual_v22`
- Training command:
  `PYTHONPATH=. qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_rule_residual_v22_release_v1 --skip-data-refresh --seed 20260411`
- Replay method:
  `scripts/run_segment_backtest.py` under the current harness after the 2026-05-10 terminal-date fix

## Corrected Replay Comparison

| window | `balanced_rule_residual_v22` | `balanced_rule_residual_v22_release_v1` | delta vs `v22` |
| --- | --- | --- | --- |
| `test_proxy` | `1099.12% / 0.976 / 53.21% / 0.4984 / 169 trades` | `1350.25% / 1.010 / 53.26% / 0.5204 / 172 trades` | `+251.13pp` return, `+0.034` Sharpe, `+0.05pp` MaxDD, `+0.0220` avg_pos |
| `test_2020` | `37.34% / 0.812 / 45.75% / 0.4079 / 23 trades` | `42.66% / 0.860 / 44.95% / 0.4303 / 22 trades` | `+5.32pp` return, `+0.048` Sharpe, `-0.80pp` MaxDD, `+0.0224` avg_pos |
| `test_2022` | `-41.75% / -1.335 / 43.63% / 0.2413 / 21 trades` | `-46.34% / -1.480 / 48.45% / 0.2518 / 19 trades` | `-4.59pp` return, `-0.145` Sharpe, `+4.82pp` MaxDD, `+0.0105` avg_pos |
| `test_2023_2024` | `180.24% / 1.336 / 31.91% / 0.4855 / 52 trades` | `234.34% / 1.479 / 32.80% / 0.5089 / 47 trades` | `+54.10pp` return, `+0.143` Sharpe, `+0.89pp` MaxDD, `+0.0234` avg_pos |

## Training Readout

- best valid snapshot: `61.0% / 1.662 / 27.84%`
- iter `20` remained close: `60.3% / 1.684 / 26.89%`
- iter `40` degraded to `52.9% / 1.635 / 27.04%`

The mechanism is visible early. More iterations are not the missing ingredient here.

## Readout

- The release pack clearly helps upside capture and overall proxy return.
- The same pack also loosens `2022` defense. This is the key failure for replacement candidacy.
- `test_2022 avg_pos` still passes the hard gate (`0.2518 < 0.35`), but the direction of change is wrong versus the direct baseline (`0.2413 -> 0.2518`).
- `test_2022 maxdd` widens from `43.63%` to `48.45%`.
- Trade count does not rise much, so this is not a churn blowout. The issue is behavioral: the actor learned a broader risk-on bias, not a conditional release rule.

## Decision

- Status: keep as `ablation`
- Do not promote or replace `balanced_rule_residual_v22`
- Next valid mechanism:
  build a low-churn fixed-actor selective release overlay that gates on
  `RB_COOLDOWN_PROXY`, `TRANS_VIX_DIVERGENCE`, and `CICC_RETURN_VOLUME_SYNC`
  instead of injecting the trio directly into PPO state
