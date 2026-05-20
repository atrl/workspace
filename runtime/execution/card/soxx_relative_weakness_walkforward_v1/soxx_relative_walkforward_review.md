# SOXX Relative Weakness Walk-Forward Review

- Profile: `balanced_rule_residual_v22`
- Dataset feature packs for overlay columns: `['vix_term', 'soxx_signals', 'transition_signals']`
- Diagnostic variant: `soxx_relative_weakness_cap_v1`
- Decision: `blocked`
- Role: saved walk-forward actor absorption validation before runtime overlay promotion

## Results

| actor_window | variant | return | sharpe | max_dd | avg_pos | trades | feature_eligible_days | eligible_days | cap_changed_days | changed_days | eligible_order_days | target_absorbed_days | position_absorbed_days | delta_return | delta_max_dd |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wf_2020 | baseline | 113.17% | 2.015 | 19.47% | 0.5322 | 22 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | +0.00pp | +0.00pp |
| wf_2020 | soxx_relative_weakness_cap_v1 | 106.66% | 1.963 | 19.47% | 0.5262 | 22 | 29 | 15 | 12 | 39 | 1 | 3 | 0 | -6.50pp | +0.00pp |
| wf_2022 | baseline | -34.11% | -0.926 | 38.27% | 0.3216 | 22 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | +0.00pp | +0.00pp |
| wf_2022 | soxx_relative_weakness_cap_v1 | -34.11% | -0.926 | 38.27% | 0.3216 | 22 | 66 | 44 | 0 | 0 | 6 | 44 | 44 | +0.00pp | +0.00pp |
| wf_2023_2024 | baseline | -20.21% | -0.277 | 38.54% | 0.4764 | 22 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | +0.00pp | +0.00pp |
| wf_2023_2024 | soxx_relative_weakness_cap_v1 | -20.21% | -0.277 | 38.54% | 0.4764 | 22 | 34 | 27 | 6 | 0 | 1 | 21 | 27 | +0.00pp | +0.00pp |

## Decision Checks

| check | passed |
|---|---:|
| `wf_2022_avg_pos_preserved` | `true` |
| `wf_2022_max_dd_preserved` | `true` |
| `wf_2023_2024_recovery_improved` | `false` |
| `wf_2023_2024_drawdown_not_materially_worse` | `true` |
| `walk_forward_target_changed` | `true` |
| `walk_forward_position_changed` | `true` |

## Readout

- `wf_2022` delta avg_pos: `+0.000000`.
- `wf_2022` delta max_dd: `+0.0000pp`.
- `wf_2023_2024` delta return: `+0.0000pp`.
- `wf_2023_2024` delta max_dd: `+0.0000pp`.
- Total cap-changed days: `18`; total position changed days: `39`.
- This review validates saved walk-forward actor behavior only; it is not runtime binding or auto-select approval.

## Eligible-Day Trace Summary

- `wf_2020`: traced `15` eligible days; cap_changed `12`; ordered `1`; primary `min_rebalance` 5, `ordered` 1, `trade_lock` 9.
- `wf_2022`: traced `44` eligible days; cap_changed `0`; ordered `6`; primary `buy_cooldown` 7, `ordered` 6, `trade_lock` 31.
- `wf_2023_2024`: traced `27` eligible days; cap_changed `6`; ordered `1`; primary `min_rebalance` 4, `ordered` 1, `trade_lock` 22.
- Full eligible-day trace: `soxx_relative_walkforward_execution_trace.csv`.
