# Selective Release Walk-Forward Actor Review

- Profile: `balanced_rule_residual_v22`
- Dataset feature packs for overlay columns: `['vix_term', 'release_signals_v1']`
- Role: walk-forward actor validation before runtime overlay promotion

## Results

| actor_window | variant | return | sharpe | max_dd | avg_pos | trades | eligible_days | changed_days | absorbed_days | delta_return | delta_max_dd |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wf_2020 | baseline | 113.17% | 2.015 | 19.47% | 0.5322 | 22 | 0 | 0 | 0 | +0.00pp | +0.00pp |
| wf_2020 | selective_release_cicc_v1 | 113.17% | 2.015 | 19.47% | 0.5322 | 22 | 0 | 0 | 0 | +0.00pp | +0.00pp |
| wf_2020 | selective_release_tight_v1 | 113.17% | 2.015 | 19.47% | 0.5322 | 22 | 0 | 0 | 0 | +0.00pp | +0.00pp |
| wf_2020 | selective_release_recovery_v1 | 113.17% | 2.015 | 19.47% | 0.5322 | 22 | 0 | 0 | 0 | +0.00pp | +0.00pp |
| wf_2020 | selective_release_recovery_threshold_bypass_v1 | 113.17% | 2.015 | 19.47% | 0.5322 | 22 | 0 | 0 | 0 | +0.00pp | +0.00pp |
| wf_2020 | selective_release_recovery_target_lift_v1 | 113.17% | 2.015 | 19.47% | 0.5322 | 22 | 0 | 0 | 0 | +0.00pp | +0.00pp |
| wf_2022 | baseline | -34.11% | -0.926 | 38.27% | 0.3216 | 22 | 0 | 0 | 0 | +0.00pp | +0.00pp |
| wf_2022 | selective_release_cicc_v1 | -34.11% | -0.926 | 38.27% | 0.3216 | 22 | 0 | 0 | 0 | +0.00pp | +0.00pp |
| wf_2022 | selective_release_tight_v1 | -34.11% | -0.926 | 38.27% | 0.3216 | 22 | 0 | 0 | 0 | +0.00pp | +0.00pp |
| wf_2022 | selective_release_recovery_v1 | -34.11% | -0.926 | 38.27% | 0.3216 | 22 | 0 | 0 | 0 | +0.00pp | +0.00pp |
| wf_2022 | selective_release_recovery_threshold_bypass_v1 | -34.11% | -0.926 | 38.27% | 0.3216 | 22 | 0 | 0 | 0 | +0.00pp | +0.00pp |
| wf_2022 | selective_release_recovery_target_lift_v1 | -34.11% | -0.926 | 38.27% | 0.3216 | 22 | 0 | 0 | 0 | +0.00pp | +0.00pp |
| wf_2023_2024 | baseline | -20.21% | -0.277 | 38.54% | 0.4764 | 22 | 0 | 0 | 0 | +0.00pp | +0.00pp |
| wf_2023_2024 | selective_release_cicc_v1 | -20.21% | -0.277 | 38.54% | 0.4764 | 22 | 16 | 0 | 16 | +0.00pp | +0.00pp |
| wf_2023_2024 | selective_release_tight_v1 | -20.21% | -0.277 | 38.54% | 0.4764 | 22 | 6 | 0 | 6 | +0.00pp | +0.00pp |
| wf_2023_2024 | selective_release_recovery_v1 | -20.21% | -0.277 | 38.54% | 0.4764 | 22 | 18 | 0 | 18 | +0.00pp | +0.00pp |
| wf_2023_2024 | selective_release_recovery_threshold_bypass_v1 | -20.21% | -0.277 | 38.54% | 0.4764 | 22 | 18 | 0 | 18 | +0.00pp | +0.00pp |
| wf_2023_2024 | selective_release_recovery_target_lift_v1 | -20.21% | -0.277 | 38.54% | 0.4764 | 22 | 18 | 0 | 18 | +0.00pp | +0.00pp |

## Readout

- This keeps the actor fixed within each walk-forward checkpoint and changes only the execution overlay.
- A promotable overlay should improve the recovery window without reopening `wf_2022` drawdown or average exposure.

## Retained Gate Check

- `wf_2022`: return `-34.11%` vs baseline `-34.11%`; max_dd `38.27%` vs `38.27%`; eligible_days `0`; changed_days `0`.
- `wf_2023_2024`: return `-20.21%` vs baseline `-20.21%`; max_dd `38.54%` vs `38.54%`; eligible_days `18`; changed_days `0`.

## Order-Generation Diagnostic

- `selective_release_recovery_threshold_bypass_v1`: eligible_days `18`, changed_days `0`, absorbed_days `18`, return `-20.21%`, max_dd `38.54%`.
- `selective_release_recovery_target_lift_v1`: eligible_days `18`, changed_days `0`, absorbed_days `18`, return `-20.21%`, max_dd `38.54%`.

- Lowering the rebalance threshold and forcing a target lift still produced no changed days on the eligible `wf_2023_2024` release dates.
- `wf_2022` remains unchanged because there are no eligible release days, so the defensive gate is still intact.
- The eligible-day trace below separates upstream execution guards from downstream min-rebalance and lot-rounding symptoms before any new rule or retrain.

## Eligible-Day Execution Trace Summary

- `selective_release_recovery_threshold_bypass_v1`: traced `16` of `18` eligible days; blockers trade_lock `15`, buy_cooldown `0`, min_rebalance `0`, step_lock `0`, lot_rounding `15`; max target after overlay `0.8500`, after guards `0.8490`; primary `ordered` 1, `trade_lock` 15.
- `selective_release_recovery_target_lift_v1`: traced `16` of `18` eligible days; blockers trade_lock `12`, buy_cooldown `0`, min_rebalance `15`, step_lock `0`, lot_rounding `0`; max target after overlay `0.8500`, after guards `0.8500`; primary `min_rebalance` 3, `ordered` 1, `trade_lock` 12.
- Readout: trade lock is the dominant upstream absorber; min-rebalance and lot-rounding counts mostly appear after the target has already been collapsed back toward current position.
- Full eligible-day trace: `selective_release_walkforward_execution_trace.csv`.
