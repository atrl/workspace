# Current Execution Plan

- Date: 2026-05-20
- Direction: `execution`
- Active line: `selective_release_order_generation_v1`
- Baseline: `balanced_rule_residual_v22`
- Direct comparison: `selective_release_walkforward_v1`
- Role: `diagnose`

## Loop

Test whether retained selective-release eligible days are absorbed only by the visible rebalance threshold or too-small target lift.

## Result

- `selective_release_recovery_threshold_bypass_v1`: `wf_2023_2024` eligible_days `18`, changed_days `0`.
- `selective_release_recovery_target_lift_v1`: `wf_2023_2024` eligible_days `18`, changed_days `0`.
- `wf_2022` remains unchanged because the retained release gate has `0` eligible days.

## Next Action

Instrument eligible-day execution state before testing another rule: current position, target before/after overlay, min rebalance, trade lock, buy cooldown, cap state, and lot rounding.
