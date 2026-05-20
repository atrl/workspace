# Current Execution Plan

- Date: 2026-05-20
- Direction: `execution`
- Active line: `selective_release_drawdown_attribution_v1`
- Baseline: `balanced_rule_residual_v22`
- Direct comparison: `selective_release_walkforward_v1`
- Role: `diagnose`

## Completed Loops

Test whether retained selective-release eligible days are absorbed only by the visible rebalance threshold or too-small target lift, then instrument actionable eligible-day execution state.

## Result

- `selective_release_recovery_threshold_bypass_v1`: `wf_2023_2024` eligible_days `18`, changed_days `0`.
- `selective_release_recovery_target_lift_v1`: `wf_2023_2024` eligible_days `18`, changed_days `0`.
- `wf_2022` remains unchanged because the retained release gate has `0` eligible days.
- Eligible-day trace narrowed the dominant upstream blocker to trade lock:
  - threshold bypass: `15 / 16` actionable eligible rows have primary blocker `trade_lock`
  - target lift: `12 / 16` actionable eligible rows have primary blocker `trade_lock`; `3 / 16` are min-rebalance near-current cases
- Release-gated full trade-lock bypass created orders but failed promotion quality:
  - `wf_2022`: unchanged, `0` eligible days
  - `wf_2023_2024`: changed_days `183`, trades `32` vs `22`, return `-21.43%` vs `-20.21%`, MaxDD `40.28%` vs `38.54%`
- Release-gated trade-lock shortening with `trade_lock_cap=1` is cleaner but still not promotable:
  - `wf_2022`: unchanged, `0` eligible days
  - `wf_2023_2024`: changed_days `175`, trades `24` vs `22`, return `-17.87%` vs `-20.21%`, Sharpe `-0.229` vs `-0.277`, MaxDD `40.65%` vs `38.54%`
  - eligible trace: `ordered=3/16`, `trade_lock=12/16`, `buy_cooldown=1/16`
- Static drawdown / exposure-gap guarded shortening also failed promotion:
  - guard: `current_drawdown<=0.03`, `min_release_exposure_gap>=0.10`
  - `wf_2022`: unchanged, `0` eligible days
  - `wf_2023_2024`: execution-ready rows `1/16`, changed_days `185`, trades `21` vs `22`, return `-19.63%` vs `-20.21%`, MaxDD `42.91%` vs `38.54%`
- Path-level drawdown attribution now explains the guarded-shortening failure:
  - first execution-ready release order: `2024-03-28`
  - guarded variant max-DD peak/trough: `2024-06-18 -> 2024-08-07`
  - delta MaxDD: `+4.37pp`
  - release-to-trough avg/max position delta vs baseline: `+0.0204` / `+0.3140`
  - later gate-eligible ordered dates: `2024-05-13`, `2024-07-09`

## Next Action

Do not promote full bypass, unguarded shortening, or static guarded shortening. Attribution supports the next distinct loop: release-state target cap or post-release de-risking, preserving `wf_2022` unchanged while bringing `wf_2023_2024` MaxDD back near baseline.
