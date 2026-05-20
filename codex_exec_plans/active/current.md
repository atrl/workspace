# Current Execution Plan

- Date: 2026-05-21
- Direction: `execution`
- Active line: `selective_release_post_release_cap_v1`
- Baseline: `balanced_rule_residual_v22`
- Direct comparison: `selective_release_walkforward_v1`
- Role: `diagnose`, now promotion-review candidate

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
- Release-state post-release cap is the first selective-release walk-forward variant that improves both recovery return and MaxDD:
  - mechanism: 90-day release state, `post_release_drawdown_threshold=0.04`, `post_release_target_cap=0.55`, sell-only post-release rebalance/trade-lock relaxation
  - `wf_2022`: unchanged, `0` eligible days
  - `wf_2023_2024`: return `-11.37%` vs `-20.21%`, Sharpe `-0.133` vs `-0.277`, MaxDD `27.66%` vs `38.54%`, trades `27` vs `22`
  - fixed full-window actor overlay grid: no-op on proxy/2020/2022/2023-2024
- Promotion governance packaging is now explicit:
  - overlay configs live in `src/soxl_quant/policy/execution_overlays.py`
  - review card: `runtime/execution/card/selective_release_post_release_cap_v1/promotion_review.md`
  - review checks pass for `wf_2022_unchanged`, `wf_2023_2024_return_improved`, `wf_2023_2024_maxdd_improved`, and `fixed_actor_noop`
  - `runtime_auto_select_allowed=false` because scheduler auto-selection still requires explicit profile flag and model-promotion approval
- Profile-bound runtime dry-run is now explicit:
  - `balanced_rule_residual_v22` binds `execution_overlay.name=selective_release_recovery_post_release_cap_v1`
  - runtime automatically loads `release_signals_v1` for that profile and emits overlay diagnostics in `SignalIntent`
  - non-dry-run paper execution advances/saves overlay state after submitted orders
  - fixed-profile dry-run produced a `2026-05-14` signal with the overlay fields and remained blocked by `walk-forward promotion gate not evaluated`
  - promotion review now records `runtime_profile_bound=true`, but `runtime_auto_select_allowed=false` remains correct because `auto_select_allowed=false` and model promotion is not approved
- Runtime auto-select now enforces the profile gate:
  - `best_model_for_family()` skips profiles that bind an execution overlay while `auto_select_allowed=false`
  - a regression test proves a promoted high-score v22 actor cannot beat v19 unless the profile overlay approval flag is explicit
  - current live `auto_select_profile("balanced")` remains `balanced_rule_residual_v19`
- Runtime selection sidecar review now exists:
  - `runtime/model/card/balanced_rule_residual_v22/runtime_selection_review.md`
  - decision `blocked`
  - blockers: v22 model walk-forward gate not evaluated, model promotion sidecar not eligible, profile overlay `auto_select_allowed=false`

## Next Action

Do not promote full bypass, unguarded shortening, static guarded shortening, or the post-release cap directly into scheduler auto-selection. The runtime dry-run path, auto-select profile gate, and runtime selection sidecar now all block v22/post-release-cap from scheduled use; next progress should be a separate narrow model-optimization experiment unless new v22/post-release-cap walk-forward model evidence is produced first.
