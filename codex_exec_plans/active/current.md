# Current Execution Plan

- Date: 2026-05-21
- Direction: `model_selection`
- Active line: `balanced_rule_residual_v23_v24_retention_audit`
- Baseline: `balanced_rule_residual_v22`
- Direct comparison: stale summary-only `balanced_rule_residual_v23` and
  `balanced_rule_residual_v24`
- Role: `diagnose` / cleanup

## Completed Loops

Production and runtime selection:

- Data freshness, scheduler inference, static/Vercel dashboard export, and
  simulated paper execution are no longer blockers.
- `balanced_rule_residual_v22` plus
  `selective_release_recovery_post_release_cap_v1` is the canonical scheduled
  runtime profile.
- The next live proof remains the next fresh post-close non-dry paper run
  selecting v22; the latest non-dry artifact is still the earlier
  `2026-05-20` v19 run because it executed before the v22 auto-select approval.

Execution overlay line:

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
- Later runtime governance closed those blockers:
  - `balanced_rule_residual_v22` is now canonical scheduled runtime selection
  - `selective_release_recovery_post_release_cap_v1` is explicitly
    `auto_select_allowed=true`
  - v19 is retained only as rollback evidence
- v23/v24 retention audit:
  - generated `v23_vs_v22_review.md` and `v24_vs_v22_review.md`
  - both profiles are summary-only historical artifacts with missing actors
  - registry rows mark both as `dropped`, `keep_actor=false`,
    `keep_summary=true`

## Next Action

Do not revive v23/v24 and do not tune SOXX cap thresholds. The next model
optimization loop needs a distinct governed mechanism with retained actors and
complete walk-forward evidence, while the next production proof remains the
first fresh post-close non-dry paper run that naturally selects v22.

2026-05-23 direction update: current request is routed as `model_selection`
with `factor` and `data` sidecars under row key
`soxx_tushare_factor_intake_v1`. The cleanup/factor loop added an optional
Tushare `us_daily_adj` cache path and refreshed SOXX factor evidence. Full
retraining is intentionally blocked until Tushare cache exists and at least one
new Tushare-derived factor clears IC/backtest/correlation gates; repeating
`balanced_rule_residual_v22_soxx` is a known duplicate failure.

2026-05-21 model-selection cleanup: `balanced_rule_residual_v19_reg_v22` was
audited as profile-only and marked dropped in the registry. Its older fixed-split
and started-walk-forward note has no durable actor, summary, or walk-forward
artifact behind it. Do not use it as the next branch unless a new governed
experiment card is created.

2026-05-21 v22 checkpoint asset audit: current v22 summary records only iter 0
and iter 80 selection rows, and only the final actor is retained. No iter
20/40/60 actors are available for replay, so checkpoint re-selection cannot
optimize the current model without a retrain or a future top-N/stage retention
harness change.

2026-05-21 production visibility update: `production/check_scheduler_health.sh`
now prints `runtime proof: pending ...` when the latest non-dry executed paper
artifact profile differs from the currently selected model profile. Current live
state is still pending: latest executed v19, selected model v22.

2026-05-21 Vercel audit: repo-owned static packaging is complete and validated
for Vercel serving (`vercel.json`, `dashboard.html`, `dashboard_data/index.json`,
`dashboard_data/model_versions.json`, 25 version payloads, latest
`paper_trading`). This does not prove a remote deployment URL is live.

2026-05-21 checkpoint-retention harness update: training profiles can now opt in
to default-off `checkpoint_retention` and persist stage/top-N actor bundles with
summary `retained_checkpoints`. This removes the v22 audit's immediate
replayability gap for future governed model runs; no existing profile behavior
changes until a profile enables it.
