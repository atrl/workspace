# Execution Overlay Knowledge

## Retained Insight

Fixed-actor overlays are the cleanest way to separate actor behavior from position-budget mapping, mode routing, and runtime execution risk controls.

The 2026-05-14 `selective_release_overlay_v1` loop strengthens this stance:
fixed `balanced_rule_residual_v22` actor plus a gated release overlay improved
`test_proxy` and `test_2023_2024` while leaving `test_2022` unchanged. This is
stronger evidence for execution gating than another ungated release-pack PPO
retrain.

The 2026-05-15 `selective_release_walkforward_v1` loop blocks direct runtime
promotion for that gate: saved walk-forward actors show `0` changed days across
all tested variants. In `wf_2023_2024`, release signals do become eligible, but
the boost is absorbed by actor target / execution mapping instead of changing
orders.

The 2026-05-20 trace and bypass loops narrowed the absorber to trade lock, then
rejected a full bypass. `selective_release_recovery_trade_lock_bypass_v1`
created orders in `wf_2023_2024` (`changed_days=183`, trades `32` vs `22`) while
leaving `wf_2022` unchanged, but it worsened recovery return and MaxDD. Treat
full trade-lock bypass as a diagnostic failure, not a runtime mechanism.

The follow-up `selective_release_recovery_trade_lock_short_v1` cap is cleaner
than full bypass but still not promotable. With `trade_lock_cap=1`,
`wf_2023_2024` return improved from `-20.21%` to `-17.87%` and trades only rose
from `22` to `24`, while `wf_2022` stayed unchanged; however MaxDD widened from
`38.54%` to `40.65%`. Treat this as evidence that the next mechanism needs a
drawdown / exposure-gap guard around release-gated shortening.

The static guarded variant did not solve the drawdown problem. In
`selective_release_recovery_trade_lock_guarded_short_v1`, both target boost and
execution relaxations were gated by `current_drawdown<=0.03` and
`min_release_exposure_gap>=0.10`; it still widened `wf_2023_2024` MaxDD to
`42.91%` while only improving return to `-19.63%`. Do not keep tuning static
entry thresholds without first attributing the post-release drawdown path.

## Failed Pattern

Generic overlay retuning is not enough when `test_2022` max drawdown and average position remain too high; widening favorable-mode envelopes also degraded other windows.

## Use With Skills

- `post-train-backtest-review` for fixed-actor overlay comparison.
- `dashboard` for visual verification of execution/dashboard outputs.

## Evidence Path

- `runtime/execution/knowledge.md`
- `runtime/execution/knowledge.md`
- `runtime/model/knowledge.md`
- `runtime/execution/card/selective_release_overlay_v1/selective_release_overlay_review.md`
- `runtime/execution/card/selective_release_walkforward_v1/selective_release_walkforward_review.md`

## Reusable Method

Keep actor fixed, run daily replay or overlay sweeps, then compare mode/window slices before deciding whether retraining is necessary.

## Next Valid Experiment Type

Before another release-gated entry rule, attribute the drawdown path caused by
the sparse guarded release order. The next execution mechanism should be
distinct from static entry-threshold tuning, for example explicit post-release
de-risking or a release target cap, and it must keep `wf_2022` unchanged while
bringing `wf_2023_2024` MaxDD back near baseline. Avoid full trade-lock bypass,
unguarded shortening, more favorable-mode exposure, or another direct
release-pack retrain.
