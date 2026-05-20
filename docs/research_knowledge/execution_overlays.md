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

The drawdown attribution artifact confirms the next mechanism should manage the
post-release path rather than the entry predicate. For the guarded-short variant,
the only execution-ready release order was `2024-03-28`, before the variant
MaxDD trough on `2024-08-07`; release-to-trough average position delta was
`+0.0204` and max position delta was `+0.3140`. Later gate-eligible orders on
`2024-05-13` and `2024-07-09` also precede the trough, so the next test should
cap or de-risk release-state exposure after activation.

The release-state target cap is the first walk-forward selective-release variant
that improves both recovery return and drawdown. In
`selective_release_recovery_post_release_cap_v1`, `wf_2022` remains unchanged,
while `wf_2023_2024` improves from `-20.21%` / MaxDD `38.54%` to `-11.37%` /
MaxDD `27.66%`. The same variant is a no-op on the fixed full-window actor
grid, so it does not disturb the original proxy/2020/2022/2023-2024
counterfactual evidence. Treat it as a promotion-review candidate, not yet a
runtime profile.

The promotion-review packaging loop confirms that distinction. The review card
for `selective_release_recovery_post_release_cap_v1` passes the walk-forward and
fixed-actor checks, but records `runtime_auto_select_allowed=false` because the
overlay still requires explicit production governance before scheduler
auto-selection.

The profile-bound dry-run loop closes the runtime plumbing gap without changing
the auto-selection decision. `balanced_rule_residual_v22` now binds
`selective_release_recovery_post_release_cap_v1`, runtime adds
`release_signals_v1`, paper dry-run emits overlay diagnostics, and non-dry-run
orders can advance overlay state. The blocker is now governance only:
`auto_select_allowed=false` and the base v22 actor sidecar still reports
`walk-forward promotion gate not evaluated`.

The scheduler auto-selection path now enforces that same governance in code.
`best_model_for_family()` skips a profile-bound execution overlay when
`execution_overlay.auto_select_allowed=false`, even if the model summary itself
is marked promoted. Manual `--profile` dry-runs remain possible for review.

The runtime selection sidecar review closes the current promotion question:
`balanced_rule_residual_v22` plus the post-release cap is blocked for scheduled
selection because the base model sidecar has not completed walk-forward
promotion and the profile overlay flag remains false. The execution line should
not continue through summary patching; it needs real walk-forward model evidence
or should yield to the next narrow model experiment.

The SOXX fixed-actor diagnostic shows SOXX can still be useful when kept out of
the actor state. `soxx_relative_weakness_cap_v1` caps the fixed
`balanced_rule_residual_v22` target only when SOXX is weak, SOXL/SOXX ratio
momentum is weak, and `SIGNED_TE_REVERSION_10` is not strong enough to justify a
reversion exception. It improves proxy return from `1511.97%` to `1621.98%`,
improves `test_2023_2024` return from `57.12%` to `72.61%`, and leaves
`test_2022` unchanged. Treat it as a candidate separation test, not promotion
evidence: `test_2022` had `44` eligible days but `0` cap-changed days, so the
next valid step is walk-forward execution absorption validation.

The SOXX cap walk-forward validation blocks that candidate. On saved
`balanced_rule_residual_v22` actors, `wf_2022` has `44` executable eligible days
but `0` cap-changed days, and `wf_2023_2024` has `27` executable eligible days /
`6` cap-changed days but `0` position-changed days and no return improvement.
`wf_2020` does change positions but loses `6.50pp` of return. Do not tune this
cap threshold or bind it to runtime; direct SOXX actor features and SOXX
fixed-actor cap have both failed the promotion path.

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

The SOXX relative cap is now blocked after walk-forward absorption validation.
Avoid full trade-lock bypass, unguarded shortening, static entry-threshold
tuning, more favorable-mode exposure, another direct release-pack retrain,
another direct SOXX actor-feature retrain, or SOXX cap threshold tuning. The
next model-optimization loop should use a distinct mechanism; otherwise resume
the production/dashboard side of the user objective.
