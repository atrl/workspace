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

Diagnose target/position absorption on `wf_2023_2024` eligible release days.
The next valid execution test should explicitly change order generation, e.g.
target-lift vs rebalance-threshold bypass under the existing gate. Avoid adding
more favorable-mode exposure or another direct release-pack retrain.
