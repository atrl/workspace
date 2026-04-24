# Execution Overlay Knowledge

## Retained Insight

Fixed-actor overlays are the cleanest way to separate actor behavior from position-budget mapping, mode routing, and runtime execution risk controls.

## Failed Pattern

Generic overlay retuning is not enough when `test_2022` max drawdown and average position remain too high; widening favorable-mode envelopes also degraded other windows.

## Use With Skills

- `post-train-backtest-review` for fixed-actor overlay comparison.
- `dashboard` for visual verification of execution/dashboard outputs.

## Evidence Path

- `memory/2026-04-23-position-budget-productionization.md`
- `memory/2026-04-23-risk-budget-usage-v1.md`
- `memory/2026-04-22-balanced-rule-residual-mode-v1-train.md`

## Reusable Method

Keep actor fixed, run daily replay or overlay sweeps, then compare mode/window slices before deciding whether retraining is necessary.

## Next Valid Experiment Type

Improve mode classification or transition detection before adding more favorable-mode exposure.
