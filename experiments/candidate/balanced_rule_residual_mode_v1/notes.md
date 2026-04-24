---
title: balanced_rule_residual_mode_v1
direction: model
status: candidate
---

# balanced_rule_residual_mode_v1 — Mode Hierarchy Scaffold

## Motivation

The current search space has plateaued on threshold, cooldown, and overlay
retuning. This branch moves to a larger mechanism change: persistent
high-level trading modes that constrain the existing residual actor.

## External Reference

- `TradeR`
- `EarnHFT`
- `Deep Stock Trading`

These are the reference class for temporal abstraction / hierarchical trading
control. This branch only implements the first harness-compatible scaffold:
rule-routed modes plus mode-constrained residual actions.

## Hypothesis

Encoding multi-day behavior modes should improve stress-period behavior more
cleanly than more threshold drift because it gives the policy a real temporal
abstraction:

- `trend_hold`
- `defensive_hold`
- `crash_rebuild`
- `flat_wait`

## Closest Historical Success

- `balanced_rule_residual_v19`
- `dwell_hold_v2`
- `regime_route_v1`

## Closest Historical Failure

- `balanced_rule_residual_v24`

That branch reduced action frequency without introducing persistent behavior
modes, so it did not create true temporal abstraction.

## Current Role

- first mode-hierarchy candidate over the retained `v19` actor family
- intended as a `replace` experiment, not another overlay ablation

## Run Result

- command:
  - `PYTHONPATH=. qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_rule_residual_mode_v1 --skip-data-refresh --seed 12345`
- training completed end-to-end without harness breakage
- summary:
  - valid: `36.2% / Sharpe 1.17 / MaxDD 26.1%`
  - test_2020: `64.3% / MaxDD 10.6%`
  - test_2022: `-13.1% / MaxDD 16.5% / avg_pos 0.135`
  - test_2023_2024: `43.0% / MaxDD 20.1%`
  - test_proxy: `886.1% / Sharpe 1.09 / MaxDD 42.3% / avg_pos 0.450`

## Delta Vs `balanced_rule_residual_v19`

- valid return falls `56.6% -> 36.2%`, so the mode bands are materially more conservative during validation
- `test_2020` return falls `109.4% -> 64.3%`, but MaxDD improves `18.1% -> 10.6%`
- `test_2022` improves meaningfully:
  - return `-22.5% -> -13.1%`
  - MaxDD `31.1% -> 16.5%`
  - avg_pos `0.261 -> 0.135`
- `test_2023_2024` keeps almost all return while cutting MaxDD `37.3% -> 20.1%`
- `test_proxy` improves on all three key axes:
  - return `735.1% -> 886.1%`
  - Sharpe `0.95 -> 1.09`
  - MaxDD `54.8% -> 42.3%`

## Verdict

- keep as `candidate`
- keep the mode hierarchy scaffold in the harness
- do not treat this as ready to replace `v19`

The mechanism appears real: persistent mode constraints materially improve weak
stress/restart windows and improve full proxy quality. The main cost is
over-suppressed bull participation, especially in `test_2020` and validation.
Next work should loosen `trend_hold` / `crash_rebuild` upside capture without
removing the defensive gains in `test_2022` and `test_2023_2024`.
