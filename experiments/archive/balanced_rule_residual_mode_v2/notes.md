# balanced_rule_residual_mode_v2 — Favorable Mode Release

## Motivation

`balanced_rule_residual_mode_v1` validated the mode hierarchy itself, but it
paid too much in validation and `test_2020` upside. The next step is not to
remove modes; it is to make favorable modes less conservative while keeping the
defensive modes unchanged.

## External Reference

- `TradeR`
- `EarnHFT`
- `Deep Stock Trading`

The reference class is still hierarchical / temporally abstract trading
control. This branch narrows the follow-up to one concrete question: can the
favorable modes carry more upside without reopening the stress windows?

## Hypothesis

`mode_v1` already proved that persistent modes materially improve `test_2022`
and proxy drawdown. The likely remaining error is not the hierarchy itself but
over-tight favorable-mode envelopes:

- `trend_hold` should allow full participation and resist premature sells
- `crash_rebuild` should re-enter faster after stress release
- `defensive_hold` and `flat_wait` should stay unchanged

## Closest Historical Success

- `balanced_rule_residual_mode_v1`
- `balanced_rule_residual_v19`

## Closest Historical Failure

- `balanced_rule_residual_v24`

That branch tried to change behavior through coarser control frequency rather
than through explicit persistent modes.

## Current Role

- second mode-hierarchy candidate over the retained `v19` actor family
- intended as a `replace` experiment
- first candidate where favorable-mode behavior is fully profile-configurable

## Run Result

- command:
  - `PYTHONPATH=. qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_rule_residual_mode_v2 --skip-data-refresh --seed 12345`
- training completed end-to-end without harness breakage
- summary:
  - valid: `35.8% / Sharpe 1.18 / MaxDD 26.1%`
  - test_2020: `42.6% / MaxDD 22.9%`
  - test_2022: `-3.6% / MaxDD 13.5% / avg_pos 0.128`
  - test_2023_2024: `47.2% / MaxDD 28.7%`
  - test_proxy: `693.8% / Sharpe 1.01 / MaxDD 50.4% / avg_pos 0.450`

## Delta Vs `balanced_rule_residual_mode_v1`

- validation is effectively unchanged:
  - return `36.2% -> 35.8%`
  - Sharpe `1.17 -> 1.18`
  - MaxDD `26.1% -> 26.1%`
- `test_2020` gets materially worse, not better:
  - return `64.3% -> 42.6%`
  - MaxDD `10.6% -> 22.9%`
- `test_2022` improves further:
  - return `-13.1% -> -3.6%`
  - MaxDD `16.5% -> 13.5%`
  - avg_pos `0.135 -> 0.128`
- `test_2023_2024` return improves, but with worse DD:
  - return `43.0% -> 47.2%`
  - MaxDD `20.1% -> 28.7%`
- `test_proxy` gives back too much of `mode_v1`:
  - return `886.1% -> 693.8%`
  - Sharpe `1.09 -> 1.01`
  - MaxDD `42.3% -> 50.4%`

## Verdict

- drop this parameterization
- keep the config-owned mode hierarchy extension points in the harness
- keep `balanced_rule_residual_mode_v1` as the retained mode-hierarchy frontier

The favorable-mode loosening did improve `2022`, but it failed the actual goal:
it did not recover bull participation and it materially degraded the full proxy
quality that justified `mode_v1`. The main harness lesson is still valuable:
future mode experiments should stay in experiment packages rather than editing
`src`, but this specific `mode_v2` setting should not remain active.
