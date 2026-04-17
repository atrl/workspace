# 2026-04-17: Dashboard window clarified, regime-routed overlay is directional but not best

## Dashboard clarification

- `runtime/generalization/dashboard.html` only renders the fixed `test` window:
  - `2020-01-02 .. 2026-03-31`
- The page does **not** show:
  - `train = 2010-2018`
  - `valid = 2019`
- The earlier impression that “2020 少了一年” was a UI explanation problem, not a missing-data problem.
- `scripts/build_backtest_dashboard.py` was updated so the page now states the split explicitly and shows year ticks on the main chart.

## Execution-overlay result

- Added `regime_route_v1` to `configs/soxl/execution_overlay_variants.json`
- Generalization evidence:
  - average vs `fast_reentry`
    - `bear_2022 return: -30.10% -> -29.27%`
    - `bear_2022 maxDD: 34.25% -> 33.49%`
    - `trades: 471 -> 360`
  - but it still loses to `cta_trend_hold` on average proxy:
    - `Sharpe 0.807 < 0.832`
    - `return 268.7% < 291.7%`

## Next optimization direction

- Keep the idea, drop the exact params.
- Next training branch should be a `balanced_regime_gate` style experiment:
  - preserve stress-side de-risking
  - compress turnover moderately
  - avoid copying the current bull multipliers into `SOXL` / `TECL`
- Training selection should penalize branches whose `2022` avg position rises without enough proxy-Sharpe compensation.
