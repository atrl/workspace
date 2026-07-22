# Cash-Floor Benchmark-Relative Walk-Forward Requirements

**Status:** accepted research requirements; implementation and promotion evidence are not yet complete.

## Objective

Build a long-only `SOXL + SOXX + cash` strategy whose pooled out-of-sample
walk-forward wealth exceeds buy-and-hold SOXX after costs while maintaining a
real cash reserve throughout the path.

The model is a risk-budget overlay on a SOXX-anchored risky sleeve. It is not a
free three-way asset selector. SOXS may supply observable features, but it is
not a tradable portfolio asset in this experiment.

## Why These Requirements Replace The Previous Gate Stack

The objective feasibility audit found that the previous combination of
`MaxDD >= -20%` and `SOXL upside capture >= 60%` turned the task into a crash
timing problem. No tested mechanical strategy passed all gates. A non-tradable
hindsight oracle proved that a mathematical feasible region exists, but the
existing leverage-advantage targets were not stable across development years.

A directional 2016-2022 annual walk-forward diagnostic using the existing
features found:

- With a 10% daily cash floor, up to 50% additional model-controlled cash, and
  a fixed 30% SOXL / 70% SOXX risky sleeve, the pooled 10bp terminal wealth
  ratio versus SOXX was `1.3088`, average cash was `20.68%`, and five of seven
  calendar folds beat SOXX.
- The same-average-cash static allocation reached only `1.1575`, so the
  existing features contained some useful cash-timing information.
- The dynamic path still had `MaxDD -58.48%` and worst rolling five-day return
  `-28.35%`, slightly worse than the matched static path on those tail metrics.
  A return-only cash head is therefore insufficient.
- Raising the risky-sleeve SOXL share to 60% lifted capture to about 60% and
  the wealth ratio to `1.7407`, but worsened MaxDD to `-72.40%`. That setting is
  excluded from v1 rather than selected for its return.

These diagnostics are architecture evidence only. They are not a promoted
backtest and must not be used as sealed-forward evidence.

## Portfolio Contract

For every trading day:

```text
cash_weight = c_t
soxl_weight = (1 - c_t) * 0.30
soxx_weight = (1 - c_t) * 0.70
```

Hard constraints:

- `0.10 <= c_t <= 0.60` every day.
- Average walk-forward cash weight must be at least `0.15`.
- SOXL, SOXX, and cash weights must be non-negative and sum to one.
- Total SOXL weight is capped at `0.27` by construction.
- Borrowing, shorting, and simultaneous SOXL/SOXS exposure are prohibited.
- SOXS has zero tradable weight; it may only contribute features.
- Signal timing is close of T; target weights apply from T+1 open to T+2 open.
- Cash return is conservatively zero in v1. A PIT T-bill series is a separate
  future extension and cannot be backfilled with a current yield.
- Primary transaction cost is 10bp of one-way turnover; 25bp is mandatory
  stress testing.

The 30% risky-sleeve SOXL share is fixed for v1. It is not selected from the
walk-forward test folds. A trainable leverage head remains blocked until a
leverage-advantage target passes the predictability audit.

## Model Contract

Use the existing causal feature family:

- `soxx_trend20`
- `soxx_trend60`
- `soxx_vol20`
- `soxx_drawdown60`
- `relative_momentum20`
- `soxs_trend5`
- `soxs_trend20`
- `inverse_tracking_gap1`
- `inverse_tracking_gap20`
- `soxl_soxs_path20`

Train two fixed, regularized heads:

1. A return-risk head for `cash_advantage_5d`.
2. A volatility head for `future_vol_5d`.

Both heads must use qlib `DatasetH` / `DataHandlerLP`. The initial model family
is fixed Ridge with training-only scaling and regularization; no outer-period
model or threshold selection is permitted.

Convert both predictions to training-only percentile risk scores. The cash
budget uses the more conservative of the two scores:

```text
risk_score_t = max(cash_advantage_percentile_t, future_vol_percentile_t)
c_t = 0.10 + 0.50 * clip(risk_score_t, 0, 1)
```

All percentile cut points are fitted inside each training fold. Test-fold or
outer-period distributions cannot calibrate the mapping.

## Walk-Forward Contract

- Development source window starts in 2011.
- Annual test folds are 2016 through 2022.
- Training expands through the prior year.
- Remove at least 20 trading rows between each training endpoint and test fold
  to cover the longest v1 target horizon.
- Refit both heads once per annual fold.
- Concatenate only test-fold returns to form pooled walk-forward evidence.
- Hyperparameters and the portfolio formula remain identical across folds.
- 2023-2026 is previously observed outer data. It is diagnostic only and must
  have `selection_uses_outer_test: false`.
- Even a development pass cannot promote directly to production. Promotion
  requires a new, time-forward sealed paper window.

## Primary Research Gates

All gates below must pass on the 2016-2022 pooled development walk-forward.

### Relative wealth

- 10bp terminal wealth ratio versus SOXX must be at least `1.10`.
- 25bp terminal wealth ratio versus SOXX must be at least `1.05`.
- At least five of seven annual folds must have wealth ratio above `1.00`.
- Median annual-fold wealth ratio must be above `1.00`.
- Stationary-block-bootstrap outperformance probability versus SOXX must be at
  least `60%`.

### Cash participation

- Minimum daily cash weight must be at least `10%`.
- Average pooled walk-forward cash weight must be at least `15%`.
- Zero-cash day count must be zero.

### Timing value

- The dynamic strategy's 10bp terminal wealth must exceed a static portfolio
  with the same average SOXL, SOXX, and cash weights.
- This matched-static comparison uses the same dates, execution timing, and
  transaction-cost convention.

### Risk usefulness

- Model MaxDD may not be more than 10 percentage points worse than SOXX MaxDD.
- Model worst rolling five-day return may not be more than 5 percentage points
  worse than SOXX.
- Relative to the same-average-weight static portfolio, model MaxDD and worst
  rolling five-day return may each be no more than 2 percentage points worse.
- Relative to the no-cash fixed 30% SOXL / 70% SOXX risky sleeve, the model
  must improve MaxDD by at least 5 percentage points and worst rolling five-day
  return by at least 3 percentage points.

These are relative risk gates, not promises of a `-20%` absolute maximum
drawdown. MaxDD, worst day, CVaR5, annualized volatility, and return/MaxDD must
still be reported for every fold and benchmark.

## Removed Hard Gates

The following are no longer promotion requirements for this experiment:

- Absolute `MaxDD >= -20%`.
- Absolute worst day `>= -12%`.
- Absolute `CVaR5 >= -8%`.
- Absolute worst rolling five-day return `>= -12%`.
- SOXL upside capture `>= 60%`.
- Return/MaxDD exceeding a same-average-position SOXL portfolio.
- Three of four previously observed outer years passing a selection gate.

They remain report-only diagnostics so the new result can be compared honestly
with prior cards.

## Benchmarks

Every walk-forward fold and pooled report must include:

1. Buy-and-hold SOXX — primary return benchmark.
2. Same-average-weight static SOXL/SOXX/cash — timing-value benchmark.
3. Fixed no-cash 30% SOXL / 70% SOXX — cash-risk-usefulness benchmark.
4. Cash-only — inactivity reference, never an eligible winner.

## Decision Labels

- `research_candidate`: all development gates pass; requires sealed forward.
- `needs_iteration_relative_wealth`: pooled or fold stability gates fail.
- `needs_iteration_cash_timing`: timing-value or risk-usefulness gates fail.
- `blocked_inactive`: cash/participation constraints produce an effectively
  inactive portfolio.
- `blocked_outer_contamination`: any outer result affects training, mapping,
  model choice, or thresholds.
- `promoted`: unavailable in this research card; reserved for a later sealed
  paper-validation card.

## Required Qlib Evidence

The implementation must use the repository qlib provider, `DatasetH`,
`DataHandlerLP`, qlib workflow recorder, and qlib-backed portfolio/risk output.
The notebook is the primary executable surface. Predictions, positions,
portfolio reports, and bootstrap paths are recorder-only artifacts. Root-card
sidecars remain small and include fold metrics, benchmark comparisons, risk
analysis, model metadata, summary, workflow configuration, and the final
Chinese conclusion.
