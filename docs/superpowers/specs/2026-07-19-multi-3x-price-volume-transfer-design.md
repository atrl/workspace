# Multi-3x Price-Volume Transfer Design

## Objective

Test whether daily price-volume pressure factors transfer across long 3x equity
ETFs and add breakdown-risk information beyond the existing price-only high-chop
features. This is a factor experiment, not a model-selection or portfolio-control
experiment.

The decision is whether any factor family is reliable enough to retain for later
SOXL risk-control research. No factor is promoted merely because a pooled metric
looks attractive.

## Research Contract

- Source data are read through `qlib.data.D.features` from the existing paired
  ETF/underlying providers.
- The derived provider is built under `data/qlib/multi_3x_price_volume_transfer_v1`
  and read back through qlib.
- A qlib `DatasetH`/`DataHandlerLP` path exposes the factor and label columns.
- The executed research notebook uses `qlib.init`, qlib Data/Dataset APIs and
  `qlib.workflow.R`; large row-level diagnostics are recorder-only.
- Every model-visible feature is computed per ETF from past observations and
  shifted by one complete session. Asset identity, VIX, and cross-asset state are
  excluded from the core price-volume factor set.
- The primary label is `path_label == breakdown` over the next 20 sessions.
- The primary subset is high-level state (`range_pos_252 >= 0.70`) because the
  question concerns breakdown risk after elevated chop, not unconditional market
  direction.

## Factor Families

All factors are oriented so a higher value means more future breakdown pressure.

1. `effort_result_pressure_20`: volume-weighted absolute daily movement divided
   by the absolute 20-session net result. High values mean large trading effort
   with little progress.
2. `volume_weighted_clv_pressure_20`: negative rolling volume-weighted close
   location value. High values mean volume repeatedly closes near the session low.
3. `failed_breakout_pressure_20`: relative-volume-weighted attempts above the
   prior 20-session high that close back below it and away from the session high.
4. `volume_sign_flip_pressure_20`: relative-volume-weighted return-sign changes.
5. `signed_volume_selling_pressure_20`: negative signed-volume imbalance.
6. `price_volume_divergence_pressure_20`: elevated 20-session price position
   without confirming signed-volume strength.
7. `leverage_residual_pressure_20`: negative `3x return - 3 * underlying return`,
   emphasized on underlying up sessions and high relative volume.
8. `realized_leverage_shortfall_pressure_60`: volume-conditioned shortfall of
   rolling ETF/underlying beta from 3.

Each raw factor is transformed by a trailing 252-session percentile (minimum 60
observations) and then delayed by one session. The existing price-only baseline
uses elevated range position, low trend efficiency, high sign-flip density, and
high-chop intensity.

## Validation Design

### Development evidence

- Exclude SOXL entirely from factor selection.
- Use non-SOXL observations ending in 2022.
- Score each underlying group separately; UPRO and SPXL remain one SPY group.
- Require adequate observations and breakdown events per group; ineligible groups
  are recorded rather than silently pooled.
- Report ROC AUC, average-precision lift over prevalence, top-minus-bottom quintile
  breakdown-rate lift, quintile monotonicity, and sample/event counts.

### Factor retention

A factor is eligible only when:

- at least 70% of eligible underlying groups have AUC above 0.50 and positive
  top-minus-bottom quintile lift;
- median group AUC is at least 0.52;
- median group quintile lift is at least 2 percentage points; and
- it is not more than 0.85 correlated with a better retained factor.

Retained factors form an equal-weight composite. This is a transparent factor
combination, not a fitted predictive model.

### Zero-shot outer test

- SOXL rows from 2023 through the common data end are held out until factor
  definitions and retention are frozen.
- Compare `price_only` and `price_volume` composites overall and by calendar year.
- The volume family adds evidence only if it improves SOXL average-precision lift
  and top-minus-bottom quintile lift, has overall AUC above 0.50, and avoids annual
  sign reversal in at least three eligible years.

## Decision Labels

- `retained_for_risk_control_followup`: at least one nonredundant factor passes
  development gates and the frozen composite adds zero-shot SOXL evidence.
- `not_retained_no_transferable_price_volume_evidence`: no factor passes the
  development transfer gate or the SOXL incremental gate fails.
- `blocked_insufficient_qlib_data`: qlib source/readback, eligible group, or outer
  SOXL sample requirements are not met.

The card is diagnostic. It does not promote a production strategy and does not
claim that a factor improves portfolio returns.

## Durable Artifacts

Card root:

- `multi_3x_price_volume_transfer_v1.ipynb`
- `workflow.yaml`
- `summary.json`
- `conclusion.md`
- `factor_summary.csv`
- `group_metrics.csv`
- `outer_test_metrics.csv`
- `factor_correlation.csv`
- `failure_modes.md`

Recorder-only:

- row-level derived factor panel
- complete development and outer diagnostic rows
- qlib Dataset samples and artifact manifest
