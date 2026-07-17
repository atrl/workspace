# SOXL Risk-Adjusted Model Frontier Design

## Goal

Find a SOXL allocation model that reduces drawdown without passing by staying
in cash. Compare three materially different model families under one qlib data,
label, execution, cost, walk-forward, and promotion contract.

The primary objective is risk-adjusted performance. A candidate may earn less
than legacy `soxlbacktest`, but it must preserve meaningful participation and
must improve return-to-drawdown rather than merely suppressing exposure.

## Current Failure To Address

The current position-model card fails closed because its selected candidate is
effectively inactive. The model search has already covered linear, ridge,
logistic, random-forest, extra-trees, histogram-gradient-boosting, LightGBM,
XGBoost, and simple LGB/XGB consensus variants. Repeating those families with
more thresholds is not a new experiment.

The new frontier therefore changes both the prediction target and the position
mapping:

- predict expected return and downside risk separately;
- use continuous soft risk budgets instead of binary all-cash gates;
- compare specialist experts with soft regime weights;
- test compact sequence models on lagged multivariate windows;
- select on validation and nested walk-forward only, never on test.

## Shared Data And Timing Contract

All three routes use the canonical qlib-managed SOXL store and the same daily
decision timing:

- information set: values observable at `close_t`;
- signal availability: after `close_t`;
- entry: next session open;
- holding return: next-open to following-open daily return;
- primary horizon: five trading days;
- all rolling inputs lagged by at least one daily bar;
- final `horizon + 1` rows purged from every train, validation, test, and
  walk-forward segment.

The initial common feature surface is deliberately compact:

- SOXL and SOXX lagged returns, trend, realized volatility, range, volume, and
  drawdown;
- VIX level, change, percentile, and SOXL-realized-volatility/VIX gap;
- SOXL-versus-SOXX tracking residual and daily-reset leverage diagnostics;
- lagged hidden-dispersion features when available in the active branch;
- calendar and session-state features already accepted by the repo contracts.

Features are standardized or oriented using training data only. Missing live
or historical sources fail closed and are recorded in the card conclusion.

## Shared Labels

Each route receives the same three labels:

1. `future_return_5d`: compounded SOXL next-open daily return over five days.
2. `future_downside_5d`: negative part of `future_return_5d`.
3. `future_crash_5d`: one when either the five-day return is at or below the
   training tenth percentile or any daily next-open return in the horizon is at
   or below -8 percent.

The tenth-percentile crash threshold is fitted on the train segment and frozen
for validation and test. Test labels never influence target construction,
direction, calibration, thresholds, ensemble weights, or model choice.

## Route 1: Tail-First Soft Budget

The tail-first route separates opportunity from danger:

- expected-return models: ridge and LightGBM regression;
- downside models: CatBoost quantile regression and qlib
  `DEnsembleModel` on downside magnitude;
- crash model: calibrated CatBoost classification;
- ensemble weights: chosen on validation utility only.

The normalized expected-return score `mu_score` and downside score
`tail_score` map to position through:

```text
raw_budget = sigmoid(mu_score)
risk_multiplier = clip(1 - crash_probability - 0.5 * tail_score, 0, 1)
target_position = floor + (cap - floor) * raw_budget * risk_multiplier
```

Default research bounds are `floor = 0.15` and `cap = 0.75`. The floor applies
only when the data-quality gate is valid. Missing or stale required data sets
the position to zero and records a fail-closed reason.

The route compares CatBoost, DEnsemble, and their validation-selected blend; it
does not promote a blend merely because it has the best test result.

## Route 2: Regime Mixture Of Experts

The mixture route contains three bounded experts:

- trend expert: ridge on trend, breadth, and cross-asset momentum;
- reversal expert: CatBoost on oversold, residual, range, and volatility-gap
  features;
- stress expert: downside-aware model that reduces risk as crash probability
  rises.

A multinomial gate estimates soft probabilities for trend, reversal, and
stress regimes from lagged inputs. Regime labels are fitted from train-only
return, trend, and volatility thresholds. The final signal is a probability
weighted mixture, not a hard switch:

```text
expected_score = sum(regime_probability * expert_score)
target_position = clip(0.15 + 0.60 * sigmoid(expected_score), 0.15, 0.75)
target_position *= soft_stress_multiplier
```

The gate must report regime occupancy. Any regime with fewer than 40 train
observations blocks the route instead of fitting an unstable expert.

## Route 3: Multivariate Sequence Models

The sequence route compares compact qlib-compatible TCN, GRU/ALSTM, and
Transformer models on the same lagged SOXL/SOXX/VIX feature windows. The
initial window is 20 sessions, with a single validation-selected alternative
of 60 sessions. Network size is intentionally small:

- at most two temporal layers;
- hidden width at most 32;
- dropout at least 0.10;
- early stopping on validation loss;
- fixed random seeds reported in the card;
- no test-driven architecture or epoch selection.

The current SOXL/SOXX/VIX history is sufficient for a diagnostic multivariate
time-series experiment but not for a large Transformer. A second
cross-sectional constituent mode is allowed only when point-in-time holdings
and aligned features pass the data gate. Without PIT holdings, that mode is
reported as `blocked_missing_pit_panel`; current constituents may not be
backfilled through history.

Sequence predictions use the same continuous soft-budget mapper as Route 1 so
model comparisons do not confound model family with position policy.

## Risk-Adjusted Promotion Contract

Every candidate must pass all of the following on the untouched test segment
and on the walk-forward aggregate:

- maximum drawdown no worse than -20 percent;
- average absolute position at least 15 percent;
- zero-position-day ratio no more than 70 percent;
- at least two of three walk-forward test folds have positive return;
- return-to-drawdown at least 1.25 times the matched legacy reference;
- absolute return captures at least 50 percent of the matched legacy return;
- CVaR 5 percent, worst-five-day mean loss, and extreme-loss-day count are not
  worse than the legacy reference;
- total turnover does not exceed the existing repo risk limit;
- no label leakage, test-time selection, incomplete final label, or missing
  required data.

If no route passes, the frontier decision is
`not_promoted_no_risk_adjusted_candidate`. A route blocked by data quality is
reported separately and cannot win by omission.

## Walk-Forward And Selection

Use an expanding-window nested walk-forward with three outer test folds. Each
outer training window contains an inner validation tail used for:

- model hyperparameter choice;
- probability or score calibration;
- ensemble weights;
- position floor/cap choice from the fixed set `(0.15, 0.60)`,
  `(0.15, 0.75)`, and `(0.20, 0.75)`;
- sequence window choice between 20 and 60.

The untouched final test segment is evaluated once after the candidate is
frozen. Overlapping five-day labels are supplemented by five non-overlapping
offset reports.

## Qlib And Artifact Architecture

Create four notebook-first, recorder-first slim cards:

- `research/soxl_tail_budget_v1/`
- `research/soxl_regime_moe_v1/`
- `research/soxl_panel_sequence_v1/`
- `research/soxl_risk_adjusted_model_frontier_v1/`

The first three cards train and evaluate their own route. The frontier card
does not retrain or select on test; it reads frozen card summaries and recorder
artifacts, checks matched data windows, and ranks candidates by the shared
promotion contract.

Each route uses `qlib.init`, `DatasetH`, `DataHandlerLP`, `R.start`, and qlib
analysis/backtest outputs. Large predictions, daily signals, positions,
portfolio reports, and all-model matrices remain recorder-only. Root card files
are limited to the notebook, workflow, summary, conclusion, selected-model
metadata, and compact comparison/risk/window tables.

## Error Handling

- stale canonical data: `blocked_stale_data`;
- unavailable required feature source: `blocked_missing_required_source`;
- incomplete labels at a segment boundary: purge and report excluded rows;
- too-small regime: `blocked_sparse_regime`;
- missing PIT constituent panel: `blocked_missing_pit_panel` for the
  cross-sectional mode only;
- non-finite prediction or position: fail the candidate and record the first
  offending date;
- failed model fit: retain the failure row in model comparison and continue
  other candidates;
- missing recorder artifact: block the frontier comparison rather than reading
  an ad hoc root export.

## Verification

Tests must cover:

- train-only crash thresholds and feature scaling;
- segment-boundary purging and future-perturbation leakage checks;
- floor/cap behavior and fail-closed data gates;
- regime occupancy and soft mixture weights summing to one;
- sequence windows containing only information through `close_t`;
- validation-only model and ensemble selection;
- participation, drawdown, CVaR, worst-five-day, turnover, and fold gates;
- slim-card artifact policy and recorder-only large outputs;
- frontier rejection when data windows or recorder evidence do not match.

The notebooks must execute in the repo qlib environment. Focused tests, qlib
contract tests, applicable broad tests, staged-diff inspection, commit, and
push are required before delivery.
