# Multi-3x High-Chop Risk Control V1 Design

## Status

- Design direction approved in chat on 2026-07-19.
- This document is the implementation boundary for the first research version.
- The strategy remains research-only until every promotion gate in this document passes.

## Objective

Build a qlib-native research card that reduces SOXL exposure during confirmed
downside states, lowers portfolio volatility during high-level choppy states,
and preserves material participation in upside breakouts.

The strategy is a transparent risk-control layer. The model estimates future
path-state probabilities; deterministic portfolio rules convert those
probabilities into long-only SOXL, SOXX, and cash weights. The model does not
directly emit an unconstrained target position.

## Fixed Scope

- Instruments: SOXL, SOXX, and cash for the evaluated portfolio.
- Research universe: the expanded normalized 3x ETF panel and each ETF's
  non-leveraged underlying proxy.
- Frequency: daily signal at close `t`, target applied from next open `t+1`.
- Direction: long-only; no shorting, inverse ETFs, options, or broker execution.
- Card: `research/multi_3x_high_chop_risk_control_v1/`.
- Reusable logic: `src/` modules; experiment choices and the Chinese conclusion
  remain in the notebook.

## Non-Goals

- Do not optimize a model to maximize SOXL total return without risk controls.
- Do not reuse reinforcement-learning rewards.
- Do not promote current-holdings-backfilled SOXX constituent history as
  point-in-time evidence.
- Do not add production scheduling or live order behavior.
- Do not use close-to-close returns as executable evidence for a close-generated
  signal.

## Evidence Behind The Design

The existing high-chop definition identifies many historical states, but their
future returns are bimodal: some become sharp breakdowns and some become strong
breakouts. High-chop or constituent dispersion alone is therefore a risk-state
feature rather than a directional sell signal.

The existing high-chop whole-position models are unstable across validation and
test. The strongest selected path still uses a base rule for most exposure and
uses the model only as a throttle. Existing repository primitives already
support volatility scaling, constrained position changes, and prior-equity
drawdown budgets. V1 should compose and harden those primitives instead of
adding another opaque full-position predictor.

## Qlib-Native Contract

The implementation must follow the official Qlib data, workflow, recorder, and
portfolio strategy behavior:

- Data: https://qlib.readthedocs.io/en/latest/component/data.html
- Workflow: https://qlib.readthedocs.io/en/latest/component/workflow.html
- Recorder: https://qlib.readthedocs.io/en/latest/component/recorder.html
- Portfolio strategy and backtest: https://qlib.readthedocs.io/en/latest/component/strategy.html

The notebook must explicitly call `qlib.init`, read the aligned panel through
`qlib.data.D` and a qlib Dataset/DataHandler contract, train and score inside a
qlib workflow, and log predictions, positions, portfolio reports, and risk
analysis to the qlib recorder. A pandas-only fallback is forbidden.

## Data Design

### Expanded 3x ETF universe

The initial expanded universe is fixed as follows:

| Cohort | 3x ETF | Underlying proxy |
|---|---|---|
| Semiconductor | SOXL | SOXX |
| Nasdaq/technology | TQQQ | QQQ |
| Nasdaq/technology | TECL | XLK |
| Broad US | UPRO | SPY |
| Broad US | SPXL | SPY |
| Dow | UDOW | DIA |
| Small cap | TNA | IWM |
| Mid cap | MIDU | MDY |
| Financials | FAS | XLF |
| Biotechnology | LABU | XBI |
| Healthcare | CURE | XLV |
| Homebuilders | NAIL | ITB |
| Retail | RETL | XRT |
| Energy | ERX | XLE |
| Real estate | DRN | XLRE |
| Consumer discretionary | WANT | XLY |

Inverse products and non-equity 3x products are excluded because their path
mechanics and state labels are not comparable to long equity 3x ETFs.

### Data lineage

Raw market data may only enter through the existing `src/data` update and qlib
store builder. The expanded provider manifest must record symbol pairs,
provider URI, as-of date, calendar, fields, row counts, hashes, and qlib readback
audit results. The research notebook reads only the resulting qlib store.

### Normalized features

Every feature must be lagged and normalized within asset using only prior
observations. Asset identity is not a model feature.

Feature families:

1. Price path: 5/20/60-day return, 252-day range position, distance from high,
   drawdown, trend efficiency, sign-change density, and path length.
2. Volatility: 5/20/60-day realized volatility, range expansion, downside
   semivariance, and volatility-of-volatility.
3. Leveraged residual: `r_3x - 3 * r_underlying`, residual volatility,
   tracking-error expansion, and upside/downside capture asymmetry.
4. Quiet-volatility gap: ETF realized-volatility percentile minus VIX
   percentile, range percentile minus VIX percentile, and absolute VIX change.
5. Cross-asset state: same-day normalized 3x ETF dispersion, positive breadth,
   downside breadth, and cohort leadership turnover.
6. SOXL-only diagnostics: lagged constituent breadth and dispersion. These are
   excluded from the transferable model and reported separately because the
   current historical constituent universe is not point-in-time.

## State And Label Design

### Current-state intensity

`high_chop_intensity` is a continuous value in `[0, 1]` based on near-high
position, low trend efficiency, elevated realized volatility, and a quiet VIX.
It controls leverage reduction but never forces a zero position by itself.

### Future path labels

For each asset-date, calculate the next 20-session terminal return, maximum
favorable excursion, and maximum adverse excursion. Normalize all three by the
20-day ex-ante volatility scale, where `rv_20` is annualized:

```text
sigma_20 = rv_20(t) * sqrt(20 / 252)
```

Labels are mutually exclusive and assigned in this order:

1. `whipsaw`: favorable excursion is at least `+0.75 * sigma_20` and adverse
   excursion is at most `-0.75 * sigma_20`.
2. `breakdown`: adverse excursion is at most `-1.00 * sigma_20` and terminal
   return is at most `-0.50 * sigma_20`.
3. `breakout`: favorable excursion is at least `+1.00 * sigma_20` and adverse
   excursion stays above `-0.75 * sigma_20`.
4. `benign_chop`: every remaining path.

The final 21 rows of every train, validation, and test segment must be purged so
that the 20-session path label cannot cross a segment boundary.

## Model Design

V1 compares three candidates:

1. Transparent threshold/state baseline without learned path probabilities.
2. Multinomial logistic regression as the calibrated linear baseline.
3. LightGBM multiclass classifier as the nonlinear candidate.

All candidates use the same qlib Dataset and segments. Probabilities must sum to
one and be calibrated using validation-only data. If a class has fewer than 50
training observations or 20 validation observations in a fold, that fold is
`blocked_insufficient_class_samples` and cannot select a model.

Model selection uses validation log loss, breakdown Brier score, breakdown tail
recall, and calibration error. Portfolio return is a secondary tiebreaker and
outer-test performance is never used for selection.

## Validation Design

Use expanding walk-forward outer test years 2023, 2024, 2025, and 2026. Within
each outer fold:

- purge label horizons at all boundaries;
- group adjacent flagged days into one episode so an episode cannot be split;
- perform inner leave-one-underlying-out validation;
- treat UPRO/SPXL as one SPY underlying group;
- report full SOXL holdout results separately from pooled multi-asset results;
- do not include asset identity in training features.

The final strategy conclusion is based on stitched out-of-sample SOXL targets,
not in-sample or oracle-selected predictions.

## Portfolio Controller

### Continuous base mapping

For each day:

```text
vol_scale   = clip(target_vol / soxl_rv_20, 0.15, 1.00)
chop_scale  = 1.00 - 0.50 * high_chop_intensity
down_scale  = (1.00 - p_breakdown) ** 2
raw_soxl    = base_position * vol_scale * chop_scale * down_scale
```

`target_vol` candidates are fixed at 25%, 35%, and 45% annualized. The selected
value must come from validation only. `base_position` is evaluated for both
constant one and the existing `base_constrained_ref`; neither is silently
replaced.

### SOXX substitution

Capital removed from SOXL may be partially reassigned to SOXX only when all of
the following are true:

- `p_breakout > p_breakdown`;
- SOXX is above its 20-day moving average;
- lagged cross-asset breadth is not in its bottom 30% historical percentile.

The SOXX replacement fractions tested are 25%, 50%, and 75% of unused capital,
capped at a 50% SOXX portfolio weight. Otherwise unused capital remains cash.
Portfolio weights must always satisfy:

```text
0 <= w_soxl <= 1
0 <= w_soxx <= 1
w_soxl + w_soxx <= 1
w_cash = 1 - w_soxl - w_soxx
```

### Hard downside gate

Three confirmations are evaluated using information available at close `t`:

1. calibrated `p_breakdown >= 0.65`;
2. SOXX is below its 10-day or 20-day moving average;
3. lagged cross-asset downside breadth is in its worst 30% historical
   percentile.

The model probability confirmation is mandatory. If confirmation 1 and either
confirmation 2 or 3 are active, cap SOXL at 15% and SOXX at 25%. If all three
are active, set SOXL to zero and cap SOXX at 10%. Price and breadth alone, or
high-chop intensity alone, cannot activate the hard downside gate.

### Drawdown budget

The controller uses only previously realized strategy equity:

- activate when prior drawdown is at or below -8%;
- cap combined SOXL and SOXX weight at 25%;
- require prior drawdown to recover above -4%;
- keep the cap for a five-session cooldown before release.

### Turnover and hysteresis

- Risk reduction may occur immediately.
- Risk increases are limited to 20 percentage points per session.
- Ignore ordinary target changes smaller than 10 percentage points.
- Hold ordinary state changes for at least three sessions.
- After a hard downside gate, require `p_breakdown < 0.40` for three consecutive
  sessions, SOXX above its 10-day moving average, and breadth recovery before
  adding SOXL.
- During recovery, add SOXX before SOXL.

This asymmetric behavior must be implemented explicitly; the generic symmetric
`max_daily_change` behavior is not sufficient for V1.

## Execution And Cost Assumptions

- Signal uses data available at close `t`.
- Target begins at next open `t+1` and is evaluated with open-to-open returns.
- Base turnover cost: 3 basis points per unit of one-way turnover.
- Stress costs: 10 and 25 basis points.
- A close-to-close diagnostic may be reported but cannot support promotion.
- Missing next-open data makes the affected row `needs_execution_data`; it is
  not silently replaced with close-to-close return.

## Benchmarks

Every report window must compare against:

1. SOXL buy-and-hold.
2. SOXX buy-and-hold.
3. Constant SOXL at the strategy's average SOXL exposure.
4. Static SOXL/SOXX/cash weights matched to average underlying beta.
5. Existing `base_constrained_ref`.
6. Existing selected high-chop base-throttle candidate.
7. Rule-only risk controller using the same portfolio constraints.

## Metrics And Promotion Gates

Primary metrics:

- annualized volatility;
- maximum drawdown;
- daily 5% CVaR;
- worst five-session return;
- underwater-day ratio;
- high-chop and post-high-chop risk metrics;
- breakdown tail recall and false-positive rate;
- turnover and next-day reversal count.

A validated candidate is promotable only if stitched out-of-sample SOXL results
meet all gates:

1. Maximum drawdown is no worse than -20% and improves at least 20% relative to
   `base_constrained_ref` on the same window.
2. Annualized volatility is no greater than 35%.
3. Daily 5% CVaR is no worse than -8%, and the worst five-session return is no
   worse than -12%.
4. Positive-regime upside capture versus SOXL is at least 60%.
5. Average combined risky weight is at least 15%; an all-cash strategy fails.
6. Return-to-drawdown exceeds constant-exposure SOXL, matched-beta static
   allocation, and SOXX buy-and-hold.
7. At least three of four outer years pass the drawdown and volatility gates;
   no year may exceed -25% maximum drawdown.
8. Results remain valid at 10 basis points of turnover cost. The 25-basis-point
   result is reported as a stress test, not a mandatory pass gate.

If no candidate passes every gate, the card decision is
`not_promoted_risk_control_gates_failed` and the best diagnostic candidate is
still recorded.

## Components And Boundaries

### `src/data`

- Extend the reusable 3x ETF pair registry and qlib provider builder.
- Audit every pair through `qlib.init`, calendar, instruments, and feature
  readback.
- Do not embed strategy parameters in the data layer.

### `src/factor` or `src/workflow` feature module

- Build lagged normalized multi-asset features and path labels from qlib-loaded
  frames.
- Keep current-constituent diagnostics isolated from transferable features.

### `src/model`

- Wrap qlib Dataset training, validation calibration, prediction, and model
  persistence for logistic and LightGBM candidates.

### `src/backtest`

- Add a portfolio-weight controller for SOXL/SOXX/cash.
- Reuse volatility scaling and prior-equity drawdown budget behavior.
- Add asymmetric risk-down/risk-up pacing and two-instrument transaction costs.

### `src/analysis`

- Read recorder-backed predictions, positions, reports, and risk analysis.
- Produce stable benchmark, regime, yearly, stress-window, calibration, and
  failure-mode summaries.

### Notebook card

- Execute the entire qlib path.
- Explain the selected model and controller parameters.
- End with a Chinese conclusion covering what ran, metrics, regime differences,
  promotion decision, risks, and next action.

## Artifact Contract

Card root keeps only:

- `multi_3x_high_chop_risk_control_v1.ipynb`
- `workflow.yaml`
- `summary.json`
- `conclusion.md`
- `selected_model.json`
- `model_selection.csv`
- `backtest_benchmark_comparison.csv`
- `risk_analysis.csv`
- `regime_metrics.csv`
- `window_metrics.csv`
- `calibration_metrics.csv`
- `failure_modes.md`

Recorder-only artifacts include full predictions, probabilities, labels, daily
weights, portfolio report, positions, all-model prediction matrices, and qlib
risk-analysis objects. `summary.json.artifact_policy.policy` must be
`slim_card_recorder_first`.

## Failure Handling

- Missing or stale qlib provider: `blocked_qlib_data_unavailable`.
- Source data beyond declared as-of: fail closed.
- Recorder failure: `blocked_qlib_recorder_unavailable`.
- Insufficient class samples: block the affected fold and disallow selection.
- Invalid or uncalibrated probability vector: block the candidate; do not
  substitute arbitrary rule probabilities.
- Missing executable next-open return: `needs_execution_data`.
- Current-constituent historical feature used in a transferable model:
  `blocked_pit_constituents`.
- No promotion-gate passer: preserve recorder evidence and conclude not
  promoted rather than selecting an oracle test winner.

## Verification

Focused tests must cover:

1. Path labels use only future rows and segment purging removes boundary
   leakage.
2. Rolling normalization uses prior observations only.
3. Leave-one-underlying-out grouping keeps UPRO/SPXL together.
4. Probability vectors are calibrated, finite, and sum to one.
5. Portfolio weights are nonnegative and sum to one including cash.
6. Probability-plus-one and three-confirmation hard gates apply the documented
   caps; price and breadth without probability cannot activate a hard gate.
7. Risk reductions bypass the risk-up pacing limit; risk increases respect the
   20-point limit and recovery hysteresis.
8. Drawdown budget reads prior equity only.
9. Costs are applied independently to SOXL and SOXX turnover.
10. Missing qlib, recorder, class, and execution data paths fail closed.
11. Notebook/card roots satisfy the slim-card recorder-first contract.

After focused tests, execute the notebook in the repo qlib environment and run
the full pytest suite. Before commit, inspect `git status --short` and the staged
diff so generated recorder artifacts and unrelated files are excluded.

## Acceptance Boundary

Implementation is complete only when the executed notebook, compact card
sidecars, and qlib recorder agree on the selected candidate and promotion
decision; all focused and full tests pass; and the branch is committed and
pushed. A completed research card may still correctly conclude that no strategy
is promotable.
