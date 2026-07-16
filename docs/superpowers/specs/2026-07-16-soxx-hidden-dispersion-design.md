# SOXX Hidden Dispersion Feature Design

## Goal

Test whether high SOXX constituent movement combined with a muted VIX response
contains stable information about future SOXX/SOXL risk or direction. Treat the
state primarily as a risk and meta-label regime; do not assume that dispersion
is mechanically bearish.

## Research Surface

Create `research/soxx_hidden_dispersion_v1/soxx_hidden_dispersion_v1.ipynb` as
the executable qlib research card. Reusable feature construction and evaluation
belong in `src/analysis/soxx_hidden_dispersion.py`; the qlib data, recorder, and
card orchestration belong in `src/workflow/soxx_hidden_dispersion.py`.

The card reads SOXX/SOXL/VIX and SOXX constituent OHLCV through
`qlib.data.D.features`. It records the full daily feature panel in the qlib
recorder and keeps only the notebook, workflow configuration, summary,
conclusion, and compact metric sidecars in the card root.

## Feature Definitions

All constituent features use information through the prior close and are lagged
one trading day before open-to-open labels.

1. `residual_dispersion_5`: five-day rolling mean of the equal-weight cross-
   sectional standard deviation of train-beta-adjusted constituent returns.
2. `range_energy_5`: five-day rolling mean of the root mean square constituent
   log high-low range.
3. `cancellation_ratio_5`: five-day rolling mean of
   `1 - abs(mean(return)) / mean(abs(return))`.
4. `correlation_breakdown_20`: one minus the mean pairwise correlation of
   constituent returns over twenty trading days.
5. `breadth_entropy_5`: five-day rolling mean of binary sign entropy, normalized
   to `[0, 1]`.
6. `leadership_turnover_5`: one minus the cross-sectional Spearman correlation
   between current and five-day-lagged constituent return ranks, scaled to
   `[0, 1]`.
7. `abnormal_volume_breadth_5`: five-day rolling mean of the fraction of
   constituents whose lagged 20-day volume z-score exceeds `1.5`.
8. `dispersion_vix_gap`: expanding train-safe z-score of
   `residual_dispersion_5` minus the expanding z-score of absolute five-day VIX
   log change.
9. `weighted_residual_x_dispersion_vix_gap`: the existing weighted residual
   momentum signal multiplied by `dispersion_vix_gap`.

Equal-weight variants are the primary evidence because the local store
backfills current SOXX holdings through history. Weight-based values remain
diagnostic until point-in-time holdings exist.

## Targets And Evaluation

Use one-day-lagged features against future five-day open-to-open SOXX and SOXL
returns and future five-day SOXX absolute return. Fit feature orientation on the
train segment only. Report, by train/validation/test:

- Spearman rank correlation;
- directional AUC for signed returns;
- top-minus-bottom quartile spread;
- five non-overlapping horizon offsets;
- risk-target rank correlation for future absolute return;
- incremental results versus the existing `weighted_residual` baseline.

The main model comparison is deliberately small:

1. `weighted_residual` baseline;
2. hidden-dispersion feature set;
3. baseline plus hidden-dispersion interaction.

## Data And Failure Boundaries

- Fail closed if either qlib provider, required feature, VIX series, or recorder
  is unavailable.
- Do not fall back to pandas-only source files.
- Require at least two usable constituents per daily cross-section and at least
  three observations for correlation metrics.
- Keep beta estimates train-only and every market feature lagged one trading
  day.
- Record provider URIs, calendar range, segments, recorder id, source fields,
  and lag policy in `summary.json`.

## Artifacts

Card-root artifacts:

- `soxx_hidden_dispersion_v1.ipynb`
- `workflow.yaml`
- `summary.json`
- `conclusion.md`
- `feature_metrics.csv`
- `nonoverlap_feature_metrics.csv`
- `model_comparison.csv`
- `risk_analysis.csv`

Recorder-only artifacts:

- `daily_hidden_dispersion_panel.csv`
- any prediction, signal, portfolio, or position detail

## Promotion Boundary

This card is diagnostic and must remain
`not_promoted_missing_pit_holdings` while the constituent universe and weights
are current-holdings backfills. A candidate can be considered for a later
validated card only if its validation and test signs are stable, its five
non-overlapping offsets are not dominated by one offset, and it improves on the
`weighted_residual` baseline without relying on the current-weight version.
