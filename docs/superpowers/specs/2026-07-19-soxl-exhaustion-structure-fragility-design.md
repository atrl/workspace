# SOXL Exhaustion, Structure, and Fragility Factor Design

## Decision

Test three transparent factor families without fitting a predictive model:

1. price-path exhaustion on a broad long-3x ETF development universe;
2. SOXL-versus-underlying leverage fragility on the same transfer universe;
3. a fixed NVDA/AVGO/AMD/TSM/ASML semiconductor bellwether proxy on pre-2023 SOXL.

The frozen outer window is 2023-01-01 through 2026-07-02 SOXL. Higher factor
values always mean more future breakdown pressure. Every feature is delayed by
one complete session and normalized only with that asset's or proxy's past.

## Data and limitations

- Daily ETF/underlying OHLCV comes from the existing per-pair qlib providers.
- The fixed bellwether proxy comes from `data/qlib/soxx_constituents`. It has
  2010-2026 price coverage, but it is not point-in-time membership and must not
  be described as historical SOXX constituents.
- SOXL/SOXX 15-minute data covers 2018-12-31 onward and is usable for a separate
  two-instrument daily-safe diagnostic. External bellwether 15-minute history is
  only complete from 2025-01-02, so it is blocked from long-window factor
  selection and promotion evidence.

## Factors

Price-path exhaustion measures failed-breakout persistence, time already spent
near the trailing high, collapsing path efficiency, lack of upside progress per
unit path length, downside range asymmetry, and deterioration in close location.

Leverage fragility measures cumulative SOXL-minus-3x-underlying residual,
up-day realized leverage shortfall, down/up beta asymmetry, volatility drag, and
relative-strength breakdown.

Fixed bellwether structure measures residual dispersion, leadership
concentration, breadth divergence, new-high participation decay, and recent
downside breadth. The five-name list is frozen before evaluation.

## Validation

- Development direction and redundancy selection use no 2023+ SOXL rows.
- The label remains the existing 20-session volatility-scaled breakdown and the
  evaluated state remains trailing-range position at or above 0.70.
- Report AUC, average-precision lift, Q5-Q1 event-rate lift, and a transparent
  conditional lift inside price-only quintiles.
- Equal-weight family composites are permitted only from factors passing their
  development gates. No fitted weights are allowed.
- Retain a family only when the frozen SOXL result improves the price-only
  baseline and has positive conditional lift. Otherwise record a failed gate.
- Years with fewer than five events are diagnostic only.

## Artifact contract

The executable notebook and small summaries live in
`research/soxl_exhaustion_structure_fragility_v1/`. Row-level development and
outer score matrices are recorder-only. The card records explicit `blocked`
status for the external-bellwether intraday family.
