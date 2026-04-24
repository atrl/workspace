# Multi-Asset Knowledge

## Retained Insight

SOXL logic can be tested against TQQQ, UPRO, SPXL, and TECL, but shared 3x ETF training needs generic rewards, volatility normalization, and asset-specific capacity before promotion claims.

## Failed Pattern

A single shared actor plus one-hot asset identity failed with SOXL-tuned rewards: all test slices degraded and the model became globally conservative.

## Evidence Path

- `memory/2026-04-14-multi-asset-joint-training.md`
- `memory/2026-04-22-multi-asset-refined-v3-failure.md`
- `runtime/generalization/`

## Reusable Method

Use cross-ETF diagnostics first. Only train jointly after the data registry confirms per-asset panels, qlib stores, leverage metadata, and volatility-scale normalization inputs.

## Next Valid Experiment Type

Run signal or overlay transfer diagnostics across the registered 3x ETF universe before another shared-actor training run.
