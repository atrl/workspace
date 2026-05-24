# SOXX Spread Oscillation Factor Review

Date: 2026-05-24
Direction: factor

## Goal

Test whether SOXX/SOXL oscillation and SOXL/SOXX spread widening can become
trainable factors for the SOXL model.

## Governance

- Baseline: current core feature schema plus retained SOXX knowledge.
- Direct comparison: prior `soxx_signals` and `SIGNED_TE_REVERSION_*` reviews.
- Role: diagnose before any feature-schema retrain.
- Changed layer: feature.
- Target window: full available SOXL/SOXX daily panel, IC horizons `1/5/10/20`.
- External reference: SOXX underlying ETF dataset plus relative-value /
  tracking-error factor family already used by `SIGNED_TE_REVERSION`.
- Hypothesis: pair spread widening or relative choppiness may capture a
  semiconductor LETF dislocation not already represented by core momentum,
  volatility, and tracking-error features.
- Acceptance criteria: mean IC `> 0.03`, IC positive ratio `> 0.55`, profit
  factor `> 1.2`, signal accuracy `> 0.55`, and max core correlation `<= 0.80`.
- Expected trade-off: ratio/spread factors may be redundant with existing trend,
  momentum, and volatility features; oscillation divergence should be more
  orthogonal but may be weak.
- Failure mode: high standalone Sharpe from near-always-long exposure or high
  correlation with existing core features.
- Comparison-table row key: `SOXX_SPREAD_OSCILLATION`.

## Historical Context

- Closest success: `REL_VOLUME_RATIO` had h20 IC `+0.058` and low correlation,
  but failed standalone PF/accuracy.
- Closest failure: `SIGNED_TE_REVERSION_*` and direct SOXX actor retrain failed
  promotion gates or walk-forward checks.
- This run is not a duplicate because it tests pair spread level/widening,
  spread-volatility, and relative choppiness rather than raw volume ratio,
  SOXX bias, or leverage-adjusted signed tracking-error reversion.

## Features

| Feature | Mechanism |
|---------|-----------|
| `SOXL_SOXX_RATIO_Z_252` | SOXL/SOXX log price-ratio level z-score |
| `SOXL_SOXX_RATIO_WIDEN_5` | Five-day change in ratio z-score |
| `SOXL_SOXX_SPREAD_VOL_Z_21` | 21-day ratio-return volatility z-score |
| `SOXL_SOXX_CHOP_DIVERGENCE_21` | SOXL choppiness relative to SOXX choppiness |

## Evidence

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_regime_features.py::test_soxx_spread_oscillation_pack_builds_ratio_and_chop_features -q
# 1 passed

PYTHONPATH=. qlib_venv/bin/python scripts/evaluate_factor_pack.py \
  --pack soxx_spread_oscillation \
  --horizons 1 5 10 20 \
  --thresholds -1.0 -0.5 0.0 0.5 1.0 1.5
```

Best IC by feature:

| Feature | Horizon | IC Mean | IC+ | t-stat |
|---------|---------|---------|-----|--------|
| `SOXL_SOXX_SPREAD_VOL_Z_21` | 20 | `+0.015606` | `0.488` | `2.603` |
| `SOXL_SOXX_RATIO_Z_252` | 20 | `-0.004930` | `0.551` | `-0.796` |
| `SOXL_SOXX_RATIO_WIDEN_5` | 5 | `-0.002802` | `0.491` | `-2.524` |
| `SOXL_SOXX_CHOP_DIVERGENCE_21` | 20 | `-0.001599` | `0.425` | `-0.241` |

Best standalone backtest by feature:

| Feature | Direction | Threshold | Sharpe | PF | Accuracy | Max DD |
|---------|-----------|-----------|--------|----|----------|--------|
| `SOXL_SOXX_SPREAD_VOL_Z_21` | long | `-1.0` | `+0.846` | `1.178` | `0.544` | `-0.912` |
| `SOXL_SOXX_CHOP_DIVERGENCE_21` | long | `-1.0` | `+0.828` | `1.157` | `0.537` | `-0.905` |
| `SOXL_SOXX_RATIO_Z_252` | long | `-1.0` | `+0.778` | `1.161` | `0.543` | `-0.802` |
| `SOXL_SOXX_RATIO_WIDEN_5` | long | `-1.0` | `+0.775` | `1.145` | `0.535` | `-0.934` |

Correlation:

| Feature | Max Core Correlation | Core Feature |
|---------|----------------------|--------------|
| `SOXL_SOXX_RATIO_Z_252` | `0.885` | `anchor_gap_180` |
| `SOXL_SOXX_RATIO_WIDEN_5` | `0.922` | `mom_5` |
| `SOXL_SOXX_SPREAD_VOL_Z_21` | `0.917` | `vol_coef_21` |
| `SOXL_SOXX_CHOP_DIVERGENCE_21` | `0.081` | `tracking_error_21` |

## Decision

Do not use this pack for model training now.

- The spread level, spread widening, and spread-volatility features are mostly
  re-expressions of existing trend, momentum, and volatility features.
- The choppiness divergence feature is genuinely orthogonal, but its IC and
  standalone execution quality are too weak.
- No feature clears the promotion gate. Revisit only if a future regime-gated
  sparse-event rule is explicitly designed; do not start a direct SOXX/SOXL
  spread feature retrain from this evidence.
