# Tushare SOXX Factor Intake v1

Date: 2026-05-23

## Goal

Add a safe path for more SOXX/Tushare-derived factors without repeating the
failed direct SOXX actor-feature retrain.

## Baseline

- Current best runtime model: `v19reg_v22_budget_usage_bull_guard_v1`.
- Trainable fast actor baseline: `balanced_rule_residual_v22`.
- Direct comparison failure: `balanced_rule_residual_v22_soxx` was already
  dropped because direct `soxx_signals` actor injection raised 2022 exposure and
  the SOXX cap follow-up was absorbed in walk-forward execution.

## External Reference

Tushare Pro supports token-based Python SDK or HTTP calls. For the relevant US
market path, official docs list:

- `us_daily_adj`: US adjusted daily market data with `vol`, `amount`, `vwap`,
  `adj_factor`, `turnover_ratio`, free/total shares, and market value fields.
- Tushare permissions are token/permission gated; US daily data is a separate
  permission surface, so this repo path must remain optional.

Sources:

- https://tushare.pro/document/1?doc_id=40
- https://tushare.pro/document/2?doc_id=338
- https://tushare.pro/document/1?doc_id=290

## Implementation

- Added `src/soxl_quant/core/tushare.py`.
- Added `scripts/fetch_tushare_data.py`.
- Added feature pack `tushare_us_daily_adj`:
  - `TUSHARE_SOXX_TURNOVER_ZSCORE_21`
  - `TUSHARE_SOXX_AMOUNT_ZSCORE_21`
  - `TUSHARE_SOXL_SOXX_TURNOVER_SPREAD_21`
  - `TUSHARE_SOXX_ADJ_FACTOR_DELTA_5`
- All cached Tushare fields are shifted by one trading day before feature
  construction to avoid after-close data leakage.
- Missing cache files produce zero-valued features so existing training and
  runtime paths do not require private Tushare credentials.

## SOXX Factor Recheck

Command:

```bash
PYTHONPATH=. qlib_venv/bin/python scripts/evaluate_factor_pack.py --pack soxx_signals
```

Readout:

| Feature | Best IC | IC+Ratio | t-stat | Max Core Corr | Decision |
|---|---:|---:|---:|---:|---|
| `REL_VOLUME_RATIO` | `+0.058251` at `20d` | `0.591` | `12.459` | `0.136` vs `volume_shock` | keep as IC candidate |
| `PRICE_RATIO_MOM` | `-0.000532` at `5d` | `0.496` | `-2.596` | `0.993` vs `mom_5` | drop/redundant |
| `SOXX_BIAS_10` | `-0.000121` at `5d` | `0.501` | `-1.687` | `0.916` vs `mom_5` | drop/redundant |

The built-in factor evaluator's top standalone backtest row for
`REL_VOLUME_RATIO` uses a permissive threshold that is effectively always long,
so the backtest line is not promotion evidence. PF `1.159` and accuracy `0.537`
remain below the factor gate.

## Decision

Do not start another direct SOXX actor-feature retrain in this loop.

The next valid retrain requires:

1. Fetch `SOXX` and `SOXL` Tushare cache with `TUSHARE_TOKEN`.
2. Evaluate `tushare_us_daily_adj` with IC, standalone backtest, and correlation
   budget.
3. Only if at least one Tushare-derived feature clears the factor gate, create a
   governed profile that uses the current best trainable baseline and records
   why it is not a duplicate of `balanced_rule_residual_v22_soxx`.

## Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_tushare_integration.py -q
# 3 passed

uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_prune_artifacts.py tests/test_tushare_integration.py \
  tests/test_regime_features.py tests/test_profile_registry.py -q
# 21 passed

PYTHONPATH=. qlib_venv/bin/python scripts/fetch_tushare_data.py \
  --api us_daily_adj --symbols SOXX SOXL --start-date 20240101 \
  --end-date 20260522 --allow-missing-token
# SKIP missing TUSHARE_TOKEN

PYTHONPATH=. qlib_venv/bin/python - <<'PY'
from src.soxl_quant.data import feature_columns_for_packs, feature_schema_signature
for packs in [(), ('soxx_signals',), ('tushare_us_daily_adj',), ('soxx_signals','tushare_us_daily_adj')]:
    print(packs, len(feature_columns_for_packs(packs)), feature_schema_signature(packs))
PY
# () 59 99a98f3b024c
# ('soxx_signals',) 62 cc8deaa3f294
# ('tushare_us_daily_adj',) 63 b7a88d5ac602
# ('soxx_signals', 'tushare_us_daily_adj') 66 7b26e0016567

MPLCONFIGDIR=/private/tmp/mplconfig PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py
# [summary] passed=203 failures=0
```
