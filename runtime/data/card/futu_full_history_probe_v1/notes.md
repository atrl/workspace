# Futu Full History Probe v1

Date: 2026-05-23

## Goal

Check whether Futu can provide full-history SOXL/SOXX data suitable for model
training, and distinguish full-history OHLCV from shorter-history order-flow
fields.

## External Reference

Futu OpenAPI `get_capital_flow` supports `start` / `end` for historical periods,
but the official parameter rules describe a 365-day default window when one side
of the date range is omitted. In this workspace, explicit older windows for
SOXL/SOXX also returned no rows, so availability must be treated as an empirical
data-provider limit for this account/source.

Reference: https://openapi.futunn.com/futu-api-doc/en/quote/get-capital-flow.html

## Probe Commands

OpenD preflight:

```bash
PYTHONPATH=. qlib_venv/bin/python production/check_futu_opend.py
# OK: OpenD is reachable at 127.0.0.1:11111
```

Capital-flow history windows:

```text
US.SOXL 2010-03-11..2011-12-31 rows=0
US.SOXL 2016-01-01..2016-12-31 rows=0
US.SOXL 2020-01-01..2020-12-31 rows=0
US.SOXL 2024-01-01..2024-12-31 rows=0
US.SOXL 2025-01-01..2026-05-22 rows=251 range=2025-05-23..2026-05-22
US.SOXL 2010-03-11..2026-05-22 rows=251 range=2025-05-23..2026-05-22

US.SOXX 2010-03-11..2011-12-31 rows=0
US.SOXX 2016-01-01..2016-12-31 rows=0
US.SOXX 2020-01-01..2020-12-31 rows=0
US.SOXX 2024-01-01..2024-12-31 rows=0
US.SOXX 2025-01-01..2026-05-22 rows=251 range=2025-05-23..2026-05-22
US.SOXX 2010-03-11..2026-05-22 rows=251 range=2025-05-23..2026-05-22
```

Historical K-line pagination:

```text
data/raw/external/futu_history_kline_US_SOXL.csv:
  rows=4060 pages=5 start=2010-03-31 end=2026-05-22

data/raw/external/futu_history_kline_US_SOXX.csv:
  rows=4076 pages=5 start=2010-03-11 end=2026-05-22
```

## Decision

- Futu OHLCV/K-line data can be cached over effectively the full available
  SOXL/SOXX training horizon and is now persisted under `data/raw/external/`.
- Futu capital-flow / small-order-flow data is **not** full history in this
  account/source. It currently covers only `2025-05-23` through `2026-05-22`.
- Do not train an RL profile with Futu small-order-flow features as if they were
  full-history features. That would create a train/evaluation coverage mismatch.
- Valid next steps are:
  - use Futu K-line only as an alternate raw OHLCV data source/comparison layer;
  - keep Futu small-order flow as a short-history live/overlay candidate;
  - find another provider for full-history order-flow before using this feature
    family in full-horizon training.

## Verification

```bash
PYTHONPATH=. qlib_venv/bin/python scripts/update_data_registry.py
# update_data_registry: OK symbols=6 panels=10 qlib=6
```
