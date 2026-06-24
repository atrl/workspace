# Data Layout

`data/` is the registry-backed market data surface. Reusable data update and qlib
store logic lives under `src/data/`; scripts are thin command-line wrappers.

## Layout

- `raw/symbols/<TICKER>.csv` — raw OHLCV symbol inputs.
- `raw/external/<NAME>.csv` — external volatility or macro inputs such as VIX term-structure files.
- `processed/panels/<asset>_with_underlying_vix.csv` — merged ETF / underlying / VIX panels.
- `processed/features/` — reusable derived feature outputs.
- `qlib/<asset>/` — qlib binary stores.
- `cache/training/<profile>.pkl` — rebuildable training or profile caches.
- `datasets.yaml` — generated registry with row counts, date ranges, and existence checks.

## Refresh Rule

1. In a network-enabled environment, refresh the canonical SOXL panel, raw surfaces, qlib store, and registry with `PYTHONPATH=. qlib_venv/bin/python -m src.data.market_update`.
2. The default provider mode is `auto`: Futu OpenD, Yahoo chart API, Polygon/CBOE, then yfinance. Use `--provider futu`, `--provider yahoo_api`, `--provider polygon`, or `--provider yfinance` to force one source.
3. In a sandboxed or no-network environment, sync downstream raw and qlib surfaces from the existing canonical panel with `PYTHONPATH=. qlib_venv/bin/python -m src.data.market_update --local-only`.
4. By default, fetched overlap dates are used only for provider validation and new rows are appended. Use `--replace-overlap` only when intentionally rebuilding overlapping history.
5. Validate freshness with `PYTHONPATH=. qlib_venv/bin/python -m pytest tests/test_market_data_update.py tests/test_schedule_refresh_pipeline.py -q`.

SOXL is the canonical default asset, but the layout supports the registered 3x ETF universe: SOXL, TQQQ, UPRO, SPXL, and TECL.
