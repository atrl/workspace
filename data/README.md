# Data Layout

`data/` is the registry-backed market data surface. Scripts should use `src.soxl_quant.core.dataset` helpers instead of hard-coding paths.

## Layout

- `raw/symbols/<TICKER>.csv` — raw OHLCV symbol inputs.
- `raw/external/<NAME>.csv` — external volatility or macro inputs such as VIX term-structure files.
- `processed/panels/<asset>_with_underlying_vix.csv` — merged ETF / underlying / VIX panels.
- `processed/features/` — reusable derived feature outputs.
- `qlib/<asset>/` — qlib binary stores.
- `cache/training/<profile>.pkl` — rebuildable training or profile caches.
- `datasets.yaml` — generated registry with row counts, date ranges, and existence checks.

## Refresh Rule

1. Refresh raw CSVs or external inputs.
2. Rebuild processed panels.
3. Rebuild qlib stores and training caches.
4. Run `PYTHONPATH=. qlib_venv/bin/python scripts/update_data_registry.py`.
5. Validate with `PYTHONPATH=. qlib_venv/bin/python scripts/update_data_registry.py --check`.

SOXL is the canonical default asset, but the layout supports the registered 3x ETF universe: SOXL, TQQQ, UPRO, SPXL, and TECL.
