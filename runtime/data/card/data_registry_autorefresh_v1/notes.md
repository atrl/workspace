# Data Registry Auto-Refresh V1

Date: 2026-05-21
Direction: data / production

## Goal

Make scheduled and local data updates refresh `data/datasets.yaml` automatically
so SOXX/model diagnostics do not read stale data freshness metadata after the
actual panel/raw/qlib surfaces update.

## Root Cause

Live state before the fix:

```text
data/processed/panels/soxl_soxx_vix_full.csv 2026-05-20 4074
data/raw/symbols/SOXL.csv                    2026-05-20 4074
data/raw/symbols/SOXX.csv                    2026-05-20 4074
data/raw/external/VIX.csv                    2026-05-20 4074
data/qlib/soxl/calendars/day.txt             2026-05-20 4074
data/datasets.yaml                           2026-05-19 4073
```

The scheduled paper runner updated the master panel, raw mirrors, and qlib
store, but did not refresh `data/datasets.yaml`. The registry checker only
validated required sections, so stale rows/end metadata still passed.

## Change

- `scripts/update_market_data.py`
  - adds `refresh_data_registry(dry_run=...)`;
  - calls it from the canonical updater after raw/qlib refresh.
- `production/soxl_paper_trading.py`
  - calls `refresh_data_registry()` after scheduled data update sync and qlib
    refresh.
- `scripts/update_data_registry.py`
  - `--check` now rebuilds current metadata and fails when stored metadata
    drifts from actual files.
- `scripts/check_harness.py`
  - includes `scripts/update_data_registry.py --check` as a default command
    check.
- `data/datasets.yaml`
  - refreshed to the current `2026-05-20` SOXL/SOXX/VIX data surface.

## Evidence

```bash
PYTHONPATH=. qlib_venv/bin/python -c "import importlib.util, pathlib; p=pathlib.Path('tests/test_update_market_data.py'); spec=importlib.util.spec_from_file_location('tu', p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); m.test_data_registry_check_reports_stale_canonical_panel_end()"
# RED before implementation: TypeError: check_registry() got an unexpected keyword argument 'current_registry'
# GREEN after implementation: exit 0

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py::test_data_update_syncs_raw_surfaces_before_qlib_refresh -q
# RED before implementation: AttributeError refresh_data_registry missing
# GREEN after implementation: 1 passed

PYTHONPATH=. qlib_venv/bin/python scripts/update_data_registry.py --check
# RED before registry refresh: stale registry symbols/panel rows and end dates

PYTHONPATH=. qlib_venv/bin/python scripts/update_market_data.py --local-only
# local-only rows=4074 last_date=2026-05-20
# refreshed data/raw/symbols/SOXL.csv, data/raw/symbols/SOXX.csv,
# data/raw/external/VIX.csv, data/qlib/soxl, data/datasets.yaml

PYTHONPATH=. qlib_venv/bin/python scripts/update_data_registry.py --check
# update_data_registry: OK symbols=6 panels=10 qlib=6
```

Current registry readout:

```text
generated_at=2026-05-21T09:28:59.755191+00:00
processed_panels.soxl_soxx_vix_full: end=2026-05-20 rows=4074
symbols.SOXL: end=2026-05-20 rows=4074
symbols.SOXX: end=2026-05-20 rows=4074
external.VIX: rows=4074
```

## Decision

This fixes the data freshness control surface for the SOXX/model continuation
loop. It does not change the active runtime model, Vercel credential status, or
scheduled paper proof gates.
