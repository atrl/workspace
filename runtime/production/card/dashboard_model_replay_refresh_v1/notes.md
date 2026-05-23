# Dashboard Model Replay Refresh

Date: 2026-05-23

Direction: production

## Goal

Make the static/Vercel dashboard useful for model inspection by refreshing
registry-owned model replay CSVs to the latest safe qlib backtest boundary
before rebuilding `dashboard.html` and `dashboard_data/`.

## Root Cause

The dashboard builder only packaged existing
`runtime/model/card/*/*_soxl_daily.csv` files. Paper trading was current through
`2026-05-22`, but many model versions still ended at `2026-03-31` or fixed
walk-forward windows. A one-model probe showed `export_profile_daily_backtest`
can replay current data, but qlib daily backtests require a following calendar
boundary, so replay curves can safely target `2026-05-21` while paper/latest
signal can use the terminal `2026-05-22` bar.

## Change

- Added `scripts/refresh_dashboard_model_replays.py`.
- The script discovers registry rows with `keep_actor=true`, non-dropped status,
  a retained actor, and a resolvable `profile.yaml`.
- It refreshes only stale `<profile>_soxl_daily.csv` files and writes
  `dashboard_data/model_replay_freshness.json`.
- `production/soxl_daily_scheduler.py` now runs this refresh before dashboard
  rebuild unless `--skip-dashboard` or `--skip-model-replay-refresh` is set.
- `scripts/check_dashboard_static_export.py` now requires and reports model
  replay freshness.
- The dashboard production strip shows `Model Replays`, and version dropdown
  labels include end date, rows, and source kind.
- Duplicate dashboard tags now prefer the matching profile directory, fixing
  `smoke` being overwritten by stale `runtime/model/card/smoke_tide/`.

## Evidence

```bash
PYTHONPATH=. qlib_venv/bin/python scripts/refresh_dashboard_model_replays.py --python qlib_venv/bin/python
# model_replay_refresh profiles=10 current=0 refreshed=10 failed=0 skipped=0 latest_data_date=2026-05-22 safe_backtest_end=2026-05-21

PYTHONPATH=. qlib_venv/bin/python scripts/build_backtest_dashboard.py
# wrote dashboard.html, dashboard_data/index.json, dashboard_data/model_versions.json, and 38 versioned exports

PYTHONPATH=. qlib_venv/bin/python scripts/check_dashboard_static_export.py
# ok dashboard_static_export generated_at=2026-05-23 11:22:25 versions=38 missing=0 model_versions=94 latest=paper_trading paper_trading_rows=6 paper_trading_end=2026-05-22 production_status=blocked_no_credentials goal_gates=pending model_replays=10/10@2026-05-21 paper_trading
```

Regression coverage:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_dashboard_model_replays.py \
  tests/test_dashboard.py::test_load_versions_prefers_matching_profile_directory_for_duplicate_tag \
  tests/test_vercel_dashboard.py::test_dashboard_static_export_integrity_check_passes -q
# 4 passed
```

Spot-check:

```text
dashboard_data/index.json:
balanced.end=2026-05-21
balanced_rule_residual_v22.end=2026-05-21
smoke.end=2026-05-21
smoke.source=runtime/model/card/smoke/smoke_soxl_daily.csv
paper_trading.end=2026-05-22
```

## Boundary

This fixes local/static bundle freshness and scheduled generation. The Vercel
remote deployment gate still reports `blocked_no_credentials` until
`VERCEL_TOKEN` or `vercel login` plus project linking is configured.
