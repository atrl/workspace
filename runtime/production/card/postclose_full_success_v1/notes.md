# Postclose Full Success v1

## Context

The first natural `05:35 Tue-Sat Asia/Shanghai` LaunchAgent run reached the
post-close scheduler, but it fell back to cached dry-run because the data panel
still ended at `2026-05-19` while freshness expected `2026-05-20`.

A later manual wrapper run showed a narrower bug: yfinance returned SOXL, SOXX,
and VIX rows through `2026-05-20`, but the merge still failed because provider
date keys were not normalized before joining.

## Changes

- `scripts/update_market_data.py`
  - normalizes fetched provider `date` columns to day-level naive timestamps
    before source-range diagnostics and merge.
- `production/soxl_paper_trading.py`
  - syncs raw SOXL/SOXX/VIX mirror CSVs after a successful production data
    update and before qlib refresh.
- Tests:
  - `test_update_master_csv_normalizes_provider_dates_before_merge`
  - `test_data_update_syncs_raw_surfaces_before_qlib_refresh`

## Readout

Manual wrapper run:

```bash
SOXL_LAUNCH_ENV_FILE="$HOME/.openclaw/soxl.env" production/run_soxl_scheduler.sh --broker paper --no-notify
```

Result:

- yfinance source ranges:
  `SOXL=2026-05-11..2026-05-20(8)`,
  `SOXX=2026-05-11..2026-05-20(8)`,
  `VIX=2026-05-11..2026-05-20(8)`,
  `overlap_last_date=2026-05-20`.
- canonical panel advanced from `4073` to `4074` rows, ending
  `2026-05-20`.
- freshness gate passed:
  `last_date=2026-05-20 expected=2026-05-20 stale_days=0`.
- model: `balanced_rule_residual_v19`.
- signal date: `2026-05-20`.
- target: `50.00%`.
- paper order: submitted and filled.
- dashboard: rebuilt.

LaunchAgent kickstart:

```bash
launchctl kickstart -k "gui/$(id -u)/com.soxl.daily.scheduler"
```

Result:

- scheduler log wrote exact `[scheduler] all done`.
- same-day idempotency skipped duplicate order for `2026-05-20`.
- `production/check_scheduler_health.sh` reported
  `postclose status: success marker found`.

## Artifacts

- `data/processed/panels/soxl_soxx_vix_full.csv`
  - rows: `4074`
  - last date: `2026-05-20`
- raw mirrors:
  - `data/raw/symbols/SOXL.csv`: `4074` rows, ending `2026-05-20`
  - `data/raw/symbols/SOXX.csv`: `4074` rows, ending `2026-05-20`
  - `data/raw/external/VIX.csv`: `4074` rows, ending `2026-05-20`
- `runtime/soxl/paper_trading/paper_broker_state.json`
  - date: `2026-05-20`
  - shares: `2800`
  - position: `0.4840856822`
  - account value: `1001806.1044`
- `runtime/soxl/paper_trading/2026-05-20/summary.json`
  - `dry_run=false`
  - `executed=true`
  - `risk_gate.approved=true`
- `runtime/soxl/paper_trading/2026-05-20/reconcile.json`
  - action required: `none`
  - note: `position aligned`
- `runtime/soxl/paper_trading/paper_trading_soxl_daily.csv`
  - rows: `4`
  - latest row: `2026-05-20`, action `SELL`, position `0.484086`
- `dashboard_data/index.json`
  - paper version: `end=2026-05-20`, `rows=4`

## Dashboard Verification

```bash
npx -y playwright screenshot --channel chrome --viewport-size=1440,1000 --wait-for-timeout=2000 \
  'http://127.0.0.1:8771/dashboard.html?ts=postclose_full_success_v1' \
  runtime/production/card/postclose_full_success_v1/dashboard_visual_2026-05-21.png
```

Screenshot artifact:

- `runtime/production/card/postclose_full_success_v1/dashboard_visual_2026-05-21.png`
  - size: `1440 x 1000`
  - non-white pixels: `99.35%`

## Final Gates

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_update_market_data.py tests/test_paper_trading.py tests/test_scheduler_health.py tests/test_production_schedule.py tests/test_daily_scheduler.py tests/test_vercel_dashboard.py tests/test_dashboard.py -q
# 48 passed

PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py
# passed=130 failures=0

PYTHONPATH=. qlib_venv/bin/python scripts/check_dashboard_static_export.py
# ok dashboard_static_export generated_at=2026-05-21 06:13:40 versions=25 missing=0 model_versions=90 latest=balanced_rule_residual_v22_soxx paper_trading_rows=4 paper_trading_end=2026-05-20 paper_trading

PYTHONPATH=. qlib_venv/bin/python scripts/build_runtime_index.py --check
# runtime index OK

git diff --check
# clean

production/check_scheduler_health.sh
# data: ok last_date=2026-05-20 expected=2026-05-20
# latest signal: date=2026-05-20 profile=balanced_rule_residual_v19 dry_run=false executed=true
# dashboard_static_export ... paper_trading_rows=4 paper_trading_end=2026-05-20
# postclose status: success marker found
```

## Decision

The post-close production loop is now proven through the repo wrapper and the
macOS LaunchAgent path: fresh data, scheduled inference, non-dry paper
execution, dashboard rebuild, Vercel-ready static bundle, and scheduler health
success marker.
