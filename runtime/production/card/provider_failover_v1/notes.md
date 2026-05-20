# Provider Failover v1

## Context

The data refresh path was still fragile after the scheduler and freshness guards
were fixed. `production/soxl_paper_trading.py` hard-coded yfinance through
`update_master_csv()`, while `scripts/update_market_data.py` already supported a
second provider, `yahoo_api`.

Live probe at `2026-05-21 02:30 CST`:

- `yfinance --dry-run`: failed with `No overlapping dates for SOXL/SOXX/VIX`
- `yahoo_api --dry-run`: failed with Yahoo HTTP `429 Too Many Requests`

The current environment still could not extend data beyond `2026-05-15`, but the
production runner should try both supported providers before falling back to
cached dry-run.

## Change

- `production/soxl_paper_trading.py`
  - added `--data-provider {auto,yfinance,yahoo_api}`
  - default `auto` tries yfinance first, then yahoo_api on retryable data-update
    failures
  - keeps the existing stale-data execution guard after the update attempt
- `production/soxl_daily_scheduler.py`
  - added `--data-provider {auto,yfinance,yahoo_api}`
  - passes the provider choice through to the paper runner
- `production/FUTU_SEMI_AUTO.md`
  - documents provider failover and explicit provider selection
- Regression coverage:
  - `tests/test_paper_trading.py::test_data_update_auto_provider_falls_back_to_yahoo_api`
  - `tests/test_daily_scheduler.py::test_scheduler_passes_data_provider_to_paper_runner`

## Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py::test_data_update_auto_provider_falls_back_to_yahoo_api -q
# 1 passed after initially failing because _run_data_update did not exist

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_daily_scheduler.py::test_scheduler_passes_data_provider_to_paper_runner -q
# 1 passed after initially failing because scheduler did not accept --data-provider

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py tests/test_daily_scheduler.py tests/test_scheduler_health.py tests/test_production_schedule.py tests/test_vercel_dashboard.py tests/test_dashboard.py -q
# 29 passed

PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py
# passed=111 failures=0

PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --dry-run --no-notify --skip-dashboard --data-provider auto
# yfinance failed: No overlapping dates for SOXL/SOXX/VIX
# yahoo_api failed: 429 Too Many Requests
# scheduler retried cached dry-run
# [paper] dry-run: skipped order placement
# [scheduler] all done (with cached dry-run)

PYTHONPATH=. qlib_venv/bin/python scripts/build_backtest_dashboard.py
# wrote dashboard.html, dashboard_data/index.json, model_versions.json, and 25 versioned exports

curl -I http://127.0.0.1:8768/dashboard.html
curl -I http://127.0.0.1:8768/dashboard_data/index.json
curl -I http://127.0.0.1:8768/dashboard_data/model_versions.json
# all returned 200 OK
```

## Readout

Provider failover is now wired into the production runner, but the current live
data blocker remains external provider availability. Both providers failed in
the current run, and the scheduler correctly preserved visibility through cached
dry-run without mutating paper broker state.
