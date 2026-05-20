# Data Freshness Health v1

## Context

The scheduler health check showed LaunchAgent and log status, but it did not
surface whether the paper-trading loop itself was ready to run on fresh data.
That left a gap after fixing the `Asia/Shanghai` schedule: an operator could see
loaded LaunchAgents and a successful pre-open dry-run while still missing that
the canonical data panel remained stale.

## Change

- `production/soxl_paper_trading.py`
  - added `_data_freshness_status()`
  - healthcheck now reports `last_date`, `expected_last_trading_day`, and
    `stale_days` under `data`
  - `all_ok` remains false when the data panel is stale
- `production/check_scheduler_health.sh`
  - adds a `Paper Trading Health` section
  - runs `production/soxl_paper_trading.py --healthcheck --skip-data-update
    --no-notify`
  - parses JSON out of mixed stdout/stderr so model-loading warnings do not hide
    the structured health payload
  - treats stale data as a displayed health status, not as a shell-script
    failure
- `tests/test_paper_trading.py`
  - covers expected-date and stale-day reporting
- `tests/test_scheduler_health.py`
  - verifies stale data, model, and broker lines are printed even when the
    underlying healthcheck exits non-zero

## Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py::test_data_freshness_status_reports_expected_date_and_lag -q
# 1 passed

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py::test_scheduler_health_reports_macos_launchagent_and_logs -q
# 1 passed

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py tests/test_scheduler_health.py tests/test_daily_scheduler.py tests/test_production_schedule.py tests/test_vercel_dashboard.py tests/test_dashboard.py -q
# 23 passed

production/check_scheduler_health.sh
# LaunchAgent com.soxl.preopen.scheduler: loaded
# LaunchAgent com.soxl.daily.scheduler: loaded
# scheduler_preopen: [scheduler] all done
# scheduler_postclose.log: NOT FOUND
# data: stale last_date=2026-05-15 expected=2026-05-20 stale_days=5
# model: ok balanced_rule_residual_v19 (soxl_balanced_rule_residual_v19_actor.pt)
# broker: ok paper broker ready
```

## Readout

The health script now makes the current blocker explicit: scheduling and
pre-open dry-run inference are installed, but the data panel is still stale at
`2026-05-15`. The first real `05:35 Tue-Sat Asia/Shanghai` post-close run still
needs to prove that online refresh can advance data beyond that date.
