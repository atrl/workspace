# Freshness Market Close v1

## Context

The data freshness check used the local `Asia/Shanghai` calendar date to infer
the expected last trading day. That made health checks too strict between local
midnight and the US market close. For example, at `2026-05-21 02:23 CST`, New
York was still `2026-05-20 14:23 EDT`, so the latest completed US trading day
was still `2026-05-19`, not `2026-05-20`.

This mattered because the corrected post-close scheduler runs at
`05:35 Tue-Sat Asia/Shanghai`, after a `17:30 America/New_York` data-ready
cutoff. Freshness should only expect the current New York trading date after
that cutoff.

## Change

- `production/soxl_paper_trading.py`
  - `_last_trading_day()` now computes the expected date in
    `America/New_York`
  - uses `17:30 ET` as the data-ready cutoff
  - before the cutoff, falls back to the previous weekday
  - keeps weekend fallback to the prior weekday
- `tests/test_paper_trading.py`
  - proves `2026-05-21 02:23 CST` expects `2026-05-19`
  - proves `2026-05-21 05:35 CST` expects `2026-05-20`
  - proves Monday pre-open expects the prior Friday

## Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py::test_last_trading_day_waits_for_new_york_post_close_cutoff tests/test_paper_trading.py::test_last_trading_day_uses_previous_friday_before_monday_us_close -q
# red: _last_trading_day() had no time injection and used local calendar date
# green: 2 passed

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py tests/test_daily_scheduler.py tests/test_scheduler_health.py tests/test_production_schedule.py tests/test_vercel_dashboard.py tests/test_dashboard.py -q
# 27 passed

PYTHONPATH=. qlib_venv/bin/python production/soxl_paper_trading.py --healthcheck --skip-data-update --no-notify
# data: last_date=2026-05-15 expected_last_trading_day=2026-05-19 stale_days=4

PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --skip-data-update --no-notify --skip-dashboard
# first pass blocked stale non-dry-run data at expected=2026-05-19
# retry completed as cached dry-run with skipped order placement

production/check_scheduler_health.sh
# data: stale last_date=2026-05-15 expected=2026-05-19 stale_days=4
```

## Readout

The health surface now matches the corrected schedule boundary:

- before `05:35 CST`, do not expect the still-open US trading day
- after the post-close data-ready window, expect the just-closed US trading day

The live proof still requires the real `05:35 Tue-Sat Asia/Shanghai` run to
advance data beyond `2026-05-15`.
