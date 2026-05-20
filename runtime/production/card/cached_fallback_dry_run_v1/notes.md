# Cached Fallback Dry-Run v1

## Context

The post-close scheduler can hit transient data-source failures such as rate
limits or empty overlapping SOXL/SOXX/VIX bars. The existing fallback retried the
paper runner with `--skip-data-update`, but it did not force `--dry-run`.

That meant a post-close data-refresh failure could still execute a simulated
paper order using stale cached data. This was misaligned with the production
goal: cached data is acceptable for visibility and dashboard refresh, but it
must not mutate broker state when the intended fresh data update failed.

## Change

- `production/soxl_daily_scheduler.py`
  - transient data-source fallback now appends both `--skip-data-update` and
    `--dry-run`
  - retry description changed to `paper trading (cached dry-run)`
  - success marker changed to `[scheduler] all done (with cached dry-run)`
  - warning notification now says cached data was used for dry-run only and no
    order was placed
- `tests/test_daily_scheduler.py`
  - regression proves the retry command includes `--dry-run`
- `tests/test_scheduler_health.py`
  - fixture log now uses the `cached dry-run` success marker

## Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_daily_scheduler.py::test_scheduler_retries_data_source_failure_as_cached_dry_run -q
# 1 passed after initially failing against the old non-dry-run retry

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_daily_scheduler.py tests/test_scheduler_health.py tests/test_production_schedule.py tests/test_paper_trading.py tests/test_vercel_dashboard.py tests/test_dashboard.py -q
# 22 passed

PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --dry-run --no-notify --skip-data-update
# completed cached-data dry-run, skipped order placement, rebuilt dashboard_data with 25 versions

production/check_scheduler_health.sh
# both LaunchAgents loaded; scheduler_preopen shows [scheduler] all done

PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py
# passed=111 failures=0
```

Expected behavior:

- If the data refresh succeeds, the post-close full scheduler can place paper
  orders normally.
- If the data refresh fails due to a transient provider issue, the scheduler can
  still produce a cached signal and rebuild the dashboard, but it cannot change
  paper broker state.
