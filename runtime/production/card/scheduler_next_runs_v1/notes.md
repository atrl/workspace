# Scheduler Next-Run Health Display

Date: 2026-05-21

## Context

`production/check_scheduler_health.sh` previously printed a static
`Next Scheduled Runs (today)` section:

- `21:00 Mon-Fri Asia/Shanghai`
- `05:35 Tue-Sat Asia/Shanghai`

After the post-close window had already passed, this made the health output look
as if today's `05:35` run was still upcoming.

## Change

The health script now uses `SOXL_SCHEDULER_HEALTH_WEEKDAY` and
`SOXL_SCHEDULER_HEALTH_HHMM` to label each scheduler window as:

- `today` when that window is still ahead on an eligible local day
- `next eligible day` after that window has passed or on an ineligible day
- `next Saturday` for the weekly walk-forward after Saturday's window passes

## Evidence

Regression:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py::test_scheduler_health_next_runs_skip_elapsed_postclose_window -q
# red first on static "today" output; green after implementation
# 1 passed
```

Focused production/dashboard suite:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py tests/test_production_schedule.py tests/test_daily_scheduler.py tests/test_paper_trading.py tests/test_vercel_dashboard.py tests/test_dashboard.py -q
# 49 passed
```

Live health readout at `2026-05-21 08:00 CST`:

```text
--- Next Scheduled Runs ---
  pre-open: today 21:00 Mon-Fri Asia/Shanghai — US pre-open dry-run + dashboard
  post-close: next eligible day 05:35 Tue-Sat Asia/Shanghai — US post-close data + paper trading + dashboard
  weekly: next Saturday 07:00 Sat Asia/Shanghai — weekly walk-forward after post-close refresh
```

## Decision

This is an observability fix only. It does not change LaunchAgent schedules or
paper execution. The v22 paper proof still waits for the next fresh post-close
non-dry execution.
