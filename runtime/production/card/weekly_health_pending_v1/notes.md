# Weekly Scheduler Health Pending State

Date: 2026-05-21

## Goal

Make scheduler health distinguish the expected pre-first-run weekly
walk-forward state from a generic missing-log failure.

## Root Cause

`com.soxl.weekly.wf` was loaded and its plist matched the repo, but the first
Saturday run had not happened yet. Because `runtime/soxl/wf_weekly.log` and
`runtime/soxl/wf_weekly.err` are created by the scheduled command, health
reported both files as `NOT FOUND` before any weekly artifact could exist.

## Change

- `production/check_scheduler_health.sh` now reports missing weekly log files as
  `pending (not created yet)`.
- The weekly completion marker reports
  `wf_weekly: pending until first weekly run` while `wf_weekly.log` is absent.
- Preopen and postclose log checks still use `NOT FOUND` for missing files.

## Evidence

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_scheduler_health.py::test_scheduler_health_reports_weekly_logs_pending_before_first_run -q
# red before implementation: weekly logs were reported as NOT FOUND
# green after implementation: 1 passed

uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_scheduler_health.py -q
# 13 passed

production/check_scheduler_health.sh
# LaunchAgent com.soxl.weekly.wf: loaded
# LaunchAgent com.soxl.weekly.wf plist: matches repo
# runtime/soxl/wf_weekly.log: pending (not created yet)
# runtime/soxl/wf_weekly.err: pending (not created yet)
# wf_weekly: pending until first weekly run
```

## Remaining Gates

- First actual weekly walk-forward run still needs the Saturday artifact and
  `[weekly-wf] all done` marker.
- Selected `balanced_rule_residual_v22` non-dry paper runtime proof still waits
  for the next fresh post-close paper run.
- Vercel remote deployment remains blocked until `VERCEL_TOKEN` or
  `vercel login` is available.
