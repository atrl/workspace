# Scheduled Goal Gate Logging V1

Date: 2026-05-21

## Goal

Make scheduled SOXL runs self-auditing by printing the final production goal
gate summary at the end of successful scheduler logs.

## Root Cause

`production/check_scheduler_health.sh` already had a consolidated
`Production Goal Gates` section, but scheduled pre-open and post-close logs did
not include it. A future scheduled run could succeed without leaving a direct
log-level answer for whether the active production goal was complete.

## Change

- Added `--health-summary` to `production/soxl_daily_scheduler.py`.
- The scheduler now prints `[scheduler] production goal gates` and the final
  `Production Goal Gates` section after a completed normal run.
- The cached dry-run fallback path also emits the same summary after its
  successful completion marker.
- Added `--health-summary` to the repo crontab and both LaunchAgent plists.
- Reloaded the installed daily and pre-open LaunchAgents so the installed plist
  files match the repo copies.

## Evidence

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_daily_scheduler.py::test_scheduler_prints_goal_gates_when_requested \
  tests/test_production_schedule.py::test_crontab_uses_shanghai_safe_us_market_windows \
  tests/test_production_schedule.py::test_launchagent_runs_post_close_in_shanghai_local_time \
  tests/test_production_schedule.py::test_launchagent_runs_preopen_cached_dry_run_in_shanghai_local_time -q
# 4 passed

plutil -lint \
  production/com.soxl.daily.scheduler.plist \
  production/com.soxl.preopen.scheduler.plist
# both OK

production/run_soxl_scheduler.sh \
  --broker paper --dry-run --no-notify --skip-data-update --skip-dashboard \
  --health-summary
# [scheduler] all done
# [scheduler] production goal gates
# --- Production Goal Gates ---
# overall: pending external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof

production/check_scheduler_health.sh
# LaunchAgent com.soxl.preopen.scheduler plist: matches repo
# LaunchAgent com.soxl.daily.scheduler plist: matches repo
# overall: pending external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof
```

## Decision

The scheduled logs now carry the same completion audit surface as the manual
health check. The active production goal is still not complete because the
remaining live gates are:

- `external_vercel_setup`
- `scheduled_postclose_target`
- `scheduled_paper_proof`

Future sessions should inspect the next scheduled post-close log first. If it
contains `overall: ok`, the production goal can be considered cleared after
rerunning the health check for confirmation.
