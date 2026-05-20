# Scheduler Health v1

## Context

After moving the SOXL scheduler to `Asia/Shanghai` local-time windows, the
health-check script still inspected Linux cron. On the current macOS /
LaunchAgent environment this made the service section blank, so the operator
could not tell whether the installed scheduler was loaded.

This card is the first health-check fix. The later
`runtime/production/card/scheduler_launchagent_v1/notes.md` card extends it by
adding the missing pre-open LaunchAgent and aligning LaunchAgent log paths with
the health script.

## Change

- `production/check_scheduler_health.sh`
  - detects macOS through `uname` or `SOXL_SCHEDULER_HEALTH_OS`
  - checks `launchctl list` for SOXL scheduler LaunchAgents on macOS
  - keeps the Linux cron branch for cron deployments
  - uses `SOXL_SCHEDULER_HEALTH_ROOT` for deterministic tests
  - replaces GNU-only `ls --time-style` with portable `stat` / `date` helpers
  - reports success markers from `scheduler_preopen.log` and
    `scheduler_postclose.log`

## Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py -q
# 1 passed

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py tests/test_production_schedule.py tests/test_daily_scheduler.py tests/test_paper_trading.py tests/test_vercel_dashboard.py tests/test_dashboard.py -q
# 21 passed

production/check_scheduler_health.sh
# LaunchAgent com.soxl.daily.scheduler: loaded
# scheduler_preopen.log: NOT FOUND
# scheduler_postclose.log: NOT FOUND
```

Interpretation:

- The LaunchAgent is loaded on the current macOS host.
- The corrected pre-open / post-close log files have not been produced yet in
  this workspace, so the next live proof is still the first real
  `05:35 Tue-Sat Asia/Shanghai` post-close run.
