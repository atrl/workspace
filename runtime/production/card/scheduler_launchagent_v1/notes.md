# Scheduler LaunchAgent v1

## Context

The cron schedule already had two SOXL production windows:

- `21:00 Mon-Fri Asia/Shanghai`: cached-data pre-open dry-run inference plus
  dashboard refresh.
- `05:35 Tue-Sat Asia/Shanghai`: post-close data refresh, paper trading, and
  dashboard refresh.

The macOS LaunchAgent path only covered the post-close job, and it still wrote
to legacy `scheduler.log` / `scheduler.err`. That meant the current macOS host
could not run the pre-open scheduled inference path, and
`production/check_scheduler_health.sh` looked for log files that the installed
LaunchAgent would not create.

## Change

- `production/com.soxl.daily.scheduler.plist`
  - remains the post-close full pipeline
  - writes to `scheduler_postclose.log` and `scheduler_postclose.err`
- `production/com.soxl.preopen.scheduler.plist`
  - new pre-open LaunchAgent
  - runs `production/soxl_daily_scheduler.py --broker paper --dry-run --no-notify --skip-data-update`
  - writes to `scheduler_preopen.log` and `scheduler_preopen.err`
- `production/check_scheduler_health.sh`
  - checks both LaunchAgent labels:
    - `com.soxl.preopen.scheduler`
    - `com.soxl.daily.scheduler`
  - displays both stdout and stderr log files.
- `production/FUTU_SEMI_AUTO.md`
  - documents installing, loading, verifying, and unloading both LaunchAgents.

## External State

Synced repo LaunchAgents to `~/Library/LaunchAgents/` and reloaded both:

```bash
cp production/com.soxl.daily.scheduler.plist production/com.soxl.preopen.scheduler.plist ~/Library/LaunchAgents/
launchctl unload ~/Library/LaunchAgents/com.soxl.daily.scheduler.plist 2>/dev/null || true
launchctl unload ~/Library/LaunchAgents/com.soxl.preopen.scheduler.plist 2>/dev/null || true
launchctl load ~/Library/LaunchAgents/com.soxl.daily.scheduler.plist
launchctl load ~/Library/LaunchAgents/com.soxl.preopen.scheduler.plist
launchctl list | grep 'com.soxl.*scheduler'
# - 0 com.soxl.preopen.scheduler
# - 0 com.soxl.daily.scheduler
```

## Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_production_schedule.py tests/test_scheduler_health.py -q
# 4 passed

launchctl start com.soxl.preopen.scheduler
# completed cached-data dry-run inference and dashboard rebuild

production/check_scheduler_health.sh
# LaunchAgent com.soxl.preopen.scheduler: loaded
# LaunchAgent com.soxl.daily.scheduler: loaded
# scheduler_preopen.log: 4.0K
# scheduler_preopen: [scheduler] all done
```

Current boundary:

- Pre-open scheduled inference path is installed and manually verified through
  LaunchAgent.
- Post-close full data-refresh path is installed, but the first live
  `05:35 Tue-Sat Asia/Shanghai` run still needs to prove online data freshness.
