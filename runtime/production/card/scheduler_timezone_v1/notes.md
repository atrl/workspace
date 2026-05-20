# Scheduler Timezone Guard v1

## Context

The production scheduler was documented as US pre-open / post-close, but the
repo is operated from an `Asia/Shanghai` local-time environment. The previous
local `09:30` and `16:35` cron / LaunchAgent schedule could run before the US
market close, so the daily data refresh was structurally likely to see stale or
empty provider data.

## Change

- `production/crontab.txt`
  - Replaced the stale local `09:30` and `16:35` SOXL scheduler rows.
  - Added `21:00 Mon-Fri Asia/Shanghai` cached-data `--dry-run` pre-open pass.
  - Added `05:35 Tue-Sat Asia/Shanghai` post-close full pipeline.
  - Moved weekly walk-forward to `07:00 Sat Asia/Shanghai`, after the post-close
    data refresh window.
- `production/com.soxl.daily.scheduler.plist`
  - Moved macOS LaunchAgent post-close full pipeline to `05:35 Tue-Sat`.
- `production/check_scheduler_health.sh`
  - Updated log names and next-run text to match the new schedule.
- `production/FUTU_SEMI_AUTO.md`
  - Documented the Asia/Shanghai cron schedule explicitly.

## Guardrail

`tests/test_production_schedule.py` now parses both cron and LaunchAgent
configuration and fails if SOXL production scheduling drifts back to the stale
local `09:30` or `16:35` windows.

## Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_production_schedule.py -q
# 2 passed

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_daily_scheduler.py tests/test_paper_trading.py tests/test_vercel_dashboard.py tests/test_production_schedule.py tests/test_dashboard.py -q
# 20 passed

PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py
# passed=111 failures=0

PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --dry-run --no-notify --skip-data-update
# completed cached-data dry-run, selected balanced_rule_residual_v19, rebuilt dashboard_data with 25 versions

curl -I http://127.0.0.1:8767/dashboard.html
curl -I http://127.0.0.1:8767/dashboard_data/index.json
# both returned 200 OK

npx -y playwright screenshot --channel chrome --viewport-size=1440,1000 --wait-for-timeout=2000 \
  http://127.0.0.1:8767/dashboard.html \
  runtime/production/card/scheduler_timezone_v1/dashboard_visual_2026-05-21.png
# captured screenshot using local Chrome
```

Screenshot artifact:

- `runtime/production/card/scheduler_timezone_v1/dashboard_visual_2026-05-21.png`
  - size: `1440 x 1000`
  - file size: `268K`
  - non-white pixels: `94.84%`

Computer Use remained blocked by pending macOS Accessibility / Screen Recording
permissions, so visual verification used Playwright with the installed local
Chrome channel instead.
