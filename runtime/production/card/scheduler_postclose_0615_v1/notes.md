# Scheduler Postclose 0615 v1

## Context

The natural `05:35 Tue-Sat Asia/Shanghai` post-close run fired, but the data
provider surface was not reliably ready:

- `05:35`, `05:44`, and `05:50` attempts only reached
  `last_date=2026-05-19` while freshness expected `2026-05-20`.
- Those attempts correctly fell back to cached dry-run and did not mutate paper
  broker state.
- A later `06:12` run reached `SOXL/SOXX/VIX` overlap through `2026-05-20`,
  refreshed qlib, and completed the non-dry paper path. Same-day idempotency
  skipped duplicate order placement.

The same LaunchAgent kickstart also exposed that the scheduled Vercel deploy
path could not find the `vercel` CLI from LaunchAgent's minimal PATH, even
though the CLI exists at `/opt/homebrew/bin/vercel`.

The subsequent scheduled dashboard rebuild exposed one more retention issue:
retained static dashboard evidence for `balanced_rule_residual_v19_moe_v2` was
present under `dashboard_data/versioned/`, but the source daily CSV had been
intentionally deleted with the dropped actor. A rebuild removed that version
from `dashboard_data/index.json`.

## Changes

- Moved the post-close full pipeline schedule to
  `06:15 Tue-Sat Asia/Shanghai`:
  - `production/com.soxl.daily.scheduler.plist`
  - `production/crontab.txt`
  - `production/FUTU_SEMI_AUTO.md`
- Centralized post-close health labels in `production/check_scheduler_health.sh`
  with `POSTCLOSE_HHMM=0615` and `POSTCLOSE_TIME_LABEL=06:15`.
- Added health coverage that keeps the post-close proof pending during the
  06:00 provider buffer window.
- Updated `production/run_soxl_scheduler.sh` to prepend common Homebrew and npm
  binary directories to PATH before launching Python, so scheduled Vercel
  deploy attempts can find `/opt/homebrew/bin/vercel`.
- Updated `scripts/build_backtest_dashboard.py` to load retained static
  dashboard exports from `dashboard_data/versioned/` when their source CSVs are
  intentionally retired.

## Evidence

Red tests:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_production_schedule.py tests/test_scheduler_health.py -q
# failed on old 05:35 cron/plist/health next-run expectations

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py::test_scheduler_health_keeps_postclose_pending_during_provider_buffer -q
# failed because 06:00 was still treated as due

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_production_schedule.py::test_launchagent_wrapper_exists_and_loads_private_env_file -q
# failed because the wrapper did not expose /opt/homebrew/bin

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_dashboard.py::test_load_versions_keeps_retained_dashboard_export_without_source_csv -q
# failed because retained static dashboard exports were not loaded into the rebuilt index
```

Green tests:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_dashboard.py tests/test_daily_scheduler.py tests/test_production_schedule.py tests/test_scheduler_health.py tests/test_vercel_dashboard.py -q
# 46 passed

PYTHONPATH=src:. qlib_venv/bin/python scripts/check_dashboard_static_export.py
# ok ... versions=28 missing=0 model_versions=94 latest=paper_trading
```

Installed LaunchAgent state after sync/reload:

```text
StartCalendarInterval:
  Hour=6 Minute=15 Weekday=2..6
LaunchAgent com.soxl.daily.scheduler plist: matches repo
```

LaunchAgent kickstart after the schedule change:

```text
[scheduler] run started at 2026-05-21T10:58:48+08:00 broker=paper dry_run=False
[paper] data fresh: last_date=2026-05-20 expected=2026-05-20 stale_days=0
[auto_select] family=balanced model=soxl_balanced_rule_residual_v22_actor.pt score=64.88 source=walk_forward_mean
[paper] already executed for 2026-05-20 broker=paper; skipping duplicate order
[scheduler] vercel dashboard deploy
vercel_dashboard status=blocked_no_credentials
[scheduler] all done
```

Wrapper dry-run with an intentionally empty shell environment:

```bash
env -i HOME="$HOME" SOXL_LAUNCH_ENV_FILE="$HOME/.openclaw/soxl.env" production/run_soxl_scheduler.sh --broker paper --dry-run --no-notify --skip-data-update --vercel-deploy
# vercel_dashboard status=blocked_no_credentials
# vercel_cli=/opt/homebrew/bin/vercel in runtime/production/card/vercel_remote_deploy_v1/vercel_deployment_check.json
```

Health readout after the kickstart:

```text
data: ok last_date=2026-05-20 expected=2026-05-20
model: ok balanced_rule_residual_v22 (soxl_balanced_rule_residual_v22_actor.pt)
postclose status: success marker found
post-close: next eligible day 06:15 Tue-Sat Asia/Shanghai
Vercel Remote Deployment: blocked_no_credentials
```

## Decision

- `06:15 Tue-Sat Asia/Shanghai` is now the canonical post-close production
  schedule.
- The paper path is healthy and idempotent for an already-executed signal date.
- Vercel remote deployment remains blocked only by missing credentials/project
  auth; the scheduled environment can now find the CLI.
- Scheduled dashboard rebuilds preserve retained versioned dashboard evidence
  even after a dropped profile's source daily CSV is removed.
- The remaining model runtime proof is still the next fresh non-dry paper date
  that writes a root summary with `profile=balanced_rule_residual_v22`.
