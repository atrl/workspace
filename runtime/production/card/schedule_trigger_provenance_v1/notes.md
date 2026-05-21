# Schedule Trigger Provenance V1

Date: 2026-05-21

## Goal

Make scheduled pre-open and post-close health proof depend on an actual
cron/LaunchAgent trigger, not only on a manually supplied `--schedule-slot`.

## Root Cause

The post-close slot parser required `schedule_slot=postclose`, but that flag can
also be supplied by a manual/debug command. That left a proof loophole where a
manual replay could look like a scheduled post-close target marker.

## Change

- `production/soxl_daily_scheduler.py` adds `schedule_trigger=<value>` to every
  run-start marker, sourced from `SOXL_SCHEDULE_TRIGGER` and defaulting to
  `manual`.
- `production/crontab.txt` sets `SOXL_SCHEDULE_TRIGGER=cron_preopen` and
  `SOXL_SCHEDULE_TRIGGER=cron_postclose`.
- `production/com.soxl.preopen.scheduler.plist` sets
  `SOXL_SCHEDULE_TRIGGER=launchagent_preopen`.
- `production/com.soxl.daily.scheduler.plist` sets
  `SOXL_SCHEDULE_TRIGGER=launchagent_postclose`.
- `production/check_scheduler_health.sh` accepts post-close target proof only
  when the successful segment has both `schedule_slot=postclose` and a scheduled
  post-close trigger.
- Pre-open target health now also requires scheduled pre-open trigger
  provenance.

## Evidence

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_daily_scheduler.py::test_scheduler_prints_schedule_trigger_in_run_start_marker \
  tests/test_production_schedule.py::test_crontab_uses_shanghai_safe_us_market_windows \
  tests/test_production_schedule.py::test_operator_doc_scheduler_examples_include_current_production_flags \
  tests/test_production_schedule.py::test_launchagent_runs_post_close_in_shanghai_local_time \
  tests/test_production_schedule.py::test_launchagent_runs_preopen_cached_dry_run_in_shanghai_local_time \
  tests/test_scheduler_health.py::test_scheduler_health_requires_postclose_schedule_trigger_for_composer_target \
  tests/test_scheduler_health.py::test_scheduler_health_keeps_prior_postclose_slot_success_when_later_manual_run_appended \
  tests/test_scheduler_health.py::test_scheduler_health_reports_preopen_target_for_current_success_marker -q
# 8 passed

uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_scheduler_health.py tests/test_daily_scheduler.py \
  tests/test_production_schedule.py tests/test_vercel_dashboard.py -q
# 70 passed

plutil -lint production/com.soxl.daily.scheduler.plist production/com.soxl.preopen.scheduler.plist
# both OK
```

Live health after syncing and reloading both scheduler LaunchAgents:

```text
LaunchAgent com.soxl.preopen.scheduler: loaded
LaunchAgent com.soxl.preopen.scheduler plist: matches repo
LaunchAgent com.soxl.daily.scheduler: loaded
LaunchAgent com.soxl.daily.scheduler plist: matches repo
postclose status: due but current success marker lacks schedule_slot=postclose
postclose target: pending latest_success_marker lacks schedule_slot=postclose
overall: pending external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof
```

Generated gate artifact:

```bash
PYTHONPATH=. qlib_venv/bin/python scripts/export_production_goal_gates.py
# production_goal_gates status=pending issues=external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof
```

## Decision

The current production gate remains pending. The next acceptable post-close
target proof must come from `launchagent_postclose` or `cron_postclose`; manual
replays remain diagnostics only unless they are backed by scheduled trigger
provenance.
