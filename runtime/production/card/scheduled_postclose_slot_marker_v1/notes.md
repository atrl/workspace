# Scheduled Post-close Slot Marker V1

Date: 2026-05-21

## Goal

Prevent manual scheduler runs from accidentally satisfying the scheduled
post-close production gate.

## Root Cause

`production/check_scheduler_health.sh` audited the latest post-close log marker
for the runtime-composer target fields, but the marker did not identify whether
the run came from the pre-open slot, post-close slot, or a manual smoke run. A
manual run written to `scheduler_postclose.log` could therefore look like a
scheduled post-close run if the composer fields matched.

## Change

- `production/soxl_daily_scheduler.py` now accepts
  `--schedule-slot {manual,preopen,postclose}` and prints
  `schedule_slot=<value>` in the run-start marker.
- Repo crontab and LaunchAgent plists pass `--schedule-slot preopen` or
  `--schedule-slot postclose`.
- `production/check_scheduler_health.sh` now requires
  `schedule_slot=postclose` before treating the post-close composer target as
  satisfied.
- Operator docs were updated so manual cron examples preserve the same slot
  markers.
- Installed daily/pre-open LaunchAgent plists were copied from repo and
  reloaded.

## Evidence

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_daily_scheduler.py::test_scheduler_prints_schedule_slot_in_run_start_marker \
  tests/test_production_schedule.py::test_crontab_uses_shanghai_safe_us_market_windows \
  tests/test_production_schedule.py::test_launchagent_runs_post_close_in_shanghai_local_time \
  tests/test_production_schedule.py::test_launchagent_runs_preopen_cached_dry_run_in_shanghai_local_time \
  tests/test_production_schedule.py::test_operator_doc_scheduler_examples_include_current_production_flags \
  tests/test_scheduler_health.py::test_scheduler_health_requires_postclose_schedule_slot_for_composer_target -q
# red before implementation: --schedule-slot missing and health accepted a no-slot marker
# green after implementation: 6 passed

plutil -lint \
  production/com.soxl.daily.scheduler.plist \
  production/com.soxl.preopen.scheduler.plist
# both OK

production/run_soxl_scheduler.sh \
  --broker paper --dry-run --no-notify --skip-data-update --skip-dashboard \
  --health-summary --schedule-slot preopen
# run-start marker contains schedule_slot=preopen
# post-close composer target: pending latest_success_marker lacks schedule_slot=postclose

production/check_scheduler_health.sh
# LaunchAgent daily/preopen plists: matches repo
# post-close composer target: pending latest_success_marker lacks schedule_slot=postclose
# overall: pending external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof
```

## Decision

The next real post-close LaunchAgent run must write
`schedule_slot=postclose` and the runtime-composer fields before the
`scheduled_postclose_target` gate can clear. This keeps the remaining simulated
paper proof gate tied to the actual scheduled post-close path.
