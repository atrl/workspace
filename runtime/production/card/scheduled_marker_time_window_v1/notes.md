# Scheduled Marker Time Window Guard

Date: 2026-05-21

Direction: production

## Goal

Prevent manual or delayed LaunchAgent replays from satisfying scheduled
pre-open or post-close proof just because they inherit valid
`SOXL_SCHEDULE_TRIGGER` provenance.

## Root Cause

The scheduler health gate already required `schedule_slot` and
`schedule_trigger`, but it did not verify the run-start timestamp against the
configured schedule window. A manual `launchctl start` outside the intended
time could still emit:

```text
schedule_slot=postclose schedule_trigger=launchagent_postclose
```

and be accepted as scheduled post-close proof.

## Change

- `production/check_scheduler_health.sh` now parses the scheduler run-start
  timestamp and requires successful scheduled markers to start inside the
  expected schedule window.
- Defaults:
  - pre-open: `21:00` plus `SOXL_SCHEDULER_HEALTH_PREOPEN_WINDOW_MINUTES`
    defaulting to 90 minutes
  - post-close: `06:15` plus
    `SOXL_SCHEDULER_HEALTH_POSTCLOSE_WINDOW_MINUTES` defaulting to 90 minutes
- If an append-only log contains an earlier valid scheduled success and a later
  manual/debug run, the earlier valid proof is retained.

## Evidence

Red tests before the fix:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_scheduler_health.py -q -k 'outside_schedule_window'
# 2 failed
```

Green tests after the fix:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_scheduler_health.py -q -k 'outside_schedule_window'
# 2 passed, 28 deselected

uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_scheduler_health.py -q
# 30 passed
```

Live health after the change:

```text
preopen status: success marker found
preopen target: ok schedule_slot=preopen schedule_trigger=preopen dry_run=True skip_data_update=True runtime_composer=budget_usage composer_profile=v19reg_v22_budget_usage_bull_guard_v1
postclose status: due but current success marker lacks schedule_slot=postclose
postclose target: pending latest_success_marker lacks schedule_slot=postclose
overall: pending external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof
```

## Decision

The timed proof gate now requires three conditions for scheduled evidence:
date, scheduled trigger provenance, and run-start proximity to the intended
schedule. This closes the manual LaunchAgent replay loophole but does not clear
the current production goal. Vercel credentials/project link, scheduled
post-close proof, and next-date non-dry paper proof remain pending.
