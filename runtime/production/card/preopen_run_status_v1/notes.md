# Pre-open Run Status V1

## Goal

Make the scheduled pre-open dry-run inference path observable with the same
time-window discipline as post-close, without changing the final production
goal gates.

## Root Cause

`production/check_scheduler_health.sh` listed `scheduler_preopen.log` and the
last generic success marker, but it did not classify whether the 21:00 Mon-Fri
Asia/Shanghai pre-open slot was still pending, due, stale, or current. After
adding `--schedule-slot`, a future natural pre-open run also needed an explicit
target check so manual or old log lines do not look like current scheduled
dry-run proof.

## Change

- Added a `Pre-open Run Status` section to scheduler health.
- The section reports:
  - `pending until 21:00 Mon-Fri Asia/Shanghai` before the slot,
  - `due but no success marker found` after the slot with no run,
  - `due but no current success marker found` when only old success exists,
  - `success marker found` for a current run.
- A current success also prints `preopen target`, requiring:
  `schedule_slot=preopen`, `dry_run=True`, `skip_data_update=True`, and the
  enabled runtime-composer target fields when composer env is configured.
- This is an observability surface only; it does not add a new final goal gate.

## Evidence

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_scheduler_health.py::test_scheduler_health_reports_preopen_due_without_success_marker \
  tests/test_scheduler_health.py::test_scheduler_health_reports_preopen_target_for_current_success_marker -q
# red before implementation: Pre-open Run Status section missing
# green after implementation: 2 passed

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py -q
# 25 passed

production/check_scheduler_health.sh
# preopen status: pending until 21:00 Mon-Fri Asia/Shanghai
# overall: pending external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof
```

## Decision

The next natural 21:00 pre-open LaunchAgent run now has an audit line that can
prove current scheduled dry-run inference and composer targeting. The thread
goal remains blocked only on the existing external Vercel setup, scheduled
post-close target, and ordinary paper proof gates.
