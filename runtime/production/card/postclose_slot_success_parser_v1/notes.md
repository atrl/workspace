# Post-close Slot Success Parser V1

Date: 2026-05-21

## Goal

Keep scheduled post-close target proof durable when later manual/debug scheduler
runs append to `runtime/soxl/paper_trading/scheduler_postclose.log`.

## Root Cause

`production/check_scheduler_health.sh` already required `schedule_slot=postclose`
before clearing the post-close composer target, but it checked only the final
run-start marker in the append-only post-close log. A later manual run without a
slot marker could therefore hide an earlier successful post-close slot segment
from the same health date.

## Change

- Added `postclose_last_success_slot_marker()` to scan the append-only log for
  the latest current-date segment with `schedule_slot=postclose` followed by a
  strict `[scheduler] all done`.
- `print_postclose_target_status()` now evaluates that successful post-close
  slot marker before falling back to the latest raw run marker.
- `postclose_success_after_cutoff()` accepts that successful post-close slot
  segment even if a later manual/debug run was appended without its own
  completion marker.
- Cached dry-run completion markers remain excluded because they do not match
  the strict success line.

## Evidence

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_scheduler_health.py::test_scheduler_health_keeps_prior_postclose_slot_success_when_later_manual_run_appended -q
# red before implementation: later manual/no-slot markers hid the prior postclose slot success

uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_scheduler_health.py::test_scheduler_health_keeps_prior_postclose_slot_success_when_later_manual_run_appended \
  tests/test_scheduler_health.py::test_scheduler_health_requires_postclose_schedule_slot_for_composer_target -q
# 2 passed

production/check_scheduler_health.sh
# data: ok last_date=2026-05-20 expected=2026-05-20
# postclose target: pending latest_success_marker lacks schedule_slot=postclose
# ordinary paper proof: pending selected_model=v19reg_v22_budget_usage_bull_guard_v1 execution=duplicate_skipped
# overall: pending external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof

PYTHONPATH=. qlib_venv/bin/python scripts/export_production_goal_gates.py
# production_goal_gates status=pending issues=external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof

PYTHONPATH=. qlib_venv/bin/python scripts/check_dashboard_static_export.py
# ok ... production_status=blocked_no_credentials goal_gates=pending paper_trading
```

## Decision

The current live log still lacks a valid `schedule_slot=postclose` success
segment, so the gate correctly remains pending. Once the next scheduled
post-close LaunchAgent run writes the slot marker, later manual appends should
not erase that proof.
