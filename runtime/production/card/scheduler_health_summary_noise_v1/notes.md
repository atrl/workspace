# Scheduler Health Summary Noise V1

Date: 2026-05-21

## Goal

Keep scheduler health-summary logs focused on the final production goal gates
when the health check itself succeeds.

## Root Cause

`production/soxl_daily_scheduler.py` captured
`production/check_scheduler_health.sh`, extracted the `Production Goal Gates`
section, but still replayed successful health-check stderr. qlib/gym warnings
from the health checker could therefore appear around the concise scheduler
goal-gate summary even though the health command exited 0.

## Change

- `_run_health_summary()` now suppresses health-check stderr when the health
  command succeeds.
- If the health command fails, stderr is still replayed and the scheduler prints
  the health-summary warning with the failing exit code.

## Evidence

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_daily_scheduler.py::test_health_summary_suppresses_successful_stderr_noise \
  tests/test_daily_scheduler.py::test_health_summary_replays_stderr_on_failure -q
# red before implementation: successful stderr noise was replayed
# green after implementation: 2 passed

production/run_soxl_scheduler.sh \
  --broker paper --dry-run --no-notify --skip-data-update --skip-dashboard \
  --health-summary 2>&1 | sed -n '/\[scheduler\] all done/,$p'
# [scheduler] all done
# [scheduler] production goal gates
# --- Production Goal Gates ---
# overall: pending external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof
```

## Decision

The scheduled health-summary section should remain concise on successful health
runs, while failed health runs still expose stderr diagnostics. This does not
change the production completion gate state:

```text
overall: pending external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof
```
