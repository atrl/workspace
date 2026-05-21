# Scheduler Stderr Signal Cleanup

Date: 2026-05-21

## Goal

Keep scheduled paper-trading stderr focused on actionable failures while Vercel
credentials remain unavailable.

## Root Cause

`production/soxl_daily_scheduler.py` correctly treats
`scripts/deploy_vercel_dashboard.py --deploy` return code `2` as nonfatal when
`--strict-vercel-deploy` is not set. However, `_run_vercel_deploy()` printed the
nonfatal continuation notice to stderr:

```text
[scheduler] Vercel deploy blocked by missing credentials; continuing after writing deployment evidence
```

That state is already recorded in
`runtime/production/card/vercel_remote_deploy_v1/vercel_deployment_check.json`
and reported by `production/check_scheduler_health.sh`, so sending the same
nonfatal status to stderr made scheduler error logs noisier.

## Change

- Non-strict Vercel return code `2` remains nonfatal.
- The missing-credentials continuation notice now prints to stdout.
- Strict Vercel failures and ordinary scheduler failures still use stderr.

## Evidence

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_daily_scheduler.py::test_scheduler_treats_missing_vercel_credentials_as_nonfatal -q
# red before implementation: notice was only in stderr
# green after implementation: 1 passed

qlib_venv/bin/python -m py_compile production/soxl_daily_scheduler.py
# ok

uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_daily_scheduler.py tests/test_scheduler_health.py -q
# 26 passed
```

## Remaining Gates

- Vercel remote deployment still requires scheduler-visible `VERCEL_TOKEN` or
  `vercel login`.
- Existing scheduler stderr retains historical provider-fallback errors for
  auditability; it is not automatically truncated.
