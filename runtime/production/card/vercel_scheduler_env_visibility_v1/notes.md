# Vercel Scheduler Env Visibility v1

## Goal

Make the Vercel deployment blocker diagnosable from scheduler health without
requiring access to private credentials or printing secret values.

## Issue

`production/check_scheduler_health.sh` reported Vercel as
`blocked_no_credentials`, but it did not say whether the LaunchAgent wrapper
could see the private env file that is supposed to provide `VERCEL_TOKEN`.
That left two different failure modes collapsed together:

- the scheduler env file was missing or not visible to LaunchAgent
- the env file existed but did not contain `VERCEL_TOKEN`

## Change

- Added a `Scheduler Private Env` section to
  `production/check_scheduler_health.sh`.
- The health check uses the same `SOXL_LAUNCH_ENV_FILE` default as the
  LaunchAgent wrappers: `${HOME}/.openclaw/soxl.env`.
- It reports whether the env file is present and whether `VERCEL_TOKEN` is
  configured in either the current process environment or the launch env file.
- It does not source the env file and does not print token values.

## Evidence

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_scheduler_health.py::test_scheduler_health_reports_missing_launch_env_file_for_vercel_token \
  tests/test_scheduler_health.py::test_scheduler_health_reports_launch_env_vercel_token_without_leaking_secret -q
# red before implementation: Scheduler Private Env section missing
# green after implementation: 2 passed

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py -q
# 17 passed

uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_paper_trading.py tests/test_scheduler_health.py \
  tests/test_daily_scheduler.py tests/test_vercel_dashboard.py -q
# 54 passed

production/check_scheduler_health.sh
# Scheduler Private Env:
# launch env file: present path=/Users/rziiiii/.openclaw/soxl.env
# VERCEL_TOKEN: missing in current environment and launch env file
# Vercel Remote Deployment remains blocked_no_credentials
# runtime proof remains ok for selected v22 duplicate_skipped
```

## Decision

The current remote dashboard blocker is now narrowed to missing
scheduler-visible `VERCEL_TOKEN`, not a missing env file, missing Vercel CLI, or
stale dashboard evidence.

Keep the actual token outside the repo. After adding it to
`~/.openclaw/soxl.env` or logging in with the same scheduler user, rerun
`production/check_scheduler_health.sh` and
`PYTHONPATH=src:. qlib_venv/bin/python scripts/deploy_vercel_dashboard.py --deploy`.
