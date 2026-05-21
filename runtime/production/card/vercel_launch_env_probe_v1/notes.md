# Vercel Launch Env Probe

Date: 2026-05-21

## Goal

Make Vercel readiness checks use the same private LaunchAgent env file that the
scheduler wrapper uses, without leaking token, project, or org values.

## Root Cause

`production/run_soxl_scheduler.sh` sources
`${SOXL_LAUNCH_ENV_FILE:-$HOME/.openclaw/soxl.env}` before it runs the daily
scheduler. `scripts/deploy_vercel_dashboard.py` only inspected the current
process environment. That meant a direct readiness check could still report
`blocked_no_credentials` after an operator added Vercel values to the scheduler
env file but did not export them into the current shell.

## Change

- `scripts/deploy_vercel_dashboard.py` now accepts `--env-file` and defaults to
  `${SOXL_LAUNCH_ENV_FILE:-$HOME/.openclaw/soxl.env}`.
- It parses simple `export KEY=value` / `KEY=value` assignments for
  `VERCEL_TOKEN`, `VERCEL_PROJECT_ID`, and `VERCEL_ORG_ID`.
- Current process env remains higher priority; the env file fills missing
  values.
- JSON and Markdown evidence record only the env file presence and applied
  variable names, not values.
- `production/check_scheduler_health.sh` now prints the Vercel evidence
  env-file status in `Vercel Remote Deployment`.

## Evidence

Red/green tests:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_vercel_dashboard.py::test_vercel_check_reads_launch_env_file_for_readiness_without_leaking \
  tests/test_scheduler_health.py::test_scheduler_health_reports_vercel_remote_deployment_status -q
# red before implementation: --env-file was unrecognized and health did not print env_file status
# green after implementation: 2 passed
```

Full focused Vercel/health suite:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_vercel_dashboard.py tests/test_scheduler_health.py -q
# 31 passed
```

Real readiness check:

```bash
PYTHONPATH=src:. qlib_venv/bin/python scripts/deploy_vercel_dashboard.py
# vercel_dashboard status=blocked_no_credentials output=/Users/rziiiii/.openclaw/workspace/runtime/production/card/vercel_remote_deploy_v1
```

Real health:

```bash
production/check_scheduler_health.sh | sed -n '/--- Vercel Remote Deployment ---/,/--- Post-close Run Status ---/p'
# status: blocked_no_credentials
# auth: blocked_no_credentials
# project: missing_project_link
# env_file: present applied_keys=none
# deployment_url: none
```

## Decision

The Vercel blocker is now checked against the same private env file used by the
scheduled pipeline. Current state remains externally blocked because the file
contains no Vercel keys:

- add `VERCEL_TOKEN`
- add `VERCEL_PROJECT_ID`
- add `VERCEL_ORG_ID`

After those are present, a non-deploy readiness check should move from
`blocked_no_credentials` / `missing_project_link` to `ready_for_deploy`; the
scheduled deploy attempt can then publish the dashboard.
