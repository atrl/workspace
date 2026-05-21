# Vercel Placeholder Guard V1

Date: 2026-05-21

## Goal

Prevent placeholder Vercel values from being treated as configured credentials
or project IDs.

## Root Cause

The Vercel readiness probe and scheduler health check treated any non-empty
`VERCEL_TOKEN`, `VERCEL_PROJECT_ID`, or `VERCEL_ORG_ID` value as configured.
That means operator-doc placeholders such as `REPLACE_ME` or `replace-me` could
create a false readiness signal if accidentally uncommented without replacing
the value.

## Change

- `scripts/deploy_vercel_dashboard.py` now ignores placeholder values when
  loading Vercel variables from the LaunchAgent env file.
- The deploy probe also treats placeholder values in the current process env as
  missing for auth/project readiness.
- `production/check_scheduler_health.sh` now reports Vercel token/project env
  as missing when the only configured values are placeholders.

## Evidence

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_vercel_dashboard.py::test_vercel_env_file_ignores_placeholder_values \
  tests/test_scheduler_health.py::test_scheduler_health_treats_vercel_placeholders_as_missing -q
# red before implementation: REPLACE_ME / replace-me were treated as configured
# green after implementation: 2 passed

PYTHONPATH=src:. qlib_venv/bin/python scripts/deploy_vercel_dashboard.py
# vercel_dashboard status=blocked_no_credentials output=/Users/rziiiii/.openclaw/workspace/runtime/production/card/vercel_remote_deploy_v1

production/check_scheduler_health.sh
# VERCEL_TOKEN: missing in current environment and launch env file
# VERCEL_PROJECT: missing VERCEL_PROJECT_ID or VERCEL_ORG_ID in current environment and launch env file
# overall: pending external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof
```

## Decision

The Vercel gate now requires real-looking non-placeholder values before it can
move from external setup blocker to deploy readiness. The live production state
remains:

```text
overall: pending external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof
```
