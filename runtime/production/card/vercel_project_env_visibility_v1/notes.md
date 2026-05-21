# Vercel Project Env Visibility

Date: 2026-05-21

## Goal

Make scheduler health show whether the LaunchAgent environment already contains
Vercel project IDs, without printing project or org ID values.

## Root Cause

Remote deployment readiness has two separate project-link paths:

- local `.vercel/project.json`
- non-interactive environment IDs: `VERCEL_PROJECT_ID` and `VERCEL_ORG_ID`

`Vercel Remote Deployment` already reports the deployment script's project
status, but `Scheduler Private Env` only showed token visibility. If an
operator adds project IDs to `${SOXL_LAUNCH_ENV_FILE:-$HOME/.openclaw/soxl.env}`
without exporting them into the current shell, health should make that visible
before the next scheduled deploy attempt.

## Change

- `production/check_scheduler_health.sh` now prints a non-secret
  `VERCEL_PROJECT` line in `Scheduler Private Env`.
- It reports:
  - `configured in current environment`
  - `configured in launch env file`
  - `configured across current environment and launch env file`
  - `missing VERCEL_PROJECT_ID or VERCEL_ORG_ID in current environment and launch env file`
- Project/org ID values are not printed.

## Evidence

Red/green focused tests:

```bash
PYTHONPATH=src:. qlib_venv/bin/python -c "import importlib.util; spec=importlib.util.spec_from_file_location('test_scheduler_health', 'tests/test_scheduler_health.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); mod.test_scheduler_health_reports_launch_env_vercel_project_without_leaking_ids()"
# red before implementation: missing VERCEL_PROJECT line
# green after implementation: exit 0
```

Live health:

```bash
production/check_scheduler_health.sh
# VERCEL_TOKEN: missing in current environment and launch env file
# VERCEL_PROJECT: missing VERCEL_PROJECT_ID or VERCEL_ORG_ID in current environment and launch env file
```

## Decision

The remaining Vercel remote-deploy setup is now visible at both layers:

1. `Scheduler Private Env` shows whether the scheduler user can see token and
   project env IDs.
2. `Vercel Remote Deployment` shows the deployment script's auth/project result
   and remote URL status.
