# Vercel Project Readiness

Date: 2026-05-21

## Goal

Expose the next non-secret Vercel deployment prerequisite before credentials are
added, so scheduled dashboard deployment does not move from one hidden blocker to
another.

## Root Cause

Vercel evidence and health already showed missing credentials, but they did not
show whether the workspace was linked to a Vercel project. Local inspection found
no `.vercel/project.json`, and the current shell had no `VERCEL_*` project env
ids.

The local Vercel CLI help for non-interactive linking requires:

```text
vercel link --yes --team <team-id> --project <project-name-or-id>
```

## Change

- `scripts/deploy_vercel_dashboard.py` now records a non-secret `project`
  readiness field in `vercel_deployment_check.json`.
- Project status is:
  - `linked` when `.vercel/project.json` contains project and org IDs
  - `env_configured` when `VERCEL_PROJECT_ID` and `VERCEL_ORG_ID` are set
  - `missing_project_link` when neither local link nor env IDs are present
  - `invalid_project_link` when `.vercel/project.json` is malformed or
    incomplete
- `production/check_scheduler_health.sh` now prints the project status in the
  Vercel Remote Deployment section.
- ID values are not printed in notes or health.

## Evidence

Red/green tests:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_vercel_dashboard.py::test_vercel_project_link_status_reports_missing_link_without_ids \
  tests/test_vercel_dashboard.py::test_vercel_project_link_status_accepts_local_project_json \
  tests/test_vercel_dashboard.py::test_vercel_project_link_status_accepts_project_env_ids \
  tests/test_scheduler_health.py::test_scheduler_health_reports_vercel_remote_deployment_status -q
# red before implementation: missing project_link_status; health did not print project status
# green after implementation: 4 passed
```

Evidence refresh:

```bash
PYTHONPATH=src:. qlib_venv/bin/python scripts/deploy_vercel_dashboard.py --deploy
# vercel_dashboard status=blocked_no_credentials output=runtime/production/card/vercel_remote_deploy_v1
# rc=2
```

Payload:

```bash
jq '{status:.status, auth:.auth.status, project:.project.status, deployment_url:.deployment_url, latest:.static_export.latest, paper_trading_end:.static_export.paper_trading_end}' \
  runtime/production/card/vercel_remote_deploy_v1/vercel_deployment_check.json
# status=blocked_no_credentials
# auth=blocked_no_credentials
# project=missing_project_link
# deployment_url=null
# latest=paper_trading
# paper_trading_end=2026-05-20
```

Health:

```bash
production/check_scheduler_health.sh
# Vercel Remote Deployment
# status: blocked_no_credentials
# auth: blocked_no_credentials
# project: missing_project_link
# deployment_url: none
# freshness: current ...
```

## Decision

Remote Vercel deployment is now narrowed to two explicit external setup steps:

1. make credentials visible to the scheduler user with `VERCEL_TOKEN` or
   `vercel login`
2. link or configure the project with `vercel link --yes --team <team-id>
   --project <project-name-or-id>` or `VERCEL_ORG_ID` plus `VERCEL_PROJECT_ID`

The local static dashboard bundle remains current and valid.
