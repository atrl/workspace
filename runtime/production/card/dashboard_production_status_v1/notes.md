# Dashboard Production Status V1

## Goal

Make the Vercel/static dashboard bundle carry a non-secret production status
artifact, so a deployed static site can expose whether the production path is
blocked by credentials, project link, data freshness, or deployment state.

## Root Cause

`scripts/deploy_vercel_dashboard.py` wrote Vercel readiness evidence under
`runtime/production/card/vercel_remote_deploy_v1/`, but the static dashboard
bundle itself only carried trading data under `dashboard_data/`. That meant a
remote Vercel deployment would serve the trading dashboard without a simple
machine-readable production status payload in the same data namespace.

## Change

- `scripts/deploy_vercel_dashboard.py`
  - writes `dashboard_data/production_status.json` on every Vercel readiness or
    deploy check;
  - includes only non-secret fields: status, auth/project status, deployment
    URL, static export summary, env-file presence, and env key names.
- `scripts/check_dashboard_static_export.py`
  - treats `dashboard_data/production_status.json` as a required static
    artifact;
  - validates required status fields and checks the production-status latest
    dashboard version matches `dashboard_data/index.json`;
  - prints `production_status=<status>` in the static bundle summary.
- `tests/test_vercel_dashboard.py`
  - covers secret-safe production-status export and static checker coverage.

## Current Evidence

```bash
PYTHONPATH=src:. qlib_venv/bin/python scripts/deploy_vercel_dashboard.py
# vercel_dashboard status=blocked_no_credentials output=runtime/production/card/vercel_remote_deploy_v1

PYTHONPATH=src:. qlib_venv/bin/python scripts/check_dashboard_static_export.py
# ok dashboard_static_export ... latest=paper_trading ... paper_trading_end=2026-05-20 production_status=blocked_no_credentials paper_trading
```

`dashboard_data/production_status.json` currently reports:

```text
status=blocked_no_credentials
auth_status=blocked_no_credentials
project_status=missing_project_link
deployment_url=null
static_export.latest=paper_trading
static_export.paper_trading_end=2026-05-20
env_file.present=true
env_file.applied_keys=[]
```

## Decision

This does not clear the remote Vercel gate. It makes the future Vercel/static
dashboard publish a redacted production-status payload once credentials and
project link are configured.
