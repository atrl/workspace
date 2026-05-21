# Vercel Health Surface

## Context

The dashboard static bundle was already checked, but
`production/check_scheduler_health.sh` only reported local static export status.
Remote deployment evidence lived in a separate card, so the production health
surface did not show whether the Vercel URL was actually deployed or blocked.

## Change

- `production/check_scheduler_health.sh` now reads
  `runtime/production/card/vercel_remote_deploy_v1/vercel_deployment_check.json`.
- It prints a `Vercel Remote Deployment` section with deployment status, auth
  status, deployment URL, dashboard latest version, paper-trading end date, and
  the evidence path.
- The status remains observability-only; the health script does not fail merely
  because remote deployment is blocked by missing credentials.

## Evidence

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py::test_scheduler_health_reports_vercel_remote_deployment_status -q
# 1 passed after initially failing on the missing Vercel section

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py -q
# 10 passed

production/check_scheduler_health.sh
# --- Vercel Remote Deployment ---
#   status: blocked_no_credentials
#   auth: blocked_no_credentials
#   deployment_url: none
#   dashboard_latest=paper_trading paper_trading_end=2026-05-20
```

## Decision

Vercel deployment is still not proven remotely because the current machine has
no Vercel credentials. The blocker is now visible in the canonical production
health output.
