# Vercel Health Freshness

Date: 2026-05-21

## Goal

Expose whether the latest recorded Vercel deployment attempt corresponds to the
current local static dashboard bundle.

## Root Cause

The local dashboard can be rebuilt after the last Vercel deploy attempt. Before
this change, `production/check_scheduler_health.sh` reported Vercel auth/deploy
status but did not compare `dashboard_data/index.json.generated_at` with
`runtime/production/card/vercel_remote_deploy_v1/vercel_deployment_check.json`.

The observed stale state was:

- local dashboard: `generated_at=2026-05-21 12:01:45`, `versions=35`
- Vercel evidence: `static_export.generated_at=2026-05-21 11:07:30`,
  `versions=28`
- Vercel auth: `blocked_no_credentials`

## Change

- Health prints `freshness: stale ...` when local dashboard output is newer
  than the recorded Vercel deployment check.
- Health prints `freshness: current ...` when the recorded Vercel deployment
  check matches the current local dashboard output.
- The Vercel deployment evidence was refreshed with the repo-owned deploy check;
  it remains blocked by missing credentials, but now references the current
  static export.

## Evidence

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_scheduler_health.py::test_scheduler_health_reports_stale_vercel_evidence_when_dashboard_is_newer -q
# red before implementation: no freshness line
# green after implementation: 1 passed

uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_scheduler_health.py -q
# 14 passed

PYTHONPATH=src:. qlib_venv/bin/python scripts/deploy_vercel_dashboard.py --deploy
# expected rc=2; status=blocked_no_credentials; evidence refreshed

production/check_scheduler_health.sh
# Vercel status: blocked_no_credentials
# freshness: current local_generated_at=2026-05-21 12:01:45 evidence_generated_at=2026-05-21 12:01:45
```

## Remaining Gates

- Remote deployment still requires `VERCEL_TOKEN` or a `vercel login` session
  visible to the scheduler user.
- Selected `balanced_rule_residual_v22` non-dry paper runtime proof still waits
  for the next fresh post-close paper run.
