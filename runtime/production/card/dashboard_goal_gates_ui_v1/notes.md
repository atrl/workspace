# Dashboard Goal Gates UI

## Goal

Expose the same production goal blockers shown by
`production/check_scheduler_health.sh` inside the static dashboard bundle before
the Vercel deploy step runs.

## Root Cause

The scheduled pipeline rebuilt the dashboard and attempted Vercel deploy before
printing the health summary. The dashboard only loaded
`dashboard_data/production_status.json`, so the rendered Vercel surface showed
auth/project status but not the production blockers that explain why the full
goal is still pending.

## Change

- Added `scripts/export_production_goal_gates.py`.
- `production/soxl_daily_scheduler.py --vercel-deploy` now runs
  `production goal gates export` after dashboard build and before Vercel deploy.
- `dashboard.html` now loads `dashboard_data/production_goal_gates.json` and
  renders `Gates` plus `Blockers` in the production status strip.
- `scripts/check_dashboard_static_export.py` reports
  `goal_gates=<status>` when the artifact is present.

## Evidence

```bash
PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py \
  --smoke --vercel-deploy --health-summary --schedule-slot preopen
# production goal gates export ran before vercel dashboard deploy
# production_goal_gates status=pending issues=external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof
# vercel_dashboard status=blocked_no_credentials

PYTHONPATH=. qlib_venv/bin/python scripts/check_dashboard_static_export.py
# ok ... production_status=blocked_no_credentials goal_gates=pending paper_trading

PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py
# passed=195 failures=0

production/check_scheduler_health.sh
# dashboard static: ok ... production_status=blocked_no_credentials goal_gates=pending paper_trading
# overall: pending external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof
```

Rendered status card text on both desktop and mobile:

```text
PRODUCTION blocked_no_credentials
AUTH blocked_no_credentials
PROJECT missing_project_link
PAPER END 2026-05-20
DEPLOY none
GATES pending
BLOCKERS external_vercel_setup, scheduled_postclose_target, scheduled_paper_proof
```

Screenshots:

- `dashboard_goal_gates_desktop.png`
- `dashboard_goal_gates_mobile.png`

## Decision

This closes the local and deploy-bundle visibility gap. It does not clear the
remaining gates: Vercel credentials/project setup and natural scheduled
post-close/paper proof are still pending.
