# Vercel Scheduled Deploy Integration

Date: 2026-05-21

## Goal

Make the existing Vercel dashboard deployment check part of the scheduled
dashboard update path, so the remote dashboard can update automatically once
Vercel credentials are available.

## Change

- Added `--vercel-deploy` to `production/soxl_daily_scheduler.py`.
- After dashboard rebuild, the scheduler can run:

```bash
qlib_venv/bin/python scripts/deploy_vercel_dashboard.py --deploy
```

- Missing Vercel credentials return code `2` is treated as non-fatal by default
  because paper trading and dashboard generation should remain healthy while the
  remote deployment is blocked.
- Added `--strict-vercel-deploy` for operator runs that should fail on missing
  credentials.
- Added `--vercel-deploy` to the repo-owned pre-open and post-close
  LaunchAgent plists.
- Updated `production/crontab.txt` daily SOXL entries to use
  `production/run_soxl_scheduler.sh` under the current repo path instead of the
  stale `/home/rziiiii/dev/workspace` path, and to pass `--vercel-deploy`.
- Synced and reloaded the local LaunchAgent copies under
  `~/Library/LaunchAgents/`.

## Live Smoke

Command:

```bash
PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --broker paper --dry-run --no-notify --skip-data-update --vercel-deploy
```

Readout:

- Auto-selected profile: `balanced_rule_residual_v22`.
- Signal date: `2026-05-20`.
- Output path:
  `runtime/soxl/paper_trading/2026-05-20/dry_run`.
- Dashboard rebuilt with `latest=paper_trading`, `versions=27`,
  `model_versions=93`, `paper_trading_end=2026-05-20`.
- Vercel deployment evidence refreshed at:
  `runtime/production/card/vercel_remote_deploy_v1/vercel_deployment_check.json`.
- Deployment status remains `blocked_no_credentials`, with
  `deploy_requested=true`.
- `production/check_scheduler_health.sh` reports both scheduled dashboard
  windows as `+ Vercel deploy attempt`, and installed LaunchAgent plists match
  the repo copies.

## Decision

The scheduled dashboard path is now wired for Vercel deployment. Remote
deployment remains blocked until `VERCEL_TOKEN` is available to the scheduler
environment or the local Vercel CLI is logged in.
