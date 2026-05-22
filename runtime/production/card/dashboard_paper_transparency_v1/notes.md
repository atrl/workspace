# Dashboard Paper Transparency V1

Date: 2026-05-22

## Goal

Fix the existing static dashboard production surface so it is clear which paper
trading data and model are being shown, and make the remote Vercel blocker
explicit.

## Findings

- Local static data is present and current through `2026-05-21`.
- The latest paper proof is non-dry and filled:
  `runtime/soxl/paper_trading/2026-05-21/summary.json`.
- The selected runtime model is
  `v19reg_v22_budget_usage_bull_guard_v1`.
- Remote Vercel deployment is blocked by missing credentials and project link,
  not by missing local dashboard data.

## Changes

- Added paper-trading `meta` to
  `dashboard_data/versioned/paper_trading/SOXL.json`.
- Added visible dashboard fields for `Paper Model`, `Paper Exec`,
  `Paper Source`, and `Paper End`.
- Cleaned the static dashboard UI toward a restrained white/blue/teal
  operational layout.
- Increased the main Plotly container height to match the generated layout so
  the chart no longer overlaps the following panels.
- Added scheduler `core done` marker support so goal-gate export and Vercel
  deployment evidence can be produced after the core scheduled work completes.
- Added an embedded paper payload fallback for `file://` or missing JSON fetch
  environments, so the production strip still shows paper model/execution/source
  when `dashboard_data/*.json` cannot be loaded by the browser.

## Current Snapshot

```text
paper_model=v19reg_v22_budget_usage_bull_guard_v1
paper_execution=filled
paper_source=2026-05-21 · target 77.5% · runtime/soxl/paper_trading/2026-05-21/summary.json
paper_end=2026-05-21
production_status=blocked_no_credentials
goal_gates=pending external_vercel_setup
```

## Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_daily_scheduler.py tests/test_scheduler_health.py \
  tests/test_dashboard.py tests/test_vercel_dashboard.py -q
# 76 passed

PYTHONPATH=. qlib_venv/bin/python scripts/check_dashboard_static_export.py
# ok dashboard_static_export generated_at=2026-05-22 14:41:54 versions=38 missing=0 model_versions=94 latest=paper_trading paper_trading_rows=5 paper_trading_end=2026-05-21 production_status=blocked_no_credentials goal_gates=pending paper_trading

production/check_scheduler_health.sh
# overall: pending external_vercel_setup
```

Rendered check:

- `runtime/production/card/dashboard_paper_transparency_v1/dashboard_verified.png`
- `runtime/production/card/dashboard_paper_transparency_v1/file_fallback_verified.png`

## Boundary

This clears local dashboard observability and static-export visibility. It does
not clear the remote Vercel gate. Remote data will remain invisible until
`VERCEL_TOKEN` or a Vercel login and project link are available.
