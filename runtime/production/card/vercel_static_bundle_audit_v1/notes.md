# Vercel Static Dashboard Bundle Audit

Date: 2026-05-21

## Scope

Audit whether the SOXL dashboard has a repo-owned static bundle suitable for
Vercel serving.

## Current Bundle

Required root files:

- `vercel.json`
- `dashboard.html`
- `dashboard_data/index.json`
- `dashboard_data/model_versions.json`
- `dashboard_data/versioned/*/SOXL.json`

Current generated bundle:

- versions: `25`
- version payloads: `25`
- model versions: `90`
- latest version: `paper_trading`
- paper-trading rows: `4`
- paper-trading end date: `2026-05-20`

## Vercel Routing

`vercel.json` serves the static dashboard through:

```json
{"source": "/", "destination": "/dashboard.html"}
```

No-cache headers are configured for:

- `/`
- `/dashboard.html`
- `/dashboard_data/:path*`

This keeps the static dashboard and JSON payloads from being pinned after the
scheduled dashboard rebuild updates the tracked files.

## Evidence Commands

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_vercel_dashboard.py -q
# 3 passed

PYTHONPATH=src:. qlib_venv/bin/python scripts/check_dashboard_static_export.py
# ok dashboard_static_export generated_at=2026-05-21 06:51:30 versions=25 missing=0 model_versions=90 latest=paper_trading paper_trading_rows=4 paper_trading_end=2026-05-20 paper_trading
```

The generated `dashboard_data/index.json` also reports
`latest_version=paper_trading`, so the deployed first-load view is the
simulated paper loop rather than an offline SOXX experiment snapshot.

## Boundary

This audit proves the repo-owned static bundle and Vercel configuration are
ready to serve. It does not prove a remote Vercel deployment URL is live, because
no deployment command or Vercel project state is stored in this repository.
