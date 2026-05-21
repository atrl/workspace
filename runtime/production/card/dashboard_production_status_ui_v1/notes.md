# Dashboard Production Status UI V1

Date: 2026-05-21
Direction: production

## Goal

Expose the existing redacted Vercel/production status artifact on the static
dashboard page so the deployed dashboard can show operational status in the
first viewport.

## Change

- `scripts/build_backtest_dashboard.py`
  - adds a compact production status strip above the dashboard metric cards;
  - loads `dashboard_data/production_status.json` with the existing async JSON
    loader;
  - displays production, auth, project, paper-trading end date, and deployment
    URL/status;
  - renders `unavailable` if the status artifact cannot be loaded.
- `dashboard.html`
  - regenerated from the canonical builder.
- Tests:
  - `tests/test_dashboard.py::test_build_html_loads_production_status_json`
  - `tests/test_vercel_dashboard.py::test_dashboard_page_loads_production_status_export`

## Evidence

```bash
PYTHONPATH=. qlib_venv/bin/python -c "import importlib.util, pathlib; p=pathlib.Path('tests/test_vercel_dashboard.py'); spec=importlib.util.spec_from_file_location('tv', p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); m.test_dashboard_page_loads_production_status_export()"
# RED before rebuild: AssertionError, committed dashboard.html had no productionStatusCard
# GREEN after rebuild: exit 0

PYTHONPATH=. qlib_venv/bin/python -c "import importlib.util, pathlib; p=pathlib.Path('tests/test_dashboard.py'); spec=importlib.util.spec_from_file_location('td', p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); m.test_build_html_loads_production_status_json()"
# GREEN after implementation: exit 0

PYTHONPATH=. qlib_venv/bin/python scripts/build_backtest_dashboard.py
# wrote dashboard.html, dashboard_data/index.json, dashboard_data/model_versions.json, dashboard_data/versioned/ (35 versions)

PYTHONPATH=. qlib_venv/bin/python scripts/check_dashboard_static_export.py
# ok dashboard_static_export generated_at=2026-05-21 17:13:12 versions=35 missing=0 model_versions=94 latest=paper_trading paper_trading_rows=4 paper_trading_end=2026-05-20 production_status=blocked_no_credentials paper_trading

rg -n "token_secretish|project_secretish|team_secretish|VERCEL_TOKEN=|VERCEL_PROJECT_ID=|VERCEL_ORG_ID=" dashboard.html dashboard_data/production_status.json || true
# no output

PYTHONPATH=src:. qlib_venv/bin/python scripts/deploy_vercel_dashboard.py
# vercel_dashboard status=blocked_no_credentials output=/Users/rziiiii/.openclaw/workspace/runtime/production/card/vercel_remote_deploy_v1

production/check_scheduler_health.sh
# dashboard static: ok dashboard_static_export generated_at=2026-05-21 17:13:12 versions=35 missing=0 model_versions=94 latest=paper_trading paper_trading_rows=4 paper_trading_end=2026-05-20 production_status=blocked_no_credentials paper_trading
# Vercel remote: blocked status=blocked_no_credentials auth=blocked_no_credentials project=missing_project_link deployment_url=none env_file_applied_keys=none
# freshness: current local_generated_at=2026-05-21 17:13:12 evidence_generated_at=2026-05-21 17:13:12
# overall: pending external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof

PYTHONPATH=. qlib_venv/bin/python scripts/build_runtime_index.py
# wrote runtime/index.json (512 entries)

PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py
# [summary] passed=191 failures=0
```

Visual verification:

- Desktop: `runtime/production/card/dashboard_production_status_ui_v1/dashboard_status_top_20260521.png`
- Mobile: `runtime/production/card/dashboard_production_status_ui_v1/dashboard_status_mobile_20260521.png`

Both screenshots show:

- `Production: blocked_no_credentials`
- `Auth: blocked_no_credentials`
- `Project: missing_project_link`
- `Paper End: 2026-05-20`
- `Deploy: none`

## Remaining Gates

This is a dashboard visibility improvement only. The production goal remains
pending on:

- scheduler-visible Vercel token and project config;
- natural post-close run with `schedule_slot=postclose`;
- next new-date non-dry paper composer artifact.
