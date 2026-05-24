# Dashboard Paper Signal Metadata V1

Date: 2026-05-24
Direction: production

## Goal

Expose the latest paper-trading inference on the Vercel/static dashboard after
the Futu `10000` reset.

## Root Cause

The latest Futu inference existed at:

```text
runtime/soxl/paper_trading/2026-05-22/dry_run/futu/summary.json
```

but `dashboard_data/versioned/paper_trading/SOXL.json` only carried the reset
seed CSV row. The dashboard builder correctly avoids attaching dry-run summaries
to ordinary paper rows, so the production strip had no `latest_profile` /
`latest_execution_status` fallback and Vercel could not show the inference.

## Current Model

```text
profile: v19reg_v22_budget_usage_bull_guard_v1
slow actor: runtime/model/card/balanced_rule_residual_v19_reg/soxl_balanced_rule_residual_v19_reg_actor.pt
fast actor: runtime/model/card/balanced_rule_residual_v22/soxl_balanced_rule_residual_v22_actor.pt
mode: trend_hold
broker: futu
dry_run: true
account_value: 10000.0
target_position: 0.8500000003
capped_target_position: 0.35
risk_gate: approved
```

## Change

- `scripts/build_backtest_dashboard.py` now exports `latest_signal_*` metadata
  from the latest paper summary artifact, including dry-run summaries.
- Ordinary paper rows are still only enriched from non-dry summaries.
- The production status strip falls back to `latest_signal_*` when the paper
  CSV row has no non-dry summary metadata.

## Evidence

```text
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_dashboard.py::test_load_versions_exports_latest_dry_run_signal_metadata_without_annotating_seed_row tests/test_dashboard.py::test_load_versions_enriches_paper_trading_row_with_latest_non_dry_summary_metadata -q
# 2 passed

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_dashboard.py tests/test_vercel_dashboard.py -q
# 33 passed

PYTHONPATH=. qlib_venv/bin/python scripts/build_backtest_dashboard.py
PYTHONPATH=. qlib_venv/bin/python scripts/export_production_goal_gates.py
PYTHONPATH=. qlib_venv/bin/python scripts/check_dashboard_static_export.py
# ok dashboard_static_export generated_at=2026-05-24 11:21:30 versions=38 missing=0 model_versions=94 latest=paper_trading paper_trading_rows=1 paper_trading_end=2026-05-22 production_status=blocked_no_credentials goal_gates=pending model_replays=10/10@2026-05-21 paper_trading
```

Visual evidence:

```text
runtime/production/card/dashboard_paper_signal_meta_v1/dashboard_verified.png
```

## Decision

Vercel/static dashboard visibility is now solved at the repo export layer. The
remaining external deployment gate is still Vercel credentials/project linkage,
not paper-trading inference generation.
