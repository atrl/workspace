# Paper Trading Dashboard Metadata

Date: 2026-05-21

Direction: production

## Goal

Make the static dashboard's `paper_trading` view identify the active runtime
profile and paper execution status for each simulated trading row, instead of
showing only curve fields from `paper_trading_soxl_daily.csv`.

## Root Cause

`runtime/soxl/paper_trading/paper_trading_soxl_daily.csv` only stores
dashboard-compatible time-series columns. After the runtime composer was
enabled, health could find the selected composer proof in:

```text
runtime/soxl/paper_trading/2026-05-20/duplicate_skip/v19reg_v22_budget_usage_bull_guard_v1/summary.json
```

but the dashboard `paper_trading` payload still lacked `profile`,
`execution_status`, and `target_position` fields for that date.

## Change

- `scripts/build_backtest_dashboard.py` now enriches paper-trading rows from
  non-dry per-date paper summaries.
- For same-date duplicates, `duplicate_skip/<profile>/summary.json` outranks
  the top-level executed summary, so the current runtime profile remains visible
  even when same-date idempotency prevents another order.
- Dry-run summaries are not attached to ordinary paper rows, preventing
  pre-open dry-run evidence from being confused with non-dry paper proof.
- The dashboard info panel now has `Profile`, `Execution`, and `Target` fields.

## Evidence

TDD regression:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_dashboard.py -q -k 'paper_trading_row'
# red before fix: KeyError: 'profile'
# green after fix: 1 passed, 12 deselected
```

Broader dashboard regression:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_vercel_dashboard.py tests/test_dashboard.py -q
# 27 passed
```

Static export:

```bash
PYTHONPATH=. qlib_venv/bin/python scripts/build_backtest_dashboard.py
PYTHONPATH=. qlib_venv/bin/python scripts/check_dashboard_static_export.py
# ok dashboard_static_export generated_at=2026-05-21 21:32:58 versions=38 missing=0 model_versions=94 latest=paper_trading paper_trading_rows=4 paper_trading_end=2026-05-20 production_status=blocked_no_credentials goal_gates=pending paper_trading
```

Rendered/local browser evidence:

```text
latest=paper_trading
paper_profile=v19reg_v22_budget_usage_bull_guard_v1
paper_execution_status=duplicate_skipped
paper_target_position=0.48408568215209924
info_ids={'profile': True, 'execution': True, 'target': True, 'chart': True}
production_strip=PRODUCTION | blocked_no_credentials | AUTH | blocked_no_credentials | PROJECT | missing_project_link | PAPER END | 2026-05-20 | DEPLOY | none | GATES | pending | BLOCKERS | external_vercel_setup, scheduled_postclose_target, scheduled_paper_proof
```

Current payload row:

```text
2026-05-20 v19reg_v22_budget_usage_bull_guard_v1 duplicate_skipped False
```

## Boundary

This fixes dashboard/paper observability only. It does not clear the ordinary
paper proof gate because the latest selected-model non-dry evidence is still a
same-date `duplicate_skipped` result. The next natural scheduled post-close run
must produce a new-date executed or skipped non-dry paper result.
