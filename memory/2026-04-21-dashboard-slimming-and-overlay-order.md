---
date: '2026-04-21'
title: 2026-04-21 dashboard slimming and next-step overlay order
kind: decision
profiles:
- balanced
- balanced_rule_residual_v19
- balanced_rule_residual_v22
- balanced_rule_residual_v23
status: active
tags:
- active
- balanced
- balanced_rule_residual_v19
- balanced_rule_residual_v22
- balanced_rule_residual_v23
- dashboard
- decision
- execution-overlay
- stress-regime
- transition-features
artifacts:
- evidence/generalization/dashboard.html
workpad: codex_exec_plans/completed/2026-04/2026-04-10-soxl-agent-loop.md
summary: Slimmed `evidence/generalization/dashboard.html` by changing historical `versioned`
  payloads to carry only compact trading-series rows and reusing current per-ETF market
  metadata client-side by date. Dashboard size fell from 50.54 MB to 20.41 MB. Mainline
  optimization order remains fixed-actor execution/risk overlay before any new retraining
  branch.
---

# 2026-04-21 dashboard slimming and next-step overlay order

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

## Dashboard result

- Previous `evidence/generalization/dashboard.html`: `50.54 MB`
- New `evidence/generalization/dashboard.html`: `20.41 MB`
- Mechanism:
  - keep current `detailed` rows rich
  - strip historical `versioned` rows down to trading-series essentials
  - hydrate OHLC / indicator / feature metadata in the browser from the current
    ETF base rows using `datetime`

## Why this matters

- The previous dashboard triggered a GitHub large-file warning on push.
- The waste was mostly in duplicated per-day market metadata across historical
  version payloads, not in the current detailed view.
- Static dashboard architecture is still kept; no service-backed frontend was
  added.

## Validation

```bash
PYTHONPATH=. qlib_venv/bin/python tests/test_dashboard.py
PYTHONPATH=. qlib_venv/bin/python -m py_compile scripts/build_backtest_dashboard.py tests/test_dashboard.py
PYTHONPATH=. qlib_venv/bin/python scripts/build_backtest_dashboard.py
PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py
```

- `tests/test_dashboard.py`: passed
- dashboard rebuild: passed
- `scripts/check_harness.py`: `passed=37 failures=0`

## Next experiment order

- Do **not** continue with another transition-state retrain immediately after
  `balanced_rule_residual_v23`.
- Next loop should stay in fixed-actor execution / risk overlay space.
- Direct anchor:
  - `balanced_rule_residual_v19`
- Comparison rails:
  - `balanced`
  - `balanced_rule_residual_v22`
- Desired mechanism:
  - selective stress-release re-entry through overlay routing
  - not generic broad de-risking

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
