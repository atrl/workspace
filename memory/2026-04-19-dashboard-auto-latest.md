---
date: 2026-04-19
title: 2026-04-19 dashboard auto latest fix
kind: decision
profiles:
- balanced_rule_residual_v1
- balanced_rule_residual_v1_soxl_daily
- soxl
status: kept
tags:
- balanced_rule_residual_v1
- balanced_rule_residual_v1_soxl_daily
- dashboard
- decision
- kept
- soxl
artifacts:
- runtime/execution/generalization/dashboard.html
- runtime/model/reviews/
- runtime/model/reviews/balanced_rule_residual_v1_soxl_daily.csv
workpad: memory/index.json
summary: Fixed the dashboard so training now auto-exports a review CSV, auto-rebuilds
  the static HTML, defaults to the latest post-train view instead of an older best-Sharpe
  or browser-saved selection, preserves explicit post-train trade actions, and merges
  OHLC/indicator fields back into post-train rows. The generated dashboard now resolves
  `latest_version` to `balanced_rule_residual_v1`.
---

# 2026-04-19 dashboard auto latest fix

## Context

The dashboard was not reliably showing the latest training result after a run finished.

## Evidence

- Added `src/soxl_quant/app/dashboard.py` to own:
  - post-train SOXL daily CSV export
  - dashboard rebuild invocation
- `run_profile()` now writes:
  - `runtime/model/reviews/<profile>_soxl_daily.csv`
- `persist_and_print_results()` now rebuilds:
  - `runtime/execution/generalization/dashboard.html`
- `scripts/build_backtest_dashboard.py` now:
  - computes `latest_version`
  - prefers `post_train_review` sources over historical `versions/*` snapshots
  - preserves explicit CSV `action` values instead of collapsing the whole non-flat span into one trade
  - merges `open/high/low/close/rsi6/hv_20/vol_coef/panic_score/market_state` into post-train rows
  - resets old non-manual browser state to latest through `soxl-dashboard-state-v9`

## Results

- Generated:
  - `runtime/model/reviews/balanced_rule_residual_v1_soxl_daily.csv`
  - `runtime/execution/generalization/dashboard.html`
- Verified generated HTML contains:
  - `"latest_version": "balanced_rule_residual_v1"`
  - `versionSource: 'auto_latest'`
- Verified latest residual view behavior:
  - `balanced_rule_residual_v1` now yields `98` paired trades in dashboard trade analysis
  - info panel fields that previously showed `n/a` on post-train rows now resolve from merged OHLC/indicator data
- Validation:
  - `tests/test_dashboard.py` passed
  - `scripts/check_harness.py => passed=35 failures=0`

## Decision

- Keep auto-rebuild and latest-view defaulting.
- Keep the dashboard skill updated with the post-train default-view contract.
