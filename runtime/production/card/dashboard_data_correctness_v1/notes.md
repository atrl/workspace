# Dashboard Data Correctness

Date: 2026-05-23

Direction: production

## Goal

Fix misleading dashboard trade/price data in the static/Vercel bundle and remove
the unusable historical model comparison / Training Curve UI.

## Root Cause

- `normalize_post_train_review_rows()` forced the terminal row to `SELL` when a
  run ended with a positive position and `HOLD`. This changed the current
  paper row on `2026-05-22` from `HOLD` to `SELL`.
- The paper CSV still contained an old seed row on `2026-03-31` without a
  non-dry paper summary. That row made the paper view start from an old price
  and caused Buy&Hold NAV to use a stale baseline.
- `compact_payload_rows()` dropped `open/high/low/close` and related market
  fields after they had been merged. Static version payloads therefore could
  not render a real SOXL price chart after async loading.
- The visible historical model comparison and Training Curve sections were based
  on stale summary metadata and were not useful for current static inspection.

## Change

- Stopped synthesizing terminal sells.
- Suppressed impossible explicit actions such as `BUY` with zero position and
  `SELL` from a zero-position state.
- For paper trading views, keep only rows that have non-dry summary metadata
  when such metadata exists, and rebase Buy&Hold NAV to the first displayed
  paper account value.
- Preserved OHLC, volume, MA180, and dashboard indicators in versioned static
  payload rows.
- Added a price-row fallback that plots close/price as a line when OHLC is not
  available.
- Removed the visible Model Version Comparison and Training Curve UI while
  retaining model version JSON for static export integrity checks.

## Evidence

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_dashboard.py tests/test_vercel_dashboard.py -q
# 32 passed

PYTHONPATH=. qlib_venv/bin/python scripts/build_backtest_dashboard.py
# wrote dashboard.html, dashboard_data/index.json, dashboard_data/model_versions.json, and 38 versioned exports

PYTHONPATH=. qlib_venv/bin/python scripts/check_dashboard_static_export.py
# ok dashboard_static_export generated_at=2026-05-23 12:18:13 versions=38 missing=0 model_versions=94 latest=paper_trading paper_trading_rows=4 paper_trading_end=2026-05-22 production_status=blocked_no_credentials goal_gates=pending model_replays=10/10@2026-05-21 paper_trading
```

Rendered verification through local HTTP + Playwright:

```text
checks={'has_training_curve': False, 'has_model_version_comparison': False, 'paper_end_visible': True}
price trace: SOXL candlestick n=4 on y2
paper rows: 4, 2026-05-18 -> 2026-05-22
first paper buyhold_nav == strategy_nav
last paper action: HOLD
screenshot_nonwhite_pct=39.14
```

Screenshot:

- `runtime/production/card/dashboard_data_correctness_v1/dashboard_data_correctness_v1.png`

## Boundary

This fixes the local/static dashboard data bundle. Vercel remote deployment still
reports `blocked_no_credentials` until `VERCEL_TOKEN` and project/org settings
are configured for non-interactive deploys.
