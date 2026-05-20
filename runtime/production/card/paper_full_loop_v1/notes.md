# Paper Full Loop v1

## Context

After the Polygon + CBOE provider recovery, the data plane was fresh for the
current completed US trading day, but the local simulated paper account still
needed proof that the full scheduler path could mutate state and feed the
dashboard.

Baseline at `2026-05-21 03:01 CST`:

- `production/check_scheduler_health.sh`: `data: ok last_date=2026-05-19 expected=2026-05-19`
- `paper_broker_state.json`: dated `2026-05-14`, `1800` shares, `33.53%` position
- `paper_trading_soxl_daily.csv`: `2` rows, ending `2026-05-14`
- `dashboard_data/index.json`: paper version ending `2026-05-14` with `2` rows

## Command

```bash
PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --no-notify --data-provider auto
```

## Readout

- yfinance failed with `No overlapping dates for SOXL/SOXX/VIX`.
- yahoo_api failed with Yahoo HTTP `429` on `^VIX`.
- polygon succeeded with `last_date=2026-05-19`.
- freshness gate passed: `last_date=2026-05-19 expected=2026-05-19 stale_days=0`.
- model: `balanced_rule_residual_v19`.
- signal date: `2026-05-18`.
- target: `50.00%`.
- price: `$151.75`.
- risk gate: approved.
- paper order: submitted and filled.
- dashboard: rebuilt.

## Artifacts

- `runtime/soxl/paper_trading/paper_broker_state.json`
  - date: `2026-05-18`
  - shares: `3000`
  - position: `0.4856047894`
  - account value: `937490.7536`
- `runtime/soxl/paper_trading/2026-05-18/summary.json`
  - `dry_run=false`
  - `executed=true`
  - `risk_gate.approved=true`
- `runtime/soxl/paper_trading/2026-05-18/reconcile.json`
  - expected position: `0.5`
  - actual position: `0.4856047894`
  - action required: `none`
  - note: `position aligned`
- `runtime/soxl/paper_trading/paper_trading_soxl_daily.csv`
  - rows: `3`
  - latest row: `2026-05-18`, action `BUY`, position `0.485605`
- `dashboard_data/index.json`
  - paper version: `start=2026-03-31`, `end=2026-05-18`, `rows=3`

## Dashboard Verification

```bash
curl -I http://127.0.0.1:8770/dashboard.html
curl -I http://127.0.0.1:8770/dashboard_data/index.json
curl -I http://127.0.0.1:8770/dashboard_data/model_versions.json
# all returned 200 OK

npx -y playwright screenshot --channel chrome --viewport-size=1440,1000 --wait-for-timeout=2000 \
  'http://127.0.0.1:8770/dashboard.html?ts=paper-full-loop-v1' \
  runtime/production/card/paper_full_loop_v1/dashboard_visual_2026-05-21.png
```

Screenshot artifact:

- `runtime/production/card/paper_full_loop_v1/dashboard_visual_2026-05-21.png`
  - size: `1440 x 1000`
  - file size: `272784` bytes
  - non-white pixels: `99.35%`

## Decision

The manual full scheduler path proves the local simulated paper broker can now
advance under fresh data and publish the result into the dashboard data package.
The remaining production proof is the natural `05:35 Tue-Sat Asia/Shanghai`
LaunchAgent run after the next completed US session.
