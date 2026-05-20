# Polygon + CBOE Provider v1

## Context

The production loop still had a data-freshness blocker after `yfinance` and
`yahoo_api` failover was wired into the scheduler. At `2026-05-21 02:4x CST`:

- `yfinance` dry-run failed with `No overlapping dates for SOXL/SOXX/VIX`.
- `yahoo_api` dry-run failed with Yahoo HTTP `429 Too Many Requests`.
- Futu OpenD was not reachable at `127.0.0.1:11111`.
- Polygon returned SOXL/SOXX daily bars through `2026-05-19`.
- Polygon VIX was not authorized, but CBOE's public VIX history CSV returned VIX
  through `2026-05-19`.

## Change

- Added `polygon` to `scripts/update_market_data.py`.
  - SOXL/SOXX: Polygon daily aggregates via `POLYGON_API_KEY`.
  - VIX: CBOE public VIX history CSV.
- Added `polygon` to production provider choices.
- Changed production `auto` provider order to:
  `yfinance -> yahoo_api -> polygon`.
- Changed scheduler `_run()` to replay child stderr even when the child exits
  successfully, so provider fallback messages remain visible in scheduler logs.

## Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_update_market_data.py::test_update_master_csv_polygon_provider_uses_cboe_vix tests/test_paper_trading.py::test_data_update_auto_provider_falls_back_to_polygon tests/test_daily_scheduler.py::test_scheduler_accepts_polygon_data_provider -q
# 3 passed after first failing against the old implementation

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_daily_scheduler.py::test_run_replays_child_stderr_on_success -q
# 1 passed after first failing because child stderr was hidden on success

PYTHONPATH=. qlib_venv/bin/python scripts/update_market_data.py --provider polygon --dry-run --skip-qlib
# before=4071 after=4073 new_rows=2 last_date=2026-05-19

PYTHONPATH=. qlib_venv/bin/python scripts/update_market_data.py --provider polygon
# wrote canonical panel and refreshed raw/qlib surfaces to 2026-05-19

PYTHONPATH=. qlib_venv/bin/python scripts/update_data_registry.py
# update_data_registry: OK symbols=6 panels=10 qlib=6

production/check_scheduler_health.sh
# data: ok last_date=2026-05-19 expected=2026-05-19
# model: ok balanced_rule_residual_v19 (soxl_balanced_rule_residual_v19_actor.pt)
# broker: ok paper broker ready

PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --dry-run --no-notify --data-provider auto
# yfinance failed, yahoo_api failed, polygon succeeded
# data fresh: last_date=2026-05-19 expected=2026-05-19 stale_days=0
# as_of=2026-05-19, signal date=2026-05-18, risk_gate approved=True
# dry-run skipped order placement and rebuilt dashboard_data with 25 versions

curl -I http://127.0.0.1:8769/dashboard.html
curl -I http://127.0.0.1:8769/dashboard_data/index.json
curl -I http://127.0.0.1:8769/dashboard_data/model_versions.json
# all returned 200 OK

npx -y playwright screenshot --channel chrome --viewport-size=1440,1000 --wait-for-timeout=2000 \
  'http://127.0.0.1:8769/dashboard.html?ts=polygon-cboe-v1' \
  runtime/production/card/polygon_cboe_provider_v1/dashboard_visual_2026-05-21.png
# captured screenshot using local Chrome
```

## Readout

The production data plane is no longer stale for the current completed US
trading day. The live schedule still needs the natural `05:35 Tue-Sat
Asia/Shanghai` LaunchAgent run to prove the post-close non-dry path after the
next completed session.

Screenshot artifact:

- `runtime/production/card/polygon_cboe_provider_v1/dashboard_visual_2026-05-21.png`
  - size: `1440 x 1000`
  - file size: `272554` bytes
  - non-white pixels: `99.35%`
