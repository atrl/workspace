# Futu 10000 Reset v1

Date: 2026-05-24

## Goal

Reset the active Futu simulated-trading path so the repo-side simulation starts
from a `10000` notional account instead of sizing from the Futu backend
simulated account balance of `1000000`.

## Boundary

- Futu OpenAPI does not expose an account-cash reset endpoint in the installed
  SDK surface.
- The repo now controls simulation sizing with `--account-notional 10000`.
- Futu remains the scheduled broker; IBKR is not used.
- No non-dry Futu order was placed during this reset loop.

## Code Changes

- `src/soxl_quant/app/futu.py`
  - Added `account_notional` to `FutuBroker`.
  - `get_account_snapshot()` reports notional account value when configured.
  - `place_target_position()` sizes share quantity from that notional snapshot.
  - Removed the unsupported `trd_market` keyword from `place_order()`.
- `production/soxl_paper_trading.py`
  - Added `--account-notional`.
  - Enables existing step clipping for explicit Futu notional simulation so a
    cold-start account can move up to the `35%` runtime step limit instead of
    blocking forever.
- `production/soxl_daily_scheduler.py`
  - Added `--account-notional` and forwards it to the paper runner.
- `production/com.soxl.daily.scheduler.plist`,
  `production/com.soxl.preopen.scheduler.plist`, and `production/crontab.txt`
  - Scheduled Futu runs now include `--account-notional 10000`.
- `production/check_scheduler_health.sh`
  - Reports the configured notional and forwards it to paper healthcheck.

## Local State Reset

The old generated paper-trading state was preserved under:

```text
runtime/soxl/paper_trading/archive_reset_20260524_10000
```

The active local state now starts from:

```text
runtime/soxl/paper_trading/paper_broker_state.json
cash=10000.0
shares=0
account_value=10000.0
position=0.0
date=2026-05-22
```

The active dashboard CSV now starts from one baseline row:

```text
datetime=2026-05-22
strategy_nav=10000.0
price=190.559998
buyhold_nav=10000.0
position=0.0
action=HOLD
```

## Evidence

Futu notional dry-run:

```bash
PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py \
  --broker futu --account-notional 10000 --dry-run --skip-data-update \
  --skip-dashboard --no-notify --schedule-slot manual \
  --runtime-composer budget_usage \
  --composer-profile v19reg_v22_budget_usage_bull_guard_v1 \
  --slow-profile balanced_rule_residual_v19_reg \
  --fast-profile balanced_rule_residual_v22 \
  --bull-participation-guard
```

Readout:

```text
account.account_value=10000.0
account.cash=10000.0
account.shares=0
signal.target_position=0.8500000003
risk_gate.approved=true
risk_gate.capped_target_position=0.35
dry_run=true
executed=false
output=runtime/soxl/paper_trading/2026-05-22/dry_run/futu
```

LaunchAgent sync:

```text
com.soxl.daily.scheduler.plist: account_notional=10000 broker=futu
com.soxl.preopen.scheduler.plist: account_notional=10000 broker=futu
launchctl list: com.soxl.preopen.scheduler and com.soxl.daily.scheduler loaded
```

Dashboard static export:

```text
ok dashboard_static_export generated_at=2026-05-24 11:10:30
versions=38
missing=0
latest=paper_trading
paper_trading_rows=1
paper_trading_end=2026-05-22
```

Scheduler health excerpt:

```text
scheduled broker: futu
account notional: 10000
runtime composer: enabled composer=budget_usage profile=v19reg_v22_budget_usage_bull_guard_v1
broker target: futu
data: ok last_date=2026-05-22 expected=2026-05-22
model: ok v19reg_v22_budget_usage_bull_guard_v1
broker: ok futu quote ok
```

## Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_futu_adapter.py::test_futu_broker_account_notional_caps_snapshot_and_order_size \
  tests/test_daily_scheduler.py::test_scheduler_passes_account_notional_to_paper_runner \
  tests/test_production_schedule.py::test_crontab_uses_shanghai_safe_us_market_windows \
  tests/test_production_schedule.py::test_launchagent_runs_post_close_in_shanghai_local_time \
  tests/test_production_schedule.py::test_launchagent_runs_preopen_cached_dry_run_in_shanghai_local_time \
  tests/test_scheduler_health.py::test_scheduler_health_uses_configured_launchagent_broker_for_healthcheck \
  tests/test_paper_trading.py::test_main_enables_step_clip_for_futu_notional_simulation -q
# passed

PYTHONPATH=. qlib_venv/bin/python scripts/check_dashboard_static_export.py
# ok dashboard_static_export ... paper_trading_rows=1 paper_trading_end=2026-05-22
```

## Decision

- Futu scheduled simulated trading now starts from a `10000` repo-side notional
  account.
- The first non-dry scheduled Futu run should size from this notional account
  and cap a cold-start jump to `35%` before submitting orders.
- The Futu backend may still display its own simulated-account cash, but that
  value no longer controls SOXL strategy sizing when `--account-notional 10000`
  is present.
