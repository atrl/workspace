# Paper Idempotency v1

## Context

The manual full paper loop on `2026-05-21 03:01 CST` successfully submitted a
paper BUY for signal date `2026-05-18`. A second scheduler pass before the next
signal date must not place another paper order for the same signal date, because
the paper broker uses a step clip and another call could move the account again
toward the same target.

Baseline before the guard:

- `runtime/soxl/paper_trading/2026-05-18/summary.json`
  - `broker=paper`
  - `dry_run=false`
  - `executed=true`
- `runtime/soxl/paper_trading/paper_broker_state.json`
  - `date=2026-05-18`
  - `shares=3000`
  - `position=0.4856047894`
  - `account_value=937490.7536`
- `runtime/soxl/paper_trading/paper_trading_soxl_daily.csv`
  - rows: `3`
  - latest row: `2026-05-18`, action `BUY`, position `0.485605`

## Change

- `production/soxl_paper_trading.py`
  - added an existing-summary guard for same-day, same-broker, non-dry-run,
    already-executed paper summaries.
  - returns before drawdown checks, order placement, post-execution audit,
    dashboard CSV append, or broker-state save when the duplicate guard matches.
- `tests/test_paper_trading.py`
  - added a regression test that fails if duplicate same-day execution reaches
    order placement, dashboard CSV append, or broker-state persistence.

## Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py::test_main_skips_duplicate_non_dry_run_for_existing_executed_summary -q
# failed before guard by reaching place_target_position; now 1 passed

PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --no-notify --skip-data-update
# [paper] already executed for 2026-05-18 broker=paper; skipping duplicate order
# [scheduler] dashboard build
# [scheduler] all done
```

Post-rerun state:

- paper broker state stayed at `date=2026-05-18`, `shares=3000`,
  `position=0.4856047894`.
- cumulative paper CSV stayed at `3` rows.
- `dashboard_data/index.json` paper summary stayed at
  `start=2026-03-31`, `end=2026-05-18`, `rows=3`.

Final gates:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py tests/test_daily_scheduler.py tests/test_vercel_dashboard.py tests/test_dashboard.py -q
# 29 passed

PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py
# [summary] passed=112 failures=0

PYTHONPATH=. qlib_venv/bin/python scripts/build_runtime_index.py --check
# runtime index OK

git diff --check
# clean

production/check_scheduler_health.sh
# LaunchAgents loaded
# data: ok last_date=2026-05-19 expected=2026-05-19
# model: ok balanced_rule_residual_v19
# broker: ok paper broker ready
```

## Decision

The simulated paper path is now idempotent for a repeated non-dry-run pass on an
already executed signal date. This guard should remain in place before trusting
the natural `05:35 Tue-Sat Asia/Shanghai` post-close LaunchAgent cadence.
