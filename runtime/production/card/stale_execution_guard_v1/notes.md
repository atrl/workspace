# Stale Execution Guard v1

## Context

The previous health-check work made stale data visible, but the paper-trading
main path still needed a hard execution guard. A post-close data update can
return successfully while leaving the canonical panel behind the expected last
trading day. In that case, a non-dry-run paper session must not continue into
qlib initialization, model selection, broker state mutation, or dashboard CSV
append.

## Change

- `production/soxl_paper_trading.py`
  - reads data freshness after the data-update step and before qlib init
  - exits non-zero for non-dry-run execution when data is stale
  - allows stale data only for dry-run visibility and prints an explicit warning
- `production/soxl_daily_scheduler.py`
  - treats stale post-update data as a cached dry-run fallback condition
  - updated operator copy to say fresh data is unavailable or stale
- `tests/test_paper_trading.py`
  - proves stale non-dry-run data blocks before qlib init
- `tests/test_daily_scheduler.py`
  - proves stale post-update data retries as cached dry-run with
    `--skip-data-update` and `--dry-run`

## Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py::test_main_blocks_non_dry_run_when_data_remains_stale -q
# failed before the guard by reaching qlib init; now 1 passed

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_daily_scheduler.py::test_scheduler_retries_stale_post_update_data_as_cached_dry_run -q
# failed before scheduler classified stale data; now 1 passed

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_paper_trading.py tests/test_daily_scheduler.py tests/test_scheduler_health.py tests/test_production_schedule.py tests/test_vercel_dashboard.py tests/test_dashboard.py -q
# 25 passed

PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --skip-data-update --no-notify --skip-dashboard
# first pass blocked: Cached data is stale for non-dry-run execution
# retry pass completed as cached dry-run
# [paper] dry-run: skipped order placement
# [scheduler] all done (with cached dry-run)

production/check_scheduler_health.sh
# data: stale last_date=2026-05-15 expected=2026-05-20 stale_days=5
# model: ok balanced_rule_residual_v19 (soxl_balanced_rule_residual_v19_actor.pt)
# broker: ok paper broker ready
```

## Readout

The production loop now has two separate protections:

- transient provider failure -> cached dry-run fallback
- update succeeds but data remains stale -> cached dry-run fallback

Fresh post-close data is still required before the scheduler can mutate paper
broker state. The next live proof remains the real `05:35 Tue-Sat
Asia/Shanghai` post-close run.
