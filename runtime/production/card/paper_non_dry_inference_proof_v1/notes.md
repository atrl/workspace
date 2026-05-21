# Paper Non-Dry Inference Proof v1

## Context

Scheduler health used `executed=true` as the only proof that the selected
runtime profile had run in non-dry paper mode. In the paper runner, `executed`
means a broker order was submitted. A valid selected-profile inference can
produce no order when the target is already within the broker lot/threshold
boundary, which would leave runtime proof falsely pending.

## Change

- `production/soxl_paper_trading.py` now persists `execution_result.json` for
  non-dry paper runs that reach the broker, including skipped no-order cases.
- `summary.json` includes the same `execution_result` payload.
- `production/check_scheduler_health.sh` accepts a selected-profile, non-dry,
  risk-approved run with `execution_result.status=skipped` as runtime proof.
- Mismatched-profile executed artifacts still report pending proof, and blocked
  or rejected selected-profile runs remain pending.

## Regression

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_paper_trading.py::test_main_persists_skipped_execution_result_for_non_dry_hold \
  tests/test_scheduler_health.py::test_scheduler_health_accepts_selected_profile_non_dry_skipped_order_as_runtime_proof -q
# red before implementation:
# - execution_result.json missing
# - runtime proof ok line missing
# green after implementation: 2 passed

uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_paper_trading.py tests/test_scheduler_health.py \
  tests/test_daily_scheduler.py -q
# 42 passed

uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_runtime.py tests/test_paper_trading.py \
  tests/test_scheduler_health.py tests/test_daily_scheduler.py -q
# 62 passed

PYTHONPATH=src:. qlib_venv/bin/python scripts/check_harness.py
# passed=168 failures=0

production/check_scheduler_health.sh
# local scheduler/data/model/broker/dashboard ok
# selected v22 runtime proof still waits for next fresh non-dry date
# Vercel remains blocked_no_credentials
```

## Harness Isolation

The first full harness run after this change exposed a test dependency on the
live production overlay state file. `tests/test_runtime.py` now points the
relative overlay-state-path default test at a temporary root, preserving the
production state while keeping the harness deterministic.

## Decision

Runtime proof should mean the selected profile completed non-dry inference and
passed the risk gate, not necessarily that a new order was submitted. This keeps
the next fresh selected-profile paper run from becoming a false negative on a
legitimate hold/no-lot-change day.
