# v19reg_v22_budget_usage_bull_guard_v1 Runtime Interface Review

## Decision

`standard_path_interface_ready_component_gate_blocked`.

The standard paper-trading path can now explicitly run the budget-usage
composer. It is not enabled by default and is not approved for non-dry
scheduled execution.

## Completed

- `src/soxl_quant/runtime.py` now exposes
  `compose_budget_usage_runtime_signal` and
  `plan_budget_usage_composed_session`.
- `production/soxl_paper_trading.py` accepts explicit
  `--runtime-composer budget_usage` arguments.
- `production/soxl_daily_scheduler.py` forwards explicit composer arguments to
  the paper runner.
- Composer dry-runs persist a `composition` payload beside the normal signal,
  risk-gate, account, and reconciliation files.
- `scripts/check_harness.py` now compiles both production entrypoints touched by
  this path.

## Live Dry-Run

```bash
PYTHONPATH=. qlib_venv/bin/python production/soxl_paper_trading.py \
  --dry-run --no-notify --skip-data-update \
  --runtime-composer budget_usage \
  --composer-profile v19reg_v22_budget_usage_bull_guard_v1 \
  --slow-profile balanced_rule_residual_v19_reg \
  --fast-profile balanced_rule_residual_v22 \
  --bull-participation-guard
```

Readout:

- signal date: `2026-05-20`
- target/current: `0.48408568215209924` / `0.48408568215209924`
- output: `runtime/soxl/paper_trading/2026-05-20/dry_run/summary.json`
- risk gate: blocked
- blocking reason:
  `soxl_balanced_rule_residual_v19_reg_actor.pt: model failed promotion gate`

The block is expected under current governance. The composer line has
walk-forward diagnostic evidence, but no composer-level runtime promotion
decision has replaced the component actor gate.

## Scheduler Smoke

```bash
PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py \
  --dry-run --no-notify --skip-data-update --skip-dashboard \
  --runtime-composer budget_usage \
  --composer-profile v19reg_v22_budget_usage_bull_guard_v1 \
  --slow-profile balanced_rule_residual_v19_reg \
  --fast-profile balanced_rule_residual_v22 \
  --bull-participation-guard
```

Result: exit code `0`; scheduler forwarded the composer arguments and completed
the cached dry-run.

## Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_runtime.py::test_compose_budget_usage_runtime_signal_records_bull_guard_audit \
  tests/test_daily_scheduler.py::test_scheduler_passes_runtime_composer_to_paper_runner \
  tests/test_paper_trading.py::test_main_uses_budget_runtime_composer_when_requested -q
# red before implementation: compose_budget_usage_runtime_signal missing
# green after implementation: 3 passed

uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_runtime.py tests/test_two_speed_budget.py \
  tests/test_paper_trading.py tests/test_daily_scheduler.py -q
# 53 passed

PYTHONPATH=src:. qlib_venv/bin/python scripts/check_harness.py
# passed=161 failures=0

production/check_scheduler_health.sh
# default data/model/broker/dashboard ok
# runtime proof pending for selected v22
# Vercel status: blocked_no_credentials
```

## Next Action

Add a composer-level promotion review or explicit runtime profile governance
decision before allowing non-dry scheduled execution.
