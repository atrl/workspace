# budget_usage_runtime_composer_v1

## Decision

The standard paper-trading and scheduler entrypoints can now run an explicit
budget-usage runtime composer. The composer-specific runtime model gate is
approved for the exact reviewed slow/fast pair, but the installed scheduler
default remains unchanged.

## Scope

- `production/soxl_paper_trading.py` accepts
  `--runtime-composer budget_usage`.
- Required explicit inputs:
  - `--composer-profile`
  - `--slow-profile`
  - `--fast-profile`
- Optional bull guard inputs mirror the diagnostic composer:
  - `--bull-participation-guard`
  - `--bull-ma-window`
  - `--bull-momentum-window`
  - `--bull-high-window`
  - `--bull-min-periods`
  - `--bull-min-momentum`
  - `--bull-max-drawdown`
- `production/soxl_daily_scheduler.py` forwards the same arguments to the paper
  runner when provided.

## Live Evidence

Paper dry-run:

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

- data: fresh through `2026-05-20`
- output: `runtime/soxl/paper_trading/2026-05-20/dry_run`
- signal: `target=48.41%`, `price=$173.20`
- risk gate: approved
- model-promotion detail:
  `composer promotion review approved: runtime/model/card/v19reg_v22_budget_usage_bull_guard_v1/runtime_promotion_review.json`

Scheduler smoke:

```bash
PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py \
  --dry-run --no-notify --skip-data-update --skip-dashboard \
  --runtime-composer budget_usage \
  --composer-profile v19reg_v22_budget_usage_bull_guard_v1 \
  --slow-profile balanced_rule_residual_v19_reg \
  --fast-profile balanced_rule_residual_v22 \
  --bull-participation-guard
```

Result: scheduler completed and forwarded the composer path to paper trading.

Promotion review:

- `runtime/model/card/v19reg_v22_budget_usage_bull_guard_v1/runtime_promotion_review.md`
- `runtime/model/card/v19reg_v22_budget_usage_bull_guard_v1/runtime_promotion_review.json`

## Verification

- Focused red/green:
  `tests/test_budget_usage_runtime_promotion_review.py`,
  `tests/test_runtime.py::test_composer_promotion_gate_status_allows_approved_review_over_component_gate`,
  `tests/test_runtime.py::test_composer_promotion_gate_status_blocks_component_model_mismatch`,
  `tests/test_runtime.py::test_composer_promotion_gate_status_blocks_unapproved_review`,
  `tests/test_runtime.py::test_compose_budget_usage_runtime_signal_records_bull_guard_audit`,
  `tests/test_daily_scheduler.py::test_scheduler_passes_runtime_composer_to_paper_runner`,
  `tests/test_paper_trading.py::test_main_uses_budget_runtime_composer_when_requested`.
- Composer gate:
  `uv run --python qlib_venv/bin/python --with pytest pytest tests/test_budget_usage_runtime_promotion_review.py tests/test_runtime.py::test_composer_promotion_gate_status_allows_approved_review_over_component_gate tests/test_runtime.py::test_composer_promotion_gate_status_blocks_component_model_mismatch tests/test_runtime.py::test_composer_promotion_gate_status_blocks_unapproved_review -q`
  -> `6 passed`.
- Related suite:
  `uv run --python qlib_venv/bin/python --with pytest pytest tests/test_budget_usage_runtime_promotion_review.py tests/test_runtime.py tests/test_two_speed_budget.py tests/test_paper_trading.py tests/test_daily_scheduler.py -q`
  -> `59 passed`.
- Full harness:
  `PYTHONPATH=src:. qlib_venv/bin/python scripts/check_harness.py`
  -> `passed=167 failures=0`.
- Scheduler health:
  default data/model/broker/dashboard are ok; runtime proof remains pending for
  selected v22 and Vercel remains `blocked_no_credentials`.

## Next Action

Do not add these composer flags to the installed scheduled LaunchAgents without
an explicit non-dry scheduling decision. The current production default remains
the single selected `balanced_rule_residual_v22` actor.
