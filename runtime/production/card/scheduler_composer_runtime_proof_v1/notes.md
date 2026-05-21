# Scheduler Composer Runtime Proof

Date: 2026-05-21

## Goal

Make scheduler health prove the same runtime path that the LaunchAgent wrapper
will run after `SOXL_RUNTIME_COMPOSER=budget_usage` is enabled.

## Root Cause

`production/check_scheduler_health.sh` correctly reported the private env
composer state, but its paper-trading healthcheck still called
`production/soxl_paper_trading.py --healthcheck` without composer arguments.

That made the health output validate single-actor
`balanced_rule_residual_v22` and accept the old v22 duplicate-skip proof even
though the next scheduled wrapper invocation would use
`v19reg_v22_budget_usage_bull_guard_v1`.

## Change

- `production/check_scheduler_health.sh` now parses the scheduler-visible
  runtime composer env once into shared shell state.
- `print_paper_health` forwards complete composer settings to
  `soxl_paper_trading.py --healthcheck`.
- Incomplete composer env remains visible as incomplete and does not get
  forwarded to paper healthcheck.

## Evidence

Red/green regression:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_scheduler_health.py::test_scheduler_health_targets_enabled_composer_for_model_and_runtime_proof -q
# red before implementation: health model stayed on balanced_rule_residual_v22
# green after implementation: 1 passed
```

Full scheduler-health suite:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py -q
# 20 passed
```

Live health before non-dry composer proof:

```bash
production/check_scheduler_health.sh
# model: ok v19reg_v22_budget_usage_bull_guard_v1 (slow=balanced_rule_residual_v19_reg, fast=balanced_rule_residual_v22)
# runtime proof: pending latest_executed_profile=balanced_rule_residual_v19 selected_model=v19reg_v22_budget_usage_bull_guard_v1
```

Real-env non-dry wrapper proof:

```bash
production/run_soxl_scheduler.sh --broker paper --no-notify --skip-data-update --skip-dashboard
# using runtime composer=budget_usage profile=v19reg_v22_budget_usage_bull_guard_v1
# as_of=2026-05-20
# risk_gate approved=True
# already executed for 2026-05-20 broker=paper; skipping duplicate order
# duplicate inference output=runtime/soxl/paper_trading/2026-05-20/duplicate_skip/v19reg_v22_budget_usage_bull_guard_v1
```

Artifact check:

```bash
jq '{date:.date, profile:.profile, dry_run:.dry_run, executed:.executed, execution:.execution_result.status, composition:.composition.name, composer_profile:.composition.composer_profile, slow:.composition.slow_profile, fast:.composition.fast_profile, risk_approved:.risk_gate.approved}' \
  runtime/soxl/paper_trading/2026-05-20/duplicate_skip/v19reg_v22_budget_usage_bull_guard_v1/summary.json
# date=2026-05-20
# profile=v19reg_v22_budget_usage_bull_guard_v1
# dry_run=false
# executed=false
# execution=duplicate_skipped
# composition=budget_usage
# composer_profile=v19reg_v22_budget_usage_bull_guard_v1
# slow=balanced_rule_residual_v19_reg
# fast=balanced_rule_residual_v22
# risk_approved=true
```

Live health after proof:

```bash
production/check_scheduler_health.sh
# model: ok v19reg_v22_budget_usage_bull_guard_v1 (slow=balanced_rule_residual_v19_reg, fast=balanced_rule_residual_v22)
# runtime proof: ok latest_non_dry_profile=v19reg_v22_budget_usage_bull_guard_v1 selected_model=v19reg_v22_budget_usage_bull_guard_v1 execution=duplicate_skipped
# runtime proof artifact=runtime/soxl/paper_trading/2026-05-20/duplicate_skip/v19reg_v22_budget_usage_bull_guard_v1/summary.json
```

## Decision

The current scheduler health surface now validates the composer runtime target,
not the previous single-actor target. The composer has current-date non-dry
inference/risk-gate proof, with duplicate-order protection preventing a second
same-day paper order.

Remaining gates:

- remote Vercel deployment still needs scheduler-visible credentials
- first weekly walk-forward scheduled artifact is still pending
- the next new trading date should produce the first ordinary scheduled
  composer paper artifact rather than a same-day duplicate-skip artifact
