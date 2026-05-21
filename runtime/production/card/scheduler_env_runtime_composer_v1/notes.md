# Scheduler Env Runtime Composer v1

## Goal

Let the installed LaunchAgent wrapper intentionally enable the approved
budget-usage runtime composer from the private scheduler env file, without
editing plist files and without changing the default scheduled profile.

## Baseline

- `production/soxl_daily_scheduler.py` and `production/soxl_paper_trading.py`
  already supported `--runtime-composer budget_usage`.
- The installed LaunchAgents call `production/run_soxl_scheduler.sh`, which
  loads `${SOXL_LAUNCH_ENV_FILE:-$HOME/.openclaw/soxl.env}`.
- The wrapper passed only explicit plist/CLI arguments through to the scheduler,
  so enabling the approved composer required editing plist arguments or running
  the scheduler manually.

## Change

- `production/run_soxl_scheduler.sh` now accepts test/operator overrides for
  `SOXL_SCHEDULER_ROOT` and `SOXL_SCHEDULER_PYTHON`, while preserving the
  existing absolute defaults.
- After loading the private env file, the wrapper appends composer flags only
  when `SOXL_RUNTIME_COMPOSER` is set to a non-`none` value and the caller did
  not already pass `--runtime-composer`.
- Supported env fields:
  - `SOXL_RUNTIME_COMPOSER`
  - `SOXL_RUNTIME_COMPOSER_PROFILE`
  - `SOXL_RUNTIME_SLOW_PROFILE`
  - `SOXL_RUNTIME_FAST_PROFILE`
  - `SOXL_RUNTIME_BULL_PARTICIPATION_GUARD`
  - optional bull guard knobs: `SOXL_RUNTIME_BULL_MA_WINDOW`,
    `SOXL_RUNTIME_BULL_MOMENTUM_WINDOW`, `SOXL_RUNTIME_BULL_HIGH_WINDOW`,
    `SOXL_RUNTIME_BULL_MIN_PERIODS`, `SOXL_RUNTIME_BULL_MIN_MOMENTUM`,
    `SOXL_RUNTIME_BULL_MAX_DRAWDOWN`

## Evidence

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_production_schedule.py::test_launchagent_wrapper_can_enable_runtime_composer_from_private_env_file -q
# red before implementation: wrapper ignored SOXL_SCHEDULER_ROOT/PYTHON and no command capture was written
# green after implementation: 1 passed

bash -n production/run_soxl_scheduler.sh
# ok

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_production_schedule.py -q
# 8 passed

SOXL_LAUNCH_ENV_FILE="$tmp_env" \
  production/run_soxl_scheduler.sh \
  --broker paper --dry-run --no-notify --skip-data-update --skip-dashboard
# using runtime composer=budget_usage
# profile=v19reg_v22_budget_usage_bull_guard_v1
# as_of=2026-05-20
# signal target=48.41%
# risk_gate approved=True
# dry-run: skipped order placement
# output=runtime/soxl/paper_trading/2026-05-20/dry_run
```

Dry-run artifact:

- `runtime/soxl/paper_trading/2026-05-20/dry_run/summary.json`

The artifact records `profile=v19reg_v22_budget_usage_bull_guard_v1`, a
composer model-gate pass through
`runtime/model/card/v19reg_v22_budget_usage_bull_guard_v1/runtime_promotion_review.json`,
and `risk_gate.approved=true`.

## Decision

Default LaunchAgent behavior remains the single selected
`balanced_rule_residual_v22` path. The approved composer can now be enabled
intentionally by adding env fields to the private scheduler env file, then
observing the next dry-run or non-dry scheduled paper artifact.

Do not use this to silently promote the composer. A fresh non-dry paper date
must still prove the intended profile path before considering a default schedule
change.
