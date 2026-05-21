# Weekly Composer Review

Date: 2026-05-21

## Goal

Make the weekly walk-forward wrapper review the active runtime target after the
scheduler env was switched from the single v22 actor to the approved
`v19reg_v22_budget_usage_bull_guard_v1` composer.

## Root Cause

`production/run_soxl_weekly_wf.sh` loaded the private env file, but it always
defaulted to `balanced_rule_residual_v22` and the single-profile
`review_runtime_selection.py` path. After `SOXL_RUNTIME_COMPOSER=budget_usage`
was enabled, that weekly evidence no longer matched the runtime target used by
the scheduler wrapper.

The wrapper also hardcoded its root and Python path, making it harder to test
without touching real runtime state.

## Change

- `production/run_soxl_weekly_wf.sh` now supports
  `SOXL_WEEKLY_WF_ROOT` and `SOXL_WEEKLY_WF_PYTHON` overrides.
- When a complete `SOXL_RUNTIME_COMPOSER=budget_usage` env is present, weekly
  runs `scripts/review_budget_usage_runtime_promotion.py` for the composer
  profile instead of the single-profile walk-forward/review path.
- The single-profile v22 path remains the fallback when no runtime composer is
  enabled.
- Runtime `.err` logs are ignored as generated runtime state.

## Evidence

Red/green regression:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_production_schedule.py::test_weekly_walkforward_wrapper_reviews_enabled_runtime_composer -q
# red before implementation: wrapper ignored test root/python and no command capture was written
# green after implementation: 1 passed
```

Manual wrapper proof using the real private scheduler env and LaunchAgent log
paths:

```bash
production/run_soxl_weekly_wf.sh >> runtime/soxl/wf_weekly.log 2>> runtime/soxl/wf_weekly.err
```

Log tail:

```text
[weekly-wf] run started at 2026-05-21T14:26:14+08:00 mode=runtime_composer composer=budget_usage profile=v19reg_v22_budget_usage_bull_guard_v1 slow=balanced_rule_residual_v19_reg fast=balanced_rule_residual_v22
[budget_usage_runtime_promotion] wrote runtime/model/card/v19reg_v22_budget_usage_bull_guard_v1
[weekly-wf] all done
```

Review payload:

```bash
jq '{profile:.profile, decision:.decision, runtime_model_gate_allowed:.runtime_model_gate_allowed, blocking_reasons:.blocking_reasons}' \
  runtime/model/card/v19reg_v22_budget_usage_bull_guard_v1/runtime_promotion_review.json
# profile=v19reg_v22_budget_usage_bull_guard_v1
# decision=approved
# runtime_model_gate_allowed=true
# blocking_reasons=[]
```

Health:

```bash
production/check_scheduler_health.sh
# wf_weekly: [weekly-wf] all done
# model: ok v19reg_v22_budget_usage_bull_guard_v1 (slow=balanced_rule_residual_v19_reg, fast=balanced_rule_residual_v22)
# runtime proof: ok latest_non_dry_profile=v19reg_v22_budget_usage_bull_guard_v1 selected_model=v19reg_v22_budget_usage_bull_guard_v1 execution=duplicate_skipped
```

## Decision

Weekly review evidence now follows the active composer runtime target. The
remaining production blocker is Vercel remote deployment credentials. The next
ordinary paper-trading proof is still the next new trading date's scheduled
composer artifact rather than the current same-day duplicate-skip proof.
