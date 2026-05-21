# Post-close Runtime Target Marker

Date: 2026-05-21

## Goal

Make scheduled post-close evidence distinguish a generic wrapper success from a
success that actually ran the enabled runtime composer target.

## Root Cause

The active LaunchAgent environment now enables:

- `SOXL_RUNTIME_COMPOSER=budget_usage`
- `SOXL_RUNTIME_COMPOSER_PROFILE=v19reg_v22_budget_usage_bull_guard_v1`
- `SOXL_RUNTIME_SLOW_PROFILE=balanced_rule_residual_v19_reg`
- `SOXL_RUNTIME_FAST_PROFILE=balanced_rule_residual_v22`

The existing `scheduler_postclose.log` success marker was created before this
runtime target was visible in the run-start line. Health could therefore show
`postclose status: success marker found` while the latest post-close log did
not prove that the current composer path had run under the scheduled wrapper.

## Change

- `production/soxl_daily_scheduler.py` now includes runtime-composer target
  fields in the run-start marker when `--runtime-composer` is enabled.
- `production/check_scheduler_health.sh` now prints a `postclose target` line
  when the private scheduler env enables a runtime composer.
- A post-close success marker is only target-complete when the latest run-start
  marker includes the enabled runtime composer, composer profile, slow profile,
  and fast profile.

## Evidence

Red/green tests:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_scheduler_health.py::test_scheduler_health_targets_enabled_composer_for_model_and_runtime_proof \
  tests/test_daily_scheduler.py::test_scheduler_prints_runtime_composer_in_run_start_marker -q
# red before implementation: missing runtime_composer run marker and missing postclose target line
# green after implementation: 2 passed
```

Real scheduler health now exposes the gap:

```bash
production/check_scheduler_health.sh | sed -n '/--- Post-close Run Status ---/,/--- Next Scheduled Runs ---/p'
# postclose status: success marker found
# postclose target: pending latest_success_marker lacks runtime_composer=budget_usage composer_profile=v19reg_v22_budget_usage_bull_guard_v1
# postclose proof: ordinary paper proof pending selected_model=v19reg_v22_budget_usage_bull_guard_v1 execution=duplicate_skipped
```

Real wrapper smoke proves future scheduled logs will carry the target marker:

```bash
production/run_soxl_scheduler.sh --broker paper --dry-run --no-notify --skip-data-update --skip-dashboard
# [scheduler] run started ... runtime_composer=budget_usage composer_profile=v19reg_v22_budget_usage_bull_guard_v1 slow_profile=balanced_rule_residual_v19_reg fast_profile=balanced_rule_residual_v22 bull_guard=True
# [paper] using runtime composer=budget_usage profile=v19reg_v22_budget_usage_bull_guard_v1 slow=balanced_rule_residual_v19_reg fast=balanced_rule_residual_v22
# [scheduler] all done
```

## Decision

The remaining ordinary simulated-paper gate is now stricter and easier to
audit. The next new-date scheduled post-close run must show:

1. `postclose target: ok runtime_composer=budget_usage composer_profile=v19reg_v22_budget_usage_bull_guard_v1`
2. `postclose proof: ordinary paper proof ok ...`

Until then, the thread goal remains active.
