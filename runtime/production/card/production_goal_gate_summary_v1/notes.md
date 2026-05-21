# Production Goal Gate Summary

Date: 2026-05-21

## Goal

Make the active production objective auditable from one health-check section
instead of requiring manual interpretation across data, dashboard, Vercel, and
post-close proof sections.

## Root Cause

`production/check_scheduler_health.sh` already exposed each lower-level signal:

- data/model/broker health
- dashboard static export status
- Vercel deploy evidence
- post-close target marker
- ordinary simulated-paper proof

But there was no final gate summary. That made the remaining thread blockers
easy to miss after each scheduled run.

## Change

- `production/check_scheduler_health.sh` now prints `Production Goal Gates`.
- The section summarizes:
  - data
  - model
  - broker
  - dashboard static export
  - Vercel remote deploy
  - post-close composer target
  - ordinary paper proof
  - overall pending issue keys

## Evidence

Red/green test:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_scheduler_health.py::test_scheduler_health_targets_enabled_composer_for_model_and_runtime_proof -q
# red before implementation: Production Goal Gates section missing
# green after implementation: 1 passed
```

Real health:

```bash
production/check_scheduler_health.sh | sed -n '/--- Production Goal Gates ---/,$p'
# data: ok last_date=2026-05-20 expected=2026-05-20
# model: ok v19reg_v22_budget_usage_bull_guard_v1 (slow=balanced_rule_residual_v19_reg, fast=balanced_rule_residual_v22)
# broker: ok paper broker ready
# dashboard static: ok dashboard_static_export ... latest=paper_trading ... paper_trading_end=2026-05-20
# Vercel remote: blocked status=blocked_no_credentials auth=blocked_no_credentials project=missing_project_link deployment_url=none env_file_applied_keys=none
# post-close composer target: pending latest_success_marker lacks runtime_composer=budget_usage composer_profile=v19reg_v22_budget_usage_bull_guard_v1
# ordinary paper proof: pending selected_model=v19reg_v22_budget_usage_bull_guard_v1 execution=duplicate_skipped
# overall: pending external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof
```

## Decision

The health checker now has a single production gate summary for the active
thread goal. Current state remains incomplete for exactly three reasons:

1. `external_vercel_setup`
2. `scheduled_postclose_target`
3. `scheduled_paper_proof`

Do not mark the thread goal complete until the summary reports `overall: ok`.
