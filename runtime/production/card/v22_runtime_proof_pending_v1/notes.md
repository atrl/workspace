# v22 Runtime Proof Pending Health Visibility

Date: 2026-05-21

## Context

The scheduler health check already showed two true facts:

- current model health selects `balanced_rule_residual_v22`
- latest executed paper artifact is still `balanced_rule_residual_v19`

Before this change, the health output did not connect those facts, so the
remaining post-close proof could look complete at a glance.

## Change

`production/check_scheduler_health.sh` now compares the latest non-dry executed
paper summary profile against the current healthcheck-selected model profile.

When they differ, it prints:

```text
runtime proof: pending latest_executed_profile=<profile> selected_model=<profile>
```

When they match, it prints:

```text
runtime proof: ok latest_executed_profile=<profile> selected_model=<profile>
```

## Evidence

Regression:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py::test_scheduler_health_reports_pending_runtime_profile_proof -q
# 1 passed
```

Focused suite:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py -q
# 8 passed
```

Live health readout after the patch:

```text
model: ok balanced_rule_residual_v22 (soxl_balanced_rule_residual_v22_actor.pt)
latest signal: date=2026-05-20 profile=balanced_rule_residual_v19 dry_run=false executed=true
runtime proof: pending latest_executed_profile=balanced_rule_residual_v19 selected_model=balanced_rule_residual_v22
```

Follow-up clarification:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py::test_scheduler_health_reports_pending_runtime_profile_proof -q
# red first on missing detail line; green after implementation
# 1 passed

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py tests/test_daily_scheduler.py tests/test_paper_trading.py tests/test_vercel_dashboard.py tests/test_dashboard.py -q
# 44 passed

production/check_scheduler_health.sh
# data: ok last_date=2026-05-20 expected=2026-05-20
# model: ok balanced_rule_residual_v22 (soxl_balanced_rule_residual_v22_actor.pt)
# latest signal: date=2026-05-20 profile=balanced_rule_residual_v19 dry_run=false executed=true
# runtime proof: pending latest_executed_profile=balanced_rule_residual_v19 selected_model=balanced_rule_residual_v22
# runtime proof detail: latest executed date already matches expected data date; next non-dry-run execution must use selected model
```

## Decision

Production remains healthy, and the pending state is expected because the latest
expected data date was already executed before v22 became the selected model.
Do not rerun `2026-05-20` to manufacture a v22 artifact; the full objective is
not complete until a fresh post-close non-dry paper run naturally writes a v22
executed artifact. The next eligible scheduled post-close run is the proof
point.
