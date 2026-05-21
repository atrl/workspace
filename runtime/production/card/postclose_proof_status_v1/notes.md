# Post-close Proof Status

Date: 2026-05-21

## Goal

Prevent the post-close health section from looking complete when the scheduler
has a success marker but the enabled composer still lacks ordinary new-date
paper proof.

## Root Cause

`production/check_scheduler_health.sh` already printed:

```text
ordinary paper proof: pending ... execution=duplicate_skipped
```

But the later `Post-close Run Status` section still printed only:

```text
postclose status: success marker found
```

That was technically true for the wrapper completion marker, but easy to
misread as the whole simulated-paper gate being complete.

## Change

- `print_ordinary_paper_proof_status()` now records the ordinary proof state.
- `print_postclose_status()` prints a `postclose proof` line after a success
  marker when ordinary proof state is known.
- Current duplicate-skip state now renders as:

```text
postclose status: success marker found
postclose proof: ordinary paper proof pending selected_model=v19reg_v22_budget_usage_bull_guard_v1 execution=duplicate_skipped
```

## Evidence

Red/green focused test:

```bash
PYTHONPATH=src:. qlib_venv/bin/python -c "import importlib.util; spec=importlib.util.spec_from_file_location('test_scheduler_health', 'tests/test_scheduler_health.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); mod.test_scheduler_health_accepts_selected_profile_duplicate_skip_as_runtime_proof()"
# red before implementation: missing postclose proof line
# green after implementation: exit 0
```

Live health:

```bash
production/check_scheduler_health.sh
# postclose status: success marker found
# postclose proof: ordinary paper proof pending selected_model=v19reg_v22_budget_usage_bull_guard_v1 execution=duplicate_skipped
```

## Decision

The post-close wrapper can be considered operational, but the simulated-paper
gate is not complete until the next new-date scheduled non-dry run produces:

```text
postclose proof: ordinary paper proof ok ...
```
