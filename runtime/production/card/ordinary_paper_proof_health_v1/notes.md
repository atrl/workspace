# Ordinary Paper Proof Health

Date: 2026-05-21

## Goal

Keep scheduler health from treating same-date duplicate-skip proof as the final
ordinary paper-trading proof for the active runtime composer.

## Root Cause

`production/check_scheduler_health.sh` correctly accepted a selected-profile
`duplicate_skipped` artifact as runtime proof because it proves non-dry
inference, risk gating, and same-date idempotency. That was still too broad for
the production goal: a duplicate-skip artifact does not prove the first
new-date scheduled paper loop has advanced under the active composer.

## Change

- Kept the existing `runtime proof` line unchanged.
- Added `ordinary paper proof` classification for selected-profile non-dry
  artifacts:
  - `executed=true` -> `ok`
  - fresh non-dry `execution=skipped` -> `ok`
  - `execution=duplicate_skipped` -> `pending`
- Added a detail line explaining that duplicate-skip proves same-date idempotent
  inference only, and that the next new-date scheduled non-dry run must produce
  executed or skipped order proof.

## Evidence

Red test:

```bash
PYTHONPATH=src:. qlib_venv/bin/python scripts/check_harness.py
# failed tests/test_scheduler_health.py::test_scheduler_health_accepts_selected_profile_duplicate_skip_as_runtime_proof
# reason: ordinary paper proof line was absent
```

Focused green checks through direct test-function execution:

```bash
PYTHONPATH=src:. qlib_venv/bin/python -c "import importlib.util; spec=importlib.util.spec_from_file_location('test_scheduler_health', 'tests/test_scheduler_health.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); mod.test_scheduler_health_accepts_selected_profile_duplicate_skip_as_runtime_proof(); mod.test_scheduler_health_accepts_selected_profile_non_dry_skipped_order_as_runtime_proof()"
# exit 0
```

Live health:

```bash
production/check_scheduler_health.sh
# runtime proof: ok latest_non_dry_profile=v19reg_v22_budget_usage_bull_guard_v1 selected_model=v19reg_v22_budget_usage_bull_guard_v1 execution=duplicate_skipped
# ordinary paper proof: pending latest_non_dry_profile=v19reg_v22_budget_usage_bull_guard_v1 selected_model=v19reg_v22_budget_usage_bull_guard_v1 execution=duplicate_skipped
# ordinary paper proof detail: duplicate-skip proves same-date idempotent inference only; next new-date scheduled non-dry run must produce executed or skipped order proof
```

## Decision

Current paper health is now split into:

1. runtime proof: composer inference and risk gate are proven for the expected
   data date through duplicate-skip idempotency
2. ordinary paper proof: still pending until the next new-date scheduled
   non-dry run produces an executed order or a fresh no-order skip

This keeps the remaining production gate visible instead of letting
duplicate-skip look like the final simulated-paper fix.
