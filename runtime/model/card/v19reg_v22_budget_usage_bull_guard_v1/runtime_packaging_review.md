# v19reg_v22_budget_usage_bull_guard_v1 Runtime Packaging Review

## Decision

`policy_layer_ready_runtime_binding_pending`.

The budget-usage bull-guard mechanism is no longer hidden inside the
counterfactual app layer. `BullParticipationGuardConfig` and guarded
`compose_budget_usage_policy_frame` now live in
`src/soxl_quant/policy/two_speed_budget.py`, so a future production/runtime
composer can import the same implementation used by the diagnostic card.

This is not runtime auto-select evidence yet.

## Completed Scope

- Moved `BullParticipationGuardConfig` into the shared policy layer.
- Moved bull-participation guard composition into
  `compose_budget_usage_policy_frame`.
- Kept `src/soxl_quant/app/budget_counterfactual.py` as a scoring/rendering
  consumer of the shared policy implementation.
- Added `production/composed_session.py` to `scripts/check_harness.py` compile
  coverage so the existing composed production entrypoint is at least syntax
  checked by the repo-owned harness.

## Runtime Binding Blockers

- `production/soxl_paper_trading.py` still plans a single selected actor
  profile.
- No governed runtime profile config exists yet for `slow_profile` plus
  `fast_profile` plus the price-derived bull guard.
- Fresh non-dry post-close paper proof for the currently selected v22 model is
  still pending because the latest executed paper artifact is still the
  pre-switch `balanced_rule_residual_v19` run.
- Vercel remote deployment remains blocked until `VERCEL_TOKEN` is set or
  `vercel login` is completed.

## Verification

```bash
PYTHONPATH=src:. qlib_venv/bin/python tests/test_two_speed_budget.py
# red before implementation: ImportError for BullParticipationGuardConfig

uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_two_speed_budget.py tests/test_budget_counterfactual.py -q
# 12 passed

PYTHONPATH=src:. qlib_venv/bin/python scripts/check_harness.py
# passed=160 failures=0

production/check_scheduler_health.sh
# data/model/broker/dashboard ok
# runtime proof pending: latest executed profile is balanced_rule_residual_v19,
# selected model is balanced_rule_residual_v22
# Vercel status: blocked_no_credentials
```

## Next Action

Add a governed runtime composer/profile interface for the two-actor
budget-usage path, then verify it through the standard post-close paper and
dashboard path before allowing runtime auto-selection.
