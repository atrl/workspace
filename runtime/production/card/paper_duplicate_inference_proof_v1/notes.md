# Selected v22 Duplicate-Skip Inference Proof

Date: 2026-05-21

## Goal

Persist proof that the selected runtime profile completed non-dry scheduled
inference when same-day idempotency prevents a duplicate paper order.

## Root Cause

The root paper artifact for `2026-05-20` is a non-dry
`balanced_rule_residual_v19` execution. A later scheduled run selected
`balanced_rule_residual_v22` on fresh `2026-05-20` data, but
`production/soxl_paper_trading.py` returned immediately when it detected an
existing non-dry execution for the same broker/date.

That preserved broker idempotency but dropped the selected v22 inference/risk
evidence, leaving health to report selected runtime proof as pending.

## Change

- Duplicate same-day non-dry attempts now write artifacts under:
  `runtime/soxl/paper_trading/<date>/duplicate_skip/<profile>/`
- The summary records:
  - `dry_run=false`
  - `executed=false`
  - `execution_result.status=duplicate_skipped`
  - `duplicate_of` pointing at the prior root non-dry execution
- The duplicate path does not place orders, append the paper daily CSV, or save
  broker state.
- Scheduler health checks selected-profile nested non-dry proof artifacts before
  reporting selected runtime proof pending on a root profile mismatch.

## Live Evidence

```bash
PYTHONPATH=src:. qlib_venv/bin/python production/soxl_paper_trading.py \
  --no-notify --skip-data-update
# selected balanced_rule_residual_v22 on fresh 2026-05-20 data
# already executed for 2026-05-20 broker=paper; skipping duplicate order
# duplicate inference output=runtime/soxl/paper_trading/2026-05-20/duplicate_skip/balanced_rule_residual_v22

production/check_scheduler_health.sh
# runtime proof: ok latest_non_dry_profile=balanced_rule_residual_v22 selected_model=balanced_rule_residual_v22 execution=duplicate_skipped
# runtime proof artifact=runtime/soxl/paper_trading/2026-05-20/duplicate_skip/balanced_rule_residual_v22/summary.json
```

## Test Evidence

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_paper_trading.py::test_main_persists_duplicate_non_dry_inference_without_order \
  tests/test_scheduler_health.py::test_scheduler_health_accepts_selected_profile_duplicate_skip_as_runtime_proof -q
# red before implementation: duplicate artifact missing; health reported pending
# green after implementation: 2 passed

uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_paper_trading.py tests/test_scheduler_health.py tests/test_daily_scheduler.py -q
# 46 passed
```

## Remaining Gates

- `duplicate_skipped` proves selected-profile non-dry inference/risk gating, not
  a new broker order.
- Vercel remote deployment still requires credentials.
- Weekly walk-forward still needs its first scheduled run artifact.
