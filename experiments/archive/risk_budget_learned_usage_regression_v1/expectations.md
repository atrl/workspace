# Expectations

## Closest Historical Success

`risk_budget_learned_oracle_distill_v1`: `test_proxy 412.9% / Sharpe 1.042 /
MaxDD 28.7%`.

## Closest Historical Failure

`risk_budget_learned_conditioned_oracle_distill_v1`: explicit mode one-hot did
not help, which suggests the head/target may be a bigger issue than regime
identity.

## Why This Is Not A Duplicate

This keeps the same teacher path and same learned-router runtime but changes the
learning target from discrete action classification to continuous usage
regression.

## Validation Commands

```bash
PYTHONPATH=. qlib_venv/bin/python scripts/build_budget_usage_regression_probe.py \
  --profile risk_budget_learned_usage_regression_v1 \
  --replay-daily evidence/post_train_review/hierarchical_online_budget_daily/soxl_hier_learned_slow5.csv \
  --skip-data-refresh \
  --steps 2000 \
  --sample-size 0 \
  --output-json evidence/post_train_review/risk_budget_learned_usage_regression_v1.json \
  --output-md evidence/post_train_review/risk_budget_learned_usage_regression_v1.md

PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py
```
