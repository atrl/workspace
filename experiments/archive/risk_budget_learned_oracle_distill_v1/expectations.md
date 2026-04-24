# Expectations

## Closest Historical Success

`hier_learned_slow5` from `hierarchical_online_budget_generalization`:
cross-ETF best deployable variant on `TQQQ`, `UPRO`, `SPXL`, and `TECL`.

## Closest Historical Failure

`risk_budget_oracle_distill_v1`: even full fixed-replay labels only reached
`test_proxy 363.8% / Sharpe 1.090`, far below the replay path.

## Why This Is Not A Duplicate

This does not distill the fixed replay counterfactual. It distills the strongest
deployable learned-router runtime path, which is a different regime definition
and the next logical capacity probe after the hierarchical runtime diagnostic.

## Validation Commands

```bash
PYTHONPATH=. qlib_venv/bin/python scripts/build_budget_usage_oracle_actor.py \
  --profile risk_budget_learned_oracle_distill_v1 \
  --replay-daily runtime/model/reviews/hierarchical_online_budget_daily/soxl_hier_learned_slow5.csv \
  --skip-data-refresh \
  --steps 1600 \
  --sample-size 0 \
  --output-json runtime/model/reviews/risk_budget_learned_oracle_distill_v1.json \
  --output-md runtime/model/reviews/risk_budget_learned_oracle_distill_v1.md

PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py
```
