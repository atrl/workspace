# Expectations

## Closest Historical Success

`risk_budget_learned_oracle_distill_v1`: `test_proxy 412.9% / Sharpe 1.042 /
MaxDD 28.7%`.

## Closest Historical Failure

`risk_budget_learned_usage_regression_v1`: swapping only the head type made the
probe much worse, so the remaining chance is sequence/backbone capacity.

## Why This Is Not A Duplicate

This keeps the same learned-router teacher path and the same discrete oracle
probe, but replaces the GRU backbone with attention. It isolates backbone
capacity rather than head type.

## Validation Commands

```bash
PYTHONPATH=. qlib_venv/bin/python scripts/build_budget_usage_oracle_actor.py \
  --profile risk_budget_learned_attention_oracle_distill_v1 \
  --replay-daily runtime/model/reviews/hierarchical_online_budget_daily/soxl_hier_learned_slow5.csv \
  --skip-data-refresh \
  --steps 1600 \
  --sample-size 0 \
  --output-json runtime/model/reviews/risk_budget_learned_attention_oracle_distill_v1.json \
  --output-md runtime/model/reviews/risk_budget_learned_attention_oracle_distill_v1.md

PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py
```
