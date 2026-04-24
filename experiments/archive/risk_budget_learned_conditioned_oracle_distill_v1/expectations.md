# Expectations

## Closest Historical Success

`risk_budget_learned_oracle_distill_v1`: `test_proxy 412.9% / Sharpe 1.042 /
MaxDD 28.7%`.

## Closest Historical Failure

Earlier single-actor oracle probes still trailed their target replay/runtime
paths even with held-out labels.

## Why This Is Not A Duplicate

This adds explicit slow-mode identity to the fast actor input. The target path
and replay source stay the same, but the actor no longer has to infer regime
membership implicitly from the raw observation.

## Validation Commands

```bash
PYTHONPATH=. qlib_venv/bin/python scripts/build_budget_usage_oracle_actor.py \
  --profile risk_budget_learned_conditioned_oracle_distill_v1 \
  --replay-daily evidence/post_train_review/hierarchical_online_budget_daily/soxl_hier_learned_slow5.csv \
  --condition-on-mode \
  --skip-data-refresh \
  --steps 1600 \
  --sample-size 0 \
  --output-json evidence/post_train_review/risk_budget_learned_conditioned_oracle_distill_v1.json \
  --output-md evidence/post_train_review/risk_budget_learned_conditioned_oracle_distill_v1.md

PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py
```
