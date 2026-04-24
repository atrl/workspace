---
date: '2026-04-24'
title: Learned-router attention oracle distillation
kind: review
profiles: []
status: dropped
tags:
- dropped
- eval-2022
- review
artifacts:
- runtime/model/reviews/hierarchical_online_budget_daily/soxl_hier_learned_slow5.csv
- runtime/model/reviews/risk_budget_learned_attention_oracle_distill_v1.json
- runtime/model/reviews/risk_budget_learned_attention_oracle_distill_v1.md
workpad: memory/index.json
summary: 'Swapping the learned-router oracle probe backbone from GRU to attention
  produced effectively identical policy metrics. This compresses the search space
  further: same-scale backbone swaps are not enough to close the learned-router gap.'
---

# Learned-router attention oracle distillation

## Context

After head and conditioning variants failed, this loop tested whether a small
attention backbone could beat the GRU oracle probe on the same teacher path.

## Evidence

Command:

```bash
PYTHONPATH=. qlib_venv/bin/python scripts/build_budget_usage_oracle_actor.py \
  --profile risk_budget_learned_attention_oracle_distill_v1 \
  --replay-daily runtime/model/reviews/hierarchical_online_budget_daily/soxl_hier_learned_slow5.csv \
  --skip-data-refresh \
  --steps 1600 \
  --sample-size 0 \
  --output-json runtime/model/reviews/risk_budget_learned_attention_oracle_distill_v1.json \
  --output-md runtime/model/reviews/risk_budget_learned_attention_oracle_distill_v1.md
```

Result:

- `test_proxy`: `412.9% / Sharpe 1.042 / MaxDD 28.7%`
- `test_2022`: `-11.7% / Sharpe -1.054 / MaxDD 13.2%`
- `test_2023_2024`: `78.6% / Sharpe 1.130 / MaxDD 24.9%`

## Decision

Drop the idea as a next-step mechanism. A same-scale backbone swap from GRU to
attention did not produce a better learned-router oracle probe.

## Next

If this line continues, the next change should be a larger sequence/state
redesign instead of another small backbone substitution.
