---
date: '2026-04-24'
title: Learned-router conditioned oracle distillation diagnostic
kind: review
profiles: []
status: dropped
tags:
- dropped
- eval-2022
- review
artifacts:
- evidence/post_train_review/hierarchical_online_budget_daily/soxl_hier_learned_slow5.csv
- evidence/post_train_review/risk_budget_learned_conditioned_oracle_distill_v1.json
- evidence/post_train_review/risk_budget_learned_conditioned_oracle_distill_v1.md
workpad: codex_exec_plans/completed/2026-04/2026-04-10-soxl-agent-loop.md
summary: Explicit slow-mode one-hot conditioning did not improve the learned-router
  oracle distillation probe. Proxy return fell from 412.9 percent to 387.1 percent
  and MaxDD rose from 28.7 percent to 30.1 percent, so simple regime identity is not
  the missing ingredient.
---

# Learned-router conditioned oracle distillation diagnostic

## Context

After the learned-router oracle probe, the next cheapest question was whether
the fast actor simply needed explicit slow-mode identity.

## Evidence

Command:

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
```

Result:

- `test_proxy`: `387.1% / Sharpe 1.008 / MaxDD 30.1%`
- `test_2022`: `-11.7% / Sharpe -1.054 / MaxDD 13.2%`
- `test_2023_2024`: `77.7% / Sharpe 1.114 / MaxDD 24.9%`

## Decision

Drop the idea as a next-step mechanism. Explicit regime identity alone did not
improve the oracle probe.

## Next

If this line continues, the next changes should target fast-actor capacity or
loss design, not just mode-conditioning the same actor.
