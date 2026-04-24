---
date: '2026-04-24'
title: Learned-router oracle distillation diagnostic
kind: review
profiles: []
status: dropped
tags:
- dropped
- eval-2022
- review
artifacts:
- evidence/post_train_review/hierarchical_online_budget_daily/soxl_hier_learned_slow5.csv
- evidence/post_train_review/risk_budget_learned_oracle_distill_v1.json
- evidence/post_train_review/risk_budget_learned_oracle_distill_v1.md
workpad: codex_exec_plans/completed/2026-04/2026-04-10-soxl-agent-loop.md
summary: No-promotion oracle distillation from the deployable learned-router slow5
  daily path reached test_proxy 412.9 percent, Sharpe 1.042, and MaxDD 28.7 percent.
  This is stronger than the prior oracle distill but still far below the target path,
  so the bottleneck remains single-actor capacity in the learned-router regime.
---

# Learned-router oracle distillation diagnostic

## Context

The hierarchical runtime diagnostic suggested that the best deployable line is a
learned router with slow refresh. This loop tested whether a single actor can
represent that line when trained directly on the resulting daily path.

## Evidence

Command:

```bash
PYTHONPATH=. qlib_venv/bin/python scripts/build_budget_usage_oracle_actor.py \
  --profile risk_budget_learned_oracle_distill_v1 \
  --replay-daily evidence/post_train_review/hierarchical_online_budget_daily/soxl_hier_learned_slow5.csv \
  --skip-data-refresh \
  --steps 1600 \
  --sample-size 0 \
  --output-json evidence/post_train_review/risk_budget_learned_oracle_distill_v1.json \
  --output-md evidence/post_train_review/risk_budget_learned_oracle_distill_v1.md
```

Results:

- label distribution: `{0: 238, 1: 1138, 2: 48, 3: 52, 4: 32}`
- BC loss / accuracy: `1.124 / 0.781`
- `test_2022`: `-11.7% / Sharpe -1.054 / MaxDD 13.2% / avg_pos 0.105`
- `test_2023_2024`: `78.6% / Sharpe 1.130 / MaxDD 24.9% / avg_pos 0.336`
- `test_proxy`: `412.9% / Sharpe 1.042 / MaxDD 28.7% / avg_pos 0.296`

## Decision

Drop as promotable by construction. The learned-router oracle is better than the
earlier fixed-replay oracle probe, but still not enough to say that a single
actor can carry the deployable learned-router runtime.

## Next

If this line continues, improve the fast actor architecture or training target
inside the learned-router regime instead of revisiting cadence-only variants.
