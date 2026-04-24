---
date: '2026-04-23'
title: Risk-budget oracle distillation and composer extraction
kind: review
profiles: []
status: dropped
tags:
- dropped
- eval-2022
- review
artifacts: []
workpad: memory/index.json
summary: No-promotion oracle distillation used full fixed-replay daily labels for
  risk_budget_usage_v1. The label set contained action-4 full usage, but the single
  actor reached only test_proxy 363.8% / Sharpe 1.090 / MaxDD 23.7%, far below fixed
  replay 2003.6%. Keep the reusable composer extraction; stop trying to compress the
  two-speed mechanism into one PPO actor.
---

# Risk-budget oracle distillation and composer extraction

## Context

`risk_budget_usage_v1` fixed replay passed the replay gates, while actor-backed
parity, native PPO, and train-split BC warm-start failed to reproduce the upside.
The remaining question was whether a single actor could express the fixed replay
path at all if trained directly on that path.

## Evidence

Command:

```bash
PYTHONPATH=. /Users/rziiiii/.openclaw/workspace/qlib_venv/bin/python scripts/build_budget_usage_oracle_actor.py \
  --profile risk_budget_oracle_distill_v1 \
  --skip-data-refresh \
  --steps 1600 \
  --sample-size 0
```

Results:

- oracle samples: `1508`
- label distribution: `{0: 127, 1: 436, 2: 383, 3: 423, 4: 139}`
- BC loss / accuracy: `1.195 / 0.711`
- `test_2022`: `-5.5% / Sharpe -0.350 / MaxDD 11.0% / avg_pos 0.117`
- `test_2023_2024`: `58.5% / Sharpe 1.180 / MaxDD 19.3% / avg_pos 0.271`
- `test_proxy`: `363.8% / Sharpe 1.090 / MaxDD 23.7% / avg_pos 0.264`

## Decision

Drop `risk_budget_oracle_distill_v1` as promotable by construction. Even with
held-out replay labels and full-usage examples, the single actor did not
approach fixed replay's `2003.6%` proxy return.

Keep the reusable `compose_budget_usage_policy_frame()` extraction and the oracle
dataset builder for future capacity diagnostics.

## Next

Stop trying to compress the two-speed mechanism into one PPO actor. The next
valid direction is an explicit runtime policy composer where the mode layer owns
risk bands and the fast layer supplies budget usage.
