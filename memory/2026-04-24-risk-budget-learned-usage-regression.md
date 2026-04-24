---
date: '2026-04-24'
title: Learned-router usage regression probe
kind: review
profiles: []
status: dropped
tags:
- dropped
- eval-2022
- review
artifacts:
- runtime/post_train_review/hierarchical_online_budget_daily/soxl_hier_learned_slow5.csv
- runtime/post_train_review/risk_budget_learned_usage_regression_v1.json
- runtime/post_train_review/risk_budget_learned_usage_regression_v1.md
workpad: codex_exec_plans/active/2026-04-10-soxl-agent-loop.md
summary: Continuous usage regression on the learned-router teacher path fit the labels
  very well but produced a much worse policy than the learned-router oracle classifier.
  Proxy return fell to 222.8 percent and MaxDD rose to 34.7 percent, so discrete label
  quantization is not the main bottleneck.
---

# Learned-router usage regression probe

## Context

The next cheap question after the failed mode-conditioning probe was whether the
5-bin action head itself was the bottleneck.

## Evidence

Command:

```bash
PYTHONPATH=. qlib_venv/bin/python scripts/build_budget_usage_regression_probe.py \
  --profile risk_budget_learned_usage_regression_v1 \
  --replay-daily runtime/post_train_review/hierarchical_online_budget_daily/soxl_hier_learned_slow5.csv \
  --skip-data-refresh \
  --steps 2000 \
  --sample-size 0 \
  --output-json runtime/post_train_review/risk_budget_learned_usage_regression_v1.json \
  --output-md runtime/post_train_review/risk_budget_learned_usage_regression_v1.md
```

Result:

- train loss / MAE: `0.00230 / 0.0351`
- `test_proxy`: `222.8% / Sharpe 0.772 / MaxDD 34.7%`
- `test_2022`: `-15.1% / Sharpe -1.436 / MaxDD 17.6%`
- `test_2023_2024`: `63.0% / Sharpe 0.978 / MaxDD 25.7%`

## Decision

Drop the idea as a next-step mechanism. A scalar regression head fit the target
labels but produced a clearly worse policy than the learned-router oracle
classifier.

## Next

If this line continues, the next changes should target backbone capacity or
sequence/state design, not head type alone.
