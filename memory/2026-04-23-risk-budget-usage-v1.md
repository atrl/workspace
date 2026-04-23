---
date: 2026-04-23
title: Risk-budget usage replay candidate
kind: review
profiles:
- balanced_rule_residual_imitation_v4_soxl_daily
- balanced_rule_residual_mode_v1_soxl_daily
status: reference
tags:
- balanced_rule_residual_imitation_v4_soxl_daily
- balanced_rule_residual_mode_v1_soxl_daily
- candidate
- eval-2022
- review
artifacts:
- runtime/backtest
- runtime/post_train_review/balanced_rule_residual_imitation_v4_soxl_daily.csv
- runtime/post_train_review/balanced_rule_residual_mode_v1_soxl_daily.csv
- runtime/post_train_review/risk_budget_usage_v1.json
- runtime/post_train_review/risk_budget_usage_v1.md
- runtime/post_train_review/risk_budget_usage_v1_soxl_daily.csv
workpad: codex_exec_plans/active/2026-04-10-soxl-agent-loop.md
summary: 'Fixed replay treated imitation_v4 as a budget usage ratio inside mode_v1
  risk bands. It passed all replay acceptance checks: test_2022 MaxDD 25.6%, avg_pos
  0.214, test_proxy 2003.6% / Sharpe 1.248 / MaxDD 45.0%. Keep as candidate fixed-replay
  mechanism; next step must implement runtime/backtest parity before any promotion
  claim.'
---

# Risk-budget usage replay candidate

## Context

The prior `two_speed_mode_budget_v1` composition used `mode_v1` as a slow anchor
and let `imitation_v4` participate above that anchor by mode slack. It improved
proxy quality but still failed strict `test_2022 MaxDD <= 30%`.

`risk_budget_usage_v1` changes the interpretation: `imitation_v4` is no longer a
target position. It is a usage ratio inside mode-owned risk bands.

## Evidence

Command:

```bash
PYTHONPATH=. /Users/rziiiii/.openclaw/workspace/qlib_venv/bin/python scripts/replay_two_speed_mode_budget.py \
  --slow-daily /Users/rziiiii/.openclaw/workspace/runtime/post_train_review/balanced_rule_residual_mode_v1_soxl_daily.csv \
  --fast-daily /Users/rziiiii/.openclaw/workspace/runtime/post_train_review/balanced_rule_residual_imitation_v4_soxl_daily.csv \
  --composition budget_usage \
  --profile-name risk_budget_usage_v1 \
  --output-json runtime/post_train_review/risk_budget_usage_v1.json \
  --output-md runtime/post_train_review/risk_budget_usage_v1.md \
  --output-daily runtime/post_train_review/risk_budget_usage_v1_soxl_daily.csv
```

Metrics:

- `test_2020`: `239.0% / Sharpe 2.285 / MaxDD 21.7% / avg_pos 0.490`.
- `test_2022`: `-17.1% / Sharpe -0.358 / MaxDD 25.6% / avg_pos 0.214`.
- `test_2023_2024`: `127.8% / Sharpe 1.278 / MaxDD 32.1% / avg_pos 0.389`.
- `test_proxy`: `2003.6% / Sharpe 1.248 / MaxDD 45.0% / trades 188`.

## Decision

Keep as a candidate fixed-replay mechanism.

Do not promote yet. This is not a full actor/runtime validation. The next step
must implement the same budget-usage composition inside runtime/backtest
positioning and compare against this daily replay.

## Next

Implement runtime/backtest parity:

- mode layer owns risk bands
- actor/RL output is interpreted as budget usage ratio
- validation must reproduce the replay direction on fixed split and test proxy
  before any training branch is justified
