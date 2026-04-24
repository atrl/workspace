---
date: 2026-04-23
title: Two-speed mode budget replay dropped
kind: failure
profiles:
- balanced_rule_residual_imitation_v4_soxl_daily
- balanced_rule_residual_mode_v1_soxl_daily
- soxl
status: dropped
tags:
- balanced_rule_residual_imitation_v4_soxl_daily
- balanced_rule_residual_mode_v1_soxl_daily
- dropped
- eval-2022
- failure
- soxl
artifacts:
- evidence/post_train_review/balanced_rule_residual_imitation_v4_soxl_daily.csv
- evidence/post_train_review/balanced_rule_residual_mode_v1_soxl_daily.csv
- evidence/post_train_review/two_speed_mode_budget_v1.
workpad: codex_exec_plans/completed/2026-04/2026-04-10-soxl-agent-loop.md
summary: Fixed-signal two-speed replay combined mode_v1 as the slow risk anchor with
  imitation_v4 as the fast participation path. Proxy and restart participation improved,
  but test_2022 MaxDD remained 33.9%, above the strict 30% gate. Keep the replay tool
  and drop the candidate; next work should avoid mild slack/budget retrains and require
  stronger crisis-state protection or external hedge mechanics.
---

# Two-speed mode budget replay dropped

## Context

The current bottleneck is not training depth. `mode_v1` provides the best
retained risk-quality frontier, while `imitation_v4` improves restart
participation but misses the strict 30% drawdown target. This replay tested
whether a fixed two-speed composition could combine those strengths without
opening another retraining branch.

## Evidence

Command:

```bash
PYTHONPATH=. /Users/rziiiii/.openclaw/workspace/qlib_venv/bin/python scripts/replay_two_speed_mode_budget.py \
  --slow-daily /Users/rziiiii/.openclaw/workspace/evidence/post_train_review/balanced_rule_residual_mode_v1_soxl_daily.csv \
  --fast-daily /Users/rziiiii/.openclaw/workspace/evidence/post_train_review/balanced_rule_residual_imitation_v4_soxl_daily.csv
```

Result:

- `test_2022`: `-16.5% / Sharpe -0.224 / MaxDD 33.9% / avg_pos 0.289`.
- `test_2023_2024`: `156.7% / Sharpe 1.278 / MaxDD 31.4%`.
- `test_proxy`: `1398.3% / Sharpe 1.100 / MaxDD 51.6% / trades 171`.

Acceptance:

- Passed: `test_2022 avg_pos <= 0.35`.
- Passed: `test_proxy Sharpe >= 0.90`.
- Passed: `test_2023_2024 return >= 46.8%`.
- Passed: proxy MaxDD improves by at least 2 points versus `imitation_v4_fast`.
- Failed: `test_2022 MaxDD <= 30.0%`.

## Decision

Drop `two_speed_mode_budget_v1` as a promotable candidate.

Keep:

- `src/soxl_quant/policy/two_speed_budget.py`
- `scripts/replay_two_speed_mode_budget.py`
- `evidence/post_train_review/two_speed_mode_budget_v1.*`
- archived experiment notes

## Next

Do not run another mild slack or budget retrain. The result narrows the issue:
participation can be restored without global exposure drift, but SOXL 2022 path
drawdown still needs stronger crisis-state protection or a different protection
layer, potentially outside single-asset long-only exposure.
