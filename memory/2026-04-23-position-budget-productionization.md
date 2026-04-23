---
date: '2026-04-23'
title: Position Budget Productionization Smoke
kind: decision
profiles:
- balanced_rule_residual_imitation_v4
- balanced_rule_residual_imitation_v4_budget
- smoke
status: reference
tags:
- balanced_rule_residual_imitation_v4
- balanced_rule_residual_imitation_v4_budget
- dashboard
- decision
- eval-2022
- execution-overlay
- reference
- smoke
artifacts: []
workpad: codex_exec_plans/active/2026-04-10-soxl-agent-loop.md
summary: Config-owned regime-conditional position budget plumbing was added across
  evaluation, qlib backtest, runtime, and training action interpretation. Smoke seed=12345
  ran end-to-end but failed strict drawdown promotion; keep the infrastructure and
  candidate scaffold, do not promote the smoke actor.
---

# Position Budget Productionization Smoke

## Context

The bottleneck review showed reward-only tuning and architecture swaps are not the current frontier. The repeated failure is selective participation under risk: the harness can suppress exposure, but it still struggles to loosen only during the right restart/rebound windows.

## Mechanism

Added `position_budget` as a separate config layer from reward shaping:

- enabled only when configured by profile
- fires only when `regime_risk >= regime_threshold`
- caps target position by consumed drawdown budget buckets
- wired into `evaluate_split`, qlib `backtest_profile`, runtime signal generation, and `PositionAction`

New candidate scaffold:

- `balanced_rule_residual_imitation_v4_budget`
- baseline: `balanced_rule_residual_imitation_v4`
- role: execution-layer budget scaffold, not a reward-only branch

## Evidence

Command:

```bash
PYTHONPATH=. /Users/rziiiii/.openclaw/workspace/qlib_venv/bin/python soxl_qlib_strategy_v2.py \
  --profiles balanced_rule_residual_imitation_v4_budget \
  --smoke \
  --seed 12345
```

Smoke metrics:

- valid: `40.5% / Sharpe 1.23 / MaxDD 31.2%`
- test_2020: `132.4% / Sharpe 2.32 / MaxDD 23.1%`
- test_2022: `-33.5% / Sharpe -1.34 / MaxDD 36.3% / avg_pos 0.271`
- test_2023_2024: `38.6% / Sharpe 0.62 / MaxDD 41.5%`
- test_proxy: `721.1% / Sharpe 0.94 / MaxDD 51.0%`
- qlib backtest: `577.3% / Sharpe 0.86 / MaxDD 50.2% / 305 trades`

Harness validation:

- `scripts/check_harness.py`: `passed=41 failures=0`
- `scripts/build_backtest_dashboard.py`: passed after optional-input handling fix

## Decision

Keep the `position_budget` plumbing and `balanced_rule_residual_imitation_v4_budget` scaffold.

Do not promote the smoke actor. It fails the strict `test_2022 MaxDD <= 30%` gate and worsens `test_2023_2024` drawdown.

Next optimization should use the budget as a reusable evaluation/runtime overlay, ideally on a restored retained actor, before starting another retraining branch.
