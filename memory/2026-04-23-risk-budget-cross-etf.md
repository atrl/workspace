---
date: '2026-04-23'
title: Risk-budget usage cross-ETF diagnostic
kind: review
profiles:
- risk_budget_usage_v1
- balanced_rule_residual_imitation_v4
status: dropped
tags:
- risk_budget_usage_v1
- balanced_rule_residual_imitation_v4
- dropped
- execution-overlay
- generalization
- review
artifacts:
- runtime/post_train_review/risk_budget_usage_v1_cross_etf.json
- runtime/post_train_review/risk_budget_usage_v1_cross_etf.md
- runtime/post_train_review/risk_budget_usage_v1_cross_etf_daily/
workpad: codex_exec_plans/active/2026-04-10-soxl-agent-loop.md
summary: Cross-ETF diagnostic split risk_budget_usage_v1 into fixed SOXL signal transfer and actor budget runtime. The fixed signal transferred strongly across SOXL/TQQQ/UPRO/SPXL/TECL, but the deployable actor/runtime path produced only 28-144% proxy return with Sharpe 0.37-0.83. Keep the diagnostic; do not promote actor runtime.
---

# Risk-budget usage cross-ETF diagnostic

## Context

`risk_budget_usage_v1` fixed replay is strong on SOXL, but previous actor parity
and oracle distillation showed the single actor cannot reproduce the fixed
composer. This loop tested whether the mechanism survives the supported 3x ETF
universe.

The retained `balanced_rule_residual_mode_v1` actor bundle is absent, so this is
not a full two-actor zero-shot composer. The diagnostic separates:

- fixed SOXL signal transfer onto each ETF's own price path
- actor-backed budget-usage runtime using the retained `imitation_v4` actor

## Evidence

Command:

```bash
PYTHONPATH=. qlib_venv/bin/python scripts/risk_budget_usage_cross_etf.py
```

Fixed SOXL signal transfer:

- `SOXL`: `2030.3% / Sharpe 1.252 / MaxDD 44.9%`
- `TQQQ`: `823.4% / Sharpe 1.287 / MaxDD 29.6%`
- `UPRO`: `427.4% / Sharpe 1.148 / MaxDD 22.8%`
- `SPXL`: `429.7% / Sharpe 1.154 / MaxDD 22.9%`
- `TECL`: `965.5% / Sharpe 1.260 / MaxDD 30.6%`

Actor budget runtime:

- `SOXL`: `144.4% / Sharpe 0.827 / MaxDD 22.1%`
- `TQQQ`: `51.2% / Sharpe 0.473 / MaxDD 22.7%`
- `UPRO`: `35.2% / Sharpe 0.436 / MaxDD 22.3%`
- `SPXL`: `28.1% / Sharpe 0.372 / MaxDD 22.9%`
- `TECL`: `46.0% / Sharpe 0.429 / MaxDD 23.7%`

## Decision

Keep `scripts/risk_budget_usage_cross_etf.py` and the evidence artifacts.

Do not promote the actor/runtime path. It generalizes drawdown compression but
not return participation. The fixed signal result is useful evidence that the
timing path transfers across related 3x ETFs, but it remains a SOXL-derived
fixed path and is still exposed to research-selection bias.

## Next

If true cross-ETF deployment remains the goal, recover or regenerate the
`mode_v1` actor and test a real two-actor zero-shot composer. Otherwise keep
`risk_budget_usage_v1` as a research counterfactual rather than a production
strategy.
