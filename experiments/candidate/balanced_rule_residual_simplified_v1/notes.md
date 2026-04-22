---
date: '2026-04-22'
title: balanced_rule_residual_simplified_v1
kind: experiment
status: candidate
baseline: balanced_rule_residual_v19
role: diagnose
changed_layer: reward + network + data
---

# balanced_rule_residual_simplified_v1

## Hypothesis

Simplify the reward function (remove many penalty terms), increase network capacity,
and extend episode length to improve learning stability and capture longer-horizon patterns.

## Result

- proxy return: **776.9%**
- proxy Sharpe: **0.91**
- proxy MaxDD: **58.5%**
- trades: 297

## Comparison vs v19

- return: 776.9% vs 735.1% (+41.8pp)
- Sharpe: 0.91 vs 0.95 (-0.04)
- MaxDD: 58.5% vs 54.8% (+3.7pp)

## Verdict

Return improves but drawdown worsens. Promotion blocked. This confirms the fundamental
trade-off: higher model capacity and longer episodes help capture upside, but do not
improve crisis discipline. The reward simplification removed drawdown guardrails that
were actually useful.

## Retrieval tags

- simplified-reward
- larger-network
- longer-episode
- drawdown-trade-off
