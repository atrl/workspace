---
title: balanced_rule_residual_refined_v2
direction: model
status: candidate
---

# balanced_rule_residual_refined_v2

## Motivation

`refined_v1` achieved excellent drawdown control (proxy MaxDD 54.6% vs v19 54.8%,
test_2023_24 MaxDD 31.0% vs v19 37.3%) but proxy return collapsed from 735% to 500%
— a 235pp loss. The model became overly conservative, holding too much cash and
missing intermediate-period upside.

Root cause: complete removal of `activity_bonus` + very low `opportunity_cost_penalty`
(0.02) gave the model no signal for "when to participate". It oscillated between
cash and position (325 trades vs v19's ~220) without conviction.

## Hypothesis

Instead of a blanket `activity_bonus` that forces trading in all regimes, use a
**targeted `trend_alignment_bonus`** that only rewards position when the trend
is clearly favorable. This provides precise incentive without creating
conflicting gradients in sideways/down markets.

Additionally, slightly increase `opportunity_cost_penalty` from 0.02 → 0.04 to
prevent excessive permanent cash holding.

## Changes from refined_v1

| parameter | refined_v1 | refined_v2 | rationale |
|-----------|-----------|-----------|-----------|
| opportunity_cost_penalty | 0.02 | **0.04** | Moderate FOMO to prevent excessive cash |
| trend_alignment_bonus | 0.0 | **0.20** | Reward position ONLY when trend is clear |
| trend_alignment_threshold | — | **0.03** | Trend confirmation gate |

All other parameters identical to refined_v1.

## Target

- proxy MaxDD: < 55% (preserve v1's DD control)
- proxy return: > 600% (recover at least 100pp from v1)
- proxy sharpe: > 0.85
- test_2022 MaxDD: < 33%

## Comparison baseline

- Canonical: `balanced_rule_residual_v19`
- Direct compare: `balanced_rule_residual_refined_v1`
