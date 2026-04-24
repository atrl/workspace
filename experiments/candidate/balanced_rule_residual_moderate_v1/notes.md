---
title: balanced_rule_residual_moderate_v1
direction: model
status: candidate
---

# balanced_rule_residual_moderate_v1

## Motivation

User clarified: does not object to forced trading or encouraging volatile trading. The goal is not "minimal reward" but "principled reward" — using Kelly framework to balance return vs activity.

The minimal_v1 experiment proved that removing ALL participation signals causes training to degrade. The v16 experiment proved that ~17 reward terms can achieve 632.4% return. The question is: which terms are essential and which are redundant?

## Hypothesis

A "moderate" reward (~12 active terms, down from ~17) can match v16's performance by:
1. Keeping proven return drivers (alpha, trend_alignment, opportunity_cost, profit_taking, panic_buy)
2. Keeping proven risk constraints (episode_penalty, turnover, micro_rebalance)
3. Reducing step-level penalties to avoid over-constraint (v16's weaker penalties helped achieve better Sharpe)
4. Removing counterproductive terms (rule_deviation_penalty punishes beating the rule)
5. Removing unproven micro-management (short_hold_turnover, flat_trade)

## Mechanism Delta vs v16

| Layer | v16 | moderate_v1 |
|-------|-----|-------------|
| Core signal | alpha (Kelly) + PBRS | same |
| Cost | turnover + smoothness + micro | same |
| Episode risk | penalty=2.0, threshold=0.15 | same |
| Step risk | drawdown=0.28, downside=0.20, risk_budget=0.24, high_exp=0.08 | **REDUCED**: 0.15, 0.10, 0.15, 0.05 |
| Participation | activity_bonus=1.25 + inactivity=0.80 (gated) | **REPLACED**: trend_alignment=0.25 + opportunity_cost=0.05 |
| Tactical | profit_taking=0.25, panic_buy=0.35 | same |
| Removed | — | rule_deviation, short_hold_turnover, flat_trade |

## External Reference

- Kelly criterion: reward should maximize log(wealth). Activity incentives should be proportional to "edge × volatility" not unconditional.
- v16 proved 632.4%/0.91/50.1% with activity_bonus
- refined_v3 proved 637.8%/0.89/54.9% with trend_alignment
- Both approaches work; moderate_v1 tests whether weaker step-risk + selective participation = best of both

## Target Slices / Windows

- proxy full (primary)
- test_2020, test_2022, test_2023_24
- walk-forward (after fixed_split pass)

## Acceptance Criteria

- proxy return >= 550% (above refined_v2, closer to v16)
- proxy MaxDD <= 50%
- test_2022 MaxDD <= 35%
- test_2022 avg_pos <= 0.35
- seed robustness: <=150pp swing vs another seed

## Expected Trade-off

- Lower step-level penalties → slightly higher return, slightly worse DD
- trend_alignment vs activity_bonus → more principled, potentially more seed-robust
- Removal of rule_deviation → less constraint on beating the rule

## Failure Mode

- Step-level penalties too weak → MaxDD > 52%
- trend_alignment insufficient vs activity_bonus → return < 500%
- Missing rule_deviation → excessive deviation from rule causes choppy behavior

## Comparison Table Row Key

`moderate_v1` — Kelly-aware moderate reward (12 terms)
