---
title: balanced_rule_residual_moderate_v1
direction: model
status: candidate
---

# balanced_rule_residual_moderate_v1

## Motivation

Find a "principled" moderate reward (~12 terms, down from ~17) that matches v16
performance by keeping proven drivers and removing counterproductive terms.

## Mechanism Delta vs v16

| Layer | v16 | moderate_v1 |
|-------|-----|-------------|
| Core signal | alpha (Kelly) + PBRS | same |
| Step risk | drawdown=0.28, downside=0.20, risk_budget=0.24 | **REDUCED**: 0.15, 0.10, 0.15 |
| Participation | activity_bonus=1.25 + inactivity=0.80 | **REPLACED**: trend_alignment=0.25 |
| Removed | — | rule_deviation, short_hold_turnover, flat_trade |

## Acceptance Criteria

- proxy return >= 550%
- proxy MaxDD <= 50%
- test_2022 MaxDD <= 35%
- seed robustness: <=150pp swing

## Evidence Links

- Landscape: `docs/experiment-landscape-review.md`
