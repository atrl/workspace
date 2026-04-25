---
title: risk_budget_usage_v1
direction: model
status: candidate
---

# risk_budget_usage_v1

## Motivation

Test budget-usage architecture: slow mode router owns risk band, fast actor
decides usage ratio inside the band.

## Hypothesis

Mapping `imitation_v4` into mode-owned risk bands keeps upside participation
while preventing 2022 crisis drawdown reopening.

## Result

| Slice | Return | Sharpe | MaxDD | avg_pos |
|-------|--------|--------|-------|---------|
| test_proxy | 274.4% | 0.942 | 32.5% | 0.269 |
| test_2020 | 40.9% | 1.609 | 23.7% | 0.274 |
| test_2022 | -17.3% | -1.213 | 18.7% | 0.127 |
| test_2023_2024 | 82.8% | 1.440 | 16.3% | 0.295 |

## Decision

**Candidate.** Architecture valid but PPO discovers conservative policy.
Replay-scale not reproduced by RL training.

## Evidence Links

- Full review: `runtime/model/reviews/risk_budget_usage_v1_full_review.md`
- Landscape: `docs/experiment-landscape-review.md`
