---
title: balanced_rule_residual_v19_academic
direction: model
status: candidate
---

# balanced_rule_residual_v19_academic

## Hypothesis

See profile.yaml for configuration delta.

## Decision

Candidate. Details in runtime/model/reviews/ and docs/experiment-landscape-review.md.

## Expectations

# Expectations: balanced_rule_residual_v19_academic

## Acceptance Criteria

- proxy return >= 500%
- proxy MaxDD <= 52%
- test_2022 MaxDD <= 35%
- Must beat v19 on at least 2 of: proxy return, proxy Sharpe, test_2022 MaxDD

## Expected Trade-off

- If academic signals are useful: +20-50pp proxy return, tighter drawdown
- If academic signals are noise: no improvement or slight degradation due to expanded obs dim

## Failure Modes

- No improvement over v19 → academic_signals pack is not valuable for SOXL RL
- Training instability → obs dim expansion from 56 → 63 hurts GRU learning

## Comparison-Table Row Key

`balanced_rule_residual_v19_academic`
