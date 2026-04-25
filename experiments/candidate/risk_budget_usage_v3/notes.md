---
title: risk_budget_usage_v3
direction: model
status: candidate
---

# risk_budget_usage_v3

## Hypothesis

See profile.yaml for configuration delta.

## Decision

Candidate. Details in runtime/model/reviews/ and docs/experiment-landscape-review.md.

## Expectations

# Expectations: risk_budget_usage_v3

## What we expect to improve

1. **Training efficiency**: mode-conditioned input should reduce the samples
   needed to learn mode-specific policies.
2. **Proxy Sharpe**: from 0.942 → 0.95+ via better mode specialization.
3. **Mode transition smoothness**: Actor sees mode changes explicitly, reducing
   transition lag.

## What we accept may get worse

1. **Overfitting to mode labels**: if mode router is imperfect, the Actor may
   overfit to mode noise.
2. **Observation dimension**: +4 dims may slightly slow training.

## Failure modes to watch

1. **Mode router errors propagate**: if the rule-based mode router misclassifies,
   the Actor receives wrong mode signal.
2. **Learned router incompatibility**: if we later switch to learned router,
   the mode one-hot distribution may shift.

## Abort criteria

- test_proxy Sharpe < 0.90
- test_2022 MaxDD > 25%
