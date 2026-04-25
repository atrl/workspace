---
title: balanced_rule_residual_moderate_v1
direction: model
status: candidate
---

# balanced_rule_residual_moderate_v1

## Hypothesis

See profile.yaml for configuration delta.

## Decision

Candidate. Details in runtime/model/reviews/ and docs/experiment-landscape-review.md.

## Expectations

# Expectations: balanced_rule_residual_moderate_v1

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

## Failure Modes

- Step-level penalties too weak → MaxDD > 52%
- trend_alignment insufficient vs activity_bonus → return < 500%
- Missing rule_deviation → excessive deviation from rule causes choppy behavior

## Comparison-Table Row Key

`balanced_rule_residual_moderate_v1`
