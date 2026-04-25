---
title: smoke_rule_residual_v1
direction: model
status: candidate
---

# Smoke Rule Residual V1

Candidate smoke profile for the first residual-over-rule experiment.

- Baseline: `smoke`
- Purpose:
  - keep rule-based position as the anchor
  - let RL output only a bounded residual delta
  - reward RL for incremental alpha vs the rule while penalizing rule deviation and micro churn
- Scope:
  - smoke-only structure proof
  - not a production or promotion candidate

## Expectations

# Expectations: smoke_rule_residual_v1

## Acceptance Criteria

- Training completes without harness breakage
- Smoke backtest produces non-NaN results
- Rule residual mechanism is functional (position = rule + bounded delta)

## Expected Trade-off

- Smoke profiles are intentionally lightweight (2 iters, 2 episodes)
- Not meant to match production quality metrics
- Purpose is structural validation only

## Failure Modes

- Rule residual overlay breaks position calculation → NaN or crash
- Reward terms incompatible with smoke scale → training divergence
- Profile config missing required fields → harness error

## Comparison-Table Row Key

`smoke_rule_residual_v1`
