---
title: risk_budget_usage_v3
direction: model
status: candidate
---

# risk_budget_usage_v3

## Motivation

v1 proved the budget-usage architecture works. v2 added mode-conditional reward
shaping and dynamic band scaling to guide PPO behavior. v3 goes one step
deeper: **let the Actor directly observe the current mode**, so it can learn
mode-specific usage policies end-to-end.

The hypothesis is that feeding mode as an explicit input feature (one-hot)
should let the network learn different usage distributions per mode faster
than inferring mode indirectly from raw features alone.

## External Reference

- Mode-conditioned policies in hierarchical RL (Sutton et al. 1999)
- Task-conditional policy networks (Hausman et al. 2018)
- Contextual policies with one-hot task identifiers

## Hypothesis

Appending the current mode as a one-hot vector to the observation should:
1. Reduce the sample complexity of learning mode-specific usage policies
2. Improve generalization across mode transitions
3. Lift proxy Sharpe by allowing the Actor to specialize per mode

## Target Slices / Windows

- `test_proxy`: Sharpe ≥ 0.95
- `test_2022`: MaxDD ≤ 18%, avg_pos ≤ 0.15
- `test_2023_2024`: return ≥ 80%

## Acceptance Criteria

- `test_proxy Sharpe >= 0.95`
- `test_2022 MaxDD <= 20.0%`
- `test_2022 avg_pos <= 0.15`

## Baseline

- **Canonical baseline**: `risk_budget_usage_v1`
- **Direct comparison baseline**: `risk_budget_usage_v2`
- **Experiment role**: `replace`
- **Changed layer**: `network` (observation input)

## Configuration Delta vs v2

```yaml
model:
  use_mode_conditioned_input: true  # NEW: append 4-dim mode one-hot to obs
```

All other configs identical to v2.

## Result

(TBD — run training and fill in)

## Decision

(TBD)

## Comparison-Table Row Key

`risk_budget_usage_v3`
