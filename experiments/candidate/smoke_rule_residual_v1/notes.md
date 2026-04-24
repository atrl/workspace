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
