# Balanced Rule Residual V1

Balanced-family candidate for the first full residual-over-rule training run.

- Baseline: `balanced`
- Structure:
  - rule-based base position
  - RL learns only a bounded delta around the base
- Reward intent:
  - retain `balanced` path-quality style
  - pay RL for incremental alpha vs the rule
  - penalize large rule deviation and weak micro churn
- Success is not "more return at any cost".
- Success means:
  - trade-quality improves directionally
  - fixed-split promotion quality stays coherent
