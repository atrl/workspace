# Balanced Rule Residual V2

Balanced-family candidate enabling Kelly-log alpha reward.

- Baseline: `balanced_rule_residual_v1`
- Delta from v1:
  - `use_kelly_alpha: true` (was `false` in v1)
  - All other params identical to v1 for clean ablation
- Reward intent:
  - Retain the residual-over-rule architecture
  - Switch from linear alpha to log-alpha: $R_t = \log(1 + \alpha_t) \times 100 \times alpha\_scale$
  - This is the growth-optimal reward formulation (Kelly criterion)
  - Expected effect: smoother position transitions, less extreme behavior in volatile regimes
- Success criteria:
  - valid Sharpe >= 1.65 (v1 = 1.69)
  - test_proxy Sharpe >= 0.95 (v1 = 0.96)
  - trade-quality buckets remain positive Kelly
  - fixed-split promotion gate still passes
