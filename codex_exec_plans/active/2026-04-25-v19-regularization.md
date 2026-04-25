# Experiment Plan: v19 + Regularization (dropout + weight decay)

## Date
2026-04-25

## Direction
network

## Canonical Baseline
balanced_rule_residual_v19

## Direct Comparison Baseline
balanced_rule_residual_v19 (no regularization)

## Experiment Role
ablate

## Changed Layer
network (GRUNet architecture + optimizer configuration)

## Target Improvement Window
Full proxy period; focus on reducing proxy MaxDD and improving recovery timing

## Accepted Trade-off
Bull-market performance may decrease slightly if overall robustness improves.

## External Reference
- Srivastava et al. (2014) "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
- Standard PPO/RL practice: weight decay (L2 regularization) prevents overfitting to dominant patterns

## Hypothesis (Updated)

**Previous hypothesis was incorrect.** v19 does NOT "fail" in bear markets — it successfully defends (2022 return -25.2% vs SOXL B&H -86.6%).

**Corrected hypothesis:** Regularization (dropout + weight decay) improves proxy-period robustness by:
1. Reducing proxy MaxDD (v19: 53.9% → target lower)
2. Improving recovery timing in 2023-2024 (v19: +79% vs B&H +182%)

## Closest Historical Success
- v19: Fixed-split gate passed (proxy Sharpe 0.79, test_2022 MaxDD 32.4%)

## Closest Historical Failure
- v19_small: Network simplification made metrics worse, confirming capacity is not the issue

## Why This Is Not a Duplicate
First experiment in the residual family to add explicit regularization (dropout + weight decay).

## Implementation Steps
1. ✅ Modify GRUNet to accept dropout parameter
2. ✅ Modify policy_builder.py to pass weight_decay to Adam optimizer
3. ✅ Create profile `balanced_rule_residual_v19_reg.yaml` with dropout=0.2, weight_decay=1e-5
4. ✅ Run fixed-split training
5. Run walk-forward if fixed-split metrics are competitive
6. Compare to v19 baseline

## Results (Fixed-Split)

| Metric | v19 | v19_reg | Delta |
|--------|-----|---------|-------|
| Proxy Sharpe | 0.79 | **0.91** | +0.12 |
| test_2022 MaxDD | 32.4% | **30.2%** | -2.2pp |
| test_2022 return | -25.2% | -21.9% | +3.3pp |
| test_2023_2024 Sharpe | 0.90 | **0.80** | -0.10 |
| test_2023_2024 return | 79.4% | 58.7% | -20.7pp |

**Observation:** Regularization improves proxy Sharpe and test_2022 MaxDD, but hurts 2023-2024 recovery performance. Trade-off exists.

## Acceptance Criteria
- Fixed-split proxy Sharpe >= 0.80 ✅ (0.91)
- test_2022 MaxDD < 35% ✅ (30.2%)

## Next Action
Run walk-forward to validate generalization. Compare wf_2022 and wf_2023_2024 to v19 baseline.
