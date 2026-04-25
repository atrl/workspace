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

## Hypothesis
Adding dropout (0.2) and weight decay (1e-5) to the GRU+MLP network will improve proxy-period robustness by reducing MaxDD.

## Closest Historical Success
- v19: Fixed-split gate passed (proxy Sharpe 0.79, test_2022 MaxDD 32.4%)

## Closest Historical Failure
- v19_small: Network simplification made metrics worse

## Implementation Steps
1. ✅ Modify GRUNet to accept dropout parameter
2. ✅ Modify policy_builder.py to pass weight_decay to Adam optimizer
3. ✅ Create profile `balanced_rule_residual_v19_reg.yaml`
4. ✅ Run fixed-split training
5. ✅ Run walk-forward
6. Compare to v19 baseline

## Results

### Fixed-Split

| Metric | v19 | v19_reg | Delta |
|--------|-----|---------|-------|
| Proxy return | 417.7% | 532.7% | +115pp |
| Proxy Sharpe | 0.79 | **0.88** | +0.09 |
| test_2022 MaxDD | 32.4% | 30.2% | -2.2pp |
| test_2022 return | -25.2% | -21.9% | +3.3pp |
| test_2023_2024 return | 79.4% | 58.7% | -20.7pp |

### Walk-Forward (Main Backtest)

| Window | v19 | v19_reg | Assessment |
|--------|-----|---------|------------|
| wf_2020 | +184.0%, MaxDD 76.7% | +181.3%, MaxDD 76.7% | Similar |
| wf_2022 | +67.6%, MaxDD 70.3% | +42.2%, MaxDD 61.6% | v19 better return |
| wf_2023_2024 | — | -7.1%, MaxDD 43.4% | v19_reg fails |
| Promotion | blocked | blocked | Both blocked |

### Critical Finding: Worker Eval ≠ Backtest

There's a massive gap between worker internal evaluation and main backtest:
- v19_reg wf_2022 worker eval: -23.9%
- v19_reg wf_2022 main backtest: +42.2%
- Gap: **66.1 percentage points**

The worker evaluation is NOT a reliable predictor of backtest performance.

## Conclusion

**v19_reg does NOT improve over v19.**
- Fixed-split: slightly better proxy Sharpe (0.88 vs 0.79)
- Walk-forward wf_2022: worse return (+42.2% vs +67.6%)
- Walk-forward wf_2023_2024: fails (-7.1%)
- MaxDD remains extremely high (61-77%)

Regularization is not the solution. The core problems are:
1. Worker evaluation systematically differs from backtest
2. MaxDD 61-77% is structurally too high
3. 2023 regime shift causes policy failure

## Decision

- `balanced_rule_residual_v19_reg`: **candidate (walk-forward blocked)**
- v19 remains the stronger baseline

## Next Actions
1. Fix worker evaluation to align with backtest
2. Address structural MaxDD (position caps, stop losses)
3. Analyze 2023 regime shift
