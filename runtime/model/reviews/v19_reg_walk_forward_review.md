# v19_reg Walk-Forward Review

Date: 2026-04-25

## Results

| Window | Return | Sharpe | MaxDD | AvgPos |
|--------|--------|--------|-------|--------|
| wf_2020 | +95.3% | 2.20 | 20.8% | — |
| wf_2022 | -23.9% | -0.85 | 32.5% | 0.278 |
| wf_2023_2024 | -18.1% | -0.31 | 37.2% | 0.439 |

## Comparison with v19 Walk-Forward

| Window | v19 | v19_reg | Delta |
|--------|-----|---------|-------|
| wf_2020 | +125.7% / 2.30 | +95.3% / 2.20 | -30.4pp / -0.10 |
| wf_2022 | -25.2% / -0.80 | -23.9% / -0.85 | +1.3pp / -0.05 |

## Critical Finding: Fixed-Split → Walk-Forward Gap

| Metric | Fixed-Split test_2023_2024 | Walk-Forward wf_2023_2024 | Gap |
|--------|---------------------------|---------------------------|-----|
| Return | +58.7% | -18.1% | -76.8pp |
| Sharpe | 0.80 | -0.31 | -1.11 |
| MaxDD | 35.1% | 37.2% | +2.1pp |

**This is a catastrophic generalization failure.** The model performs well on fixed-split (+58.7%) but loses money on walk-forward (-18.1%).

## Analysis

1. **wf_2020 is good** — Both v19 and v19_reg excel in bull markets (Sharpe 2.20+)
2. **wf_2022 is expected** — In a year where SOXL B&H lost -86.6%, losing -24% is successful defense
3. **wf_2023_2024 is the problem** — The model fails to capture the 2023-2024 recovery when trained on 2010-2022 data

The fixed-split model (trained on 2010-2021, validated on 2022) sees +58.7% in 2023-2024. But the walk-forward model (trained on 2010-2022, validated on 2023) sees -18.1% in 2024.

This suggests the **2023 validation year in walk-forward is critical** — the model's performance on 2023 validation determines its policy for 2024, and if 2023 has patterns not seen in 2010-2022, the model makes wrong decisions.

## Implications

1. **Fixed-split metrics are unreliable** — Even with regularization, fixed-split test_2023_2024 (+58.7%) does NOT predict walk-forward wf_2023_2024 (-18.1%)
2. **Regularization doesn't solve the core problem** — Dropout + weight decay improved proxy Sharpe (0.79→0.91) but didn't prevent walk-forward collapse
3. **The 2023 validation window is a blind spot** — Models trained through 2022 fail to adapt to 2023's regime

## Decisions

- `balanced_rule_residual_v19_reg`: **candidate (walk-forward blocked)** — fixed-split metrics look good but walk-forward wf_2023_2024 fails catastrophically
- Do NOT promote based on fixed-split metrics alone
- Need to understand why 2023 validation causes policy degradation before any new v19 variants

## Next Actions

1. Diagnose why 2023 validation leads to poor 2024 performance
2. Check if 2023 has regime characteristics not present in 2010-2022 training data
3. Consider adding 2023-like synthetic episodes to training
4. Test if the problem is specific to v19_reg or affects all models (run balanced walk-forward on wf_2023_2024)
