# v22 Training Complete: Same Bear-Market Collapse

Date: 2026-04-25

## Results

`balanced_rule_residual_v22` (v19 variant with adjusted reward coefficients) completed training.

| Metric | Valid | test_2020 | test_2022 | test_2023_2024 |
|--------|-------|-----------|-----------|----------------|
| Return | +53.7% | +103.2% | **-21.0%** | +48.2% |
| Sharpe | 1.65 | 2.30 | **-0.76** | 0.72 |
| MaxDD | 26.0% | 20.8% | 30.1% | 35.7% |
| AvgPos | 0.54 | 0.49 | 0.27 | 0.50 |
| StressDays | — | 48 | 129 | 64 |

## Key Findings

1. **Proxy MaxDD 48.7%** — way above acceptable threshold, model is too risky even on aggregate
2. **test_2022 Sharpe -0.76** — identical bear-market collapse pattern to v19 (-0.80) and balanced (-0.91)
3. **test_2023_2024 Sharpe 0.72** — decent recovery window but MaxDD 35.7% is high

## Conclusion

Reward coefficient adjustments on the residual family do NOT solve the fundamental bear-market generalization problem. This is the **third consecutive confirmation** of family-level failure (balanced, v19, v22).

## Decision

- `balanced_rule_residual_v22`: **dropped**
- No new residual-family experiments until bear-market-aware training is implemented

## Next Direction

Priority is bear-market generalization:
1. **Data**: Include longer bear history or synthetic bear episodes
2. **Sampling**: Oversample stress episodes during training
3. **Reward**: Add bear-market-specific penalties
4. **Features**: Add leading indicators (credit spreads, yield curve)
