# Walk-Forward Completion: v19_small and v22

## Date
2026-04-25

## Goal
Validate whether network simplification (v19_small) or VIX term features (v22) improve walk-forward generalization beyond the v19 baseline.

## Results

### Walk-Forward Comparison Table

| Profile | wf_2020 Ret | wf_2022 Ret | wf_2023_24 Ret | wf_2020 Sharpe | wf_2022 Sharpe | wf_2023_24 Sharpe |
|---------|-------------|-------------|----------------|----------------|----------------|-------------------|
| balanced | 125.7% | -25.8% | -15.2% | 2.30 | -0.91 | -0.22 |
| v19 | 114.9% | -25.2% | -18.7% | 2.35 | -0.80 | -0.23 |
| v19_small | 96.9% | -31.6% | -17.1% | 2.04 | -1.11 | -0.27 |
| v22 | 117.7% | -23.5% | -22.5% | 2.27 | -0.84 | -0.36 |

### Key Findings

1. **ALL models fail walk-forward in bear markets** - no exceptions. Every profile loses money in wf_2022 and wf_2023_2024.

2. **v19_small is WORSE than v19 in every way**:
   - wf_2022: -31.6% vs -25.2% (v19), Sharpe -1.11 vs -0.80
   - wf_2023_2024: -17.1% vs -18.7% (v19) - slightly better but still bad
   - Network simplification (hidden=[32], gru=16) made bear-market performance significantly worse

3. **v22 shows marginal 2022 improvement but worse 2023-2024**:
   - wf_2022: -23.5% vs -25.2% (v19) - best bear-market return, but still deeply negative
   - wf_2023_2024: -22.5% vs -18.7% (v19) - WORSE than v19
   - VIX term features provide marginal 2022 improvement but degrade 2023-2024 performance

4. **balanced is still the "best" for 2023-2024** (-15.2%), but all are negative

5. **Fixed-split metrics are unreliable predictors**:
   - v19 fixed-split proxy Sharpe: 0.97 → walk-forward wf_2022 Sharpe: -0.80
   - v22 fixed-split 2022 Sharpe: -0.76 → walk-forward wf_2022 Sharpe: -0.84 (similar)
   - v19_small fixed-split 2022 Sharpe: -1.11 → walk-forward wf_2022 Sharpe: -1.11 (identical)

## Conclusions

- **Overfitting is NOT a capacity problem**: Simplifying the network (v19_small) made performance worse, not better. The model needs capacity but is overfitting due to insufficient regularization, data diversity, or reward design.
- **VIX term features do not solve generalization**: Marginal improvement in 2022 (-23.5% vs -25.2%) is not meaningful, and 2023-2024 degradation (-22.5% vs -18.7%) is worse.
- **The fundamental problem is bear-market generalization**: All models perform well in bull markets (wf_2020 Sharpe 2.0+) but fail catastrophically in bear markets.
- **Walk-forward is the ONLY valid promotion standard**: Fixed-split metrics (even test_2022/test_2023_2024) are not reliable predictors of true generalization.

## Next Actions

1. Do NOT run more network-size ablations (v19_tiny should not be launched)
2. Do NOT run more feature-pack variants without a fundamental change
3. Focus on mechanisms that explicitly address bear-market generalization:
   - Bear-market-aware reward shaping (e.g., penalize losses asymmetrically)
   - Data augmentation or adversarial training
   - Explicit regime detection and regime-conditioned policies
   - Curriculum learning that includes bear-market episodes
   - Regularization techniques (dropout, weight decay, early stopping on validation)
4. Before any new experiment, must cite external reference and explain why it should improve bear-market generalization specifically

## Files Changed
- `scripts/build_backtest_dashboard.py`: Fixed `RUNTIME_DIR` NameError
- `docs/experiment-landscape-review.md`: Updated v19, v19_small, v22 rows with walk-forward data
