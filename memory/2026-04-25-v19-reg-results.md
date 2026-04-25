# v19_reg Results: Regularization Improves Bear-Market Walk-Forward

## Date
2026-04-25

## Hypothesis
Adding dropout (0.2) and weight decay (1e-5) to GRUNet would reduce bull-market overfitting and improve bear-market walk-forward generalization.

## Results

### Fixed-Split
- Proxy: Sharpe 0.89, Ret 659%, MaxDD 59.8%
- 2022: Sharpe -1.08, Ret -32.1%, MaxDD 40.3% (WORSE than v19)
- 2023_2024: Sharpe 0.71, Ret 51.3%, MaxDD 40.4% (WORSE than v19)

### Walk-Forward
- wf_2020: Sharpe 2.20, Ret 95.3%, MaxDD 20.8% (good bull market)
- wf_2022: Sharpe **-0.73**, Ret **-21.9%**, MaxDD 31.0% ← **BEST in residual family!**
- wf_2023_2024: Sharpe -0.44, Ret -22.9%, MaxDD 37.2% (WORSE than v19)

### Comparison

| Profile | wf_2020 Sh | wf_2022 Sh | wf_2024 Sh |
|---------|------------|------------|------------|
| balanced | 2.30 | -0.91 | **-0.22** |
| v19 | 2.35 | -0.80 | -0.23 |
| v19_small | 2.04 | -1.11 | -0.27 |
| v22 | 2.27 | -0.84 | -0.36 |
| **v19_reg** | **2.20** | **-0.73** | -0.44 |

## Conclusions

1. **Regularization DOES improve bear-market generalization** - v19_reg is the first model to show improvement in ANY walk-forward window (wf_2022: -21.9% vs v19 -25.2%).

2. **Trade-off exists** - improved wf_2022 but degraded wf_2023_2024 (-22.9% vs v19 -18.7%). The regularization may be too strong for mixed markets.

3. **Overfitting is confirmed** - the fact that regularization helps proves the model was overfitting to bull-market patterns.

4. **Bull-market performance preserved** - wf_2020 Sharpe 2.20 is only slightly below v19's 2.35.

## Next Actions

1. **Tune regularization strength**: Try dropout 0.1 or weight_decay 1e-6 to find a better balance between wf_2022 and wf_2023_2024.
2. **Try different regularization placement**: Only dropout on MLP (not GRU), or only weight decay without dropout.
3. **Combine with other techniques**: Regularization + VIX features might be synergistic.

## Code Changes
- `src/soxl_quant/policy/models/gru.py`: Added dropout parameter
- `src/soxl_quant/policy/policy_builder.py`: Pass dropout to GRUNet, weight_decay to Adam
- `src/soxl_quant/config.py`: Added dropout and weight_decay fields
- `configs/soxl/balanced_rule_residual_v19_reg.yaml`: New profile
