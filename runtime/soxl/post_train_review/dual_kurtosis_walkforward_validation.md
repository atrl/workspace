# dual_kurtosis Overlay Walk-Forward Validation

Date: 2026-04-26
Method: Fixed v19 actor + dual_kurtosis overlay, run on walk-forward windows

## Results

| Window | Baseline Return | Baseline Sharpe | Baseline MaxDD | Overlay Return | Overlay Sharpe | Overlay MaxDD | Delta Return | Delta MaxDD |
|--------|-----------------|-----------------|----------------|----------------|----------------|---------------|--------------|-------------|
| wf_2020 | 92.8% | 1.96 | 20.0% | 111.3% | 2.03 | 20.0% | **+18.5pp** | -0.0pp |
| wf_2022 | -42.1% | -1.28 | 47.5% | -42.1% | -1.28 | 47.5% | 0.0pp | 0.0pp |
| wf_2023_2024 | 25.5% | 0.67 | 29.1% | 24.2% | 0.62 | 33.9% | -1.3pp | **+4.8pp** |

## Critical Finding

**Walk-forward validation REJECTS dual_kurtosis overlay as a production candidate.**

### Why fixed-split showed +245pp but walk-forward fails

1. **wf_2020 (COVID crisis)**: Overlay works (+18.5pp)
   - vol_stress triggers during March 2020 crash and subsequent rebounds
   - This is the "ideal" scenario for the overlay

2. **wf_2022 (Bear market)**: Overlay is completely inactive
   - vol_of_vol_21 and kurtosis signals don't trigger in the way expected
   - Baseline is -42.1%, overlay doesn't help at all

3. **wf_2023_2024 (Recovery/bull)**: Overlay slightly hurts
   - kurtosis_bull triggers but adds risk without commensurate return
   - MaxDD worsens by +4.8pp

### Root Cause Analysis

The fixed-split result (2020-2026) is dominated by:
- 2020 COVID crash/rebound (+92.8% → +111.3%)
- 2021 bull market (where kurtosis_bull likely contributed)
- These large positive years mask the lack of benefit in 2022 and slight harm in 2024

**The overlay is window-dependent and not robust across regimes.**

## Implications

1. **Do NOT deploy dual_kurtosis overlay to production**
2. **Walk-forward validation is essential** — fixed-split results can be misleading
3. The +245pp fixed-split result is not representative of out-of-sample performance

## Next Steps

1. Investigate why overlay is inactive in 2022
2. Consider regime-specific parameter tuning per window
3. Re-evaluate whether ANY overlay can pass walk-forward validation

## Root Cause: Why Overlay Fails in 2022

### 2022 vs 2020 Comparison

| Metric | 2020 (COVID) | 2022 (Bear) |
|--------|-------------|-------------|
| Stress days | 38.3% (97/253) | 78.5% (197/251) |
| Bull days | 24.5% | 0.0% |
| vol_of_vol_21 mean | **0.186** | **0.106** |
| vol_stress triggers | **58 days** | **10 days** |
| kurtosis_bull triggers | 11 days | 0 days |

### Key Insight

**vol_of_vol_21 is NOT a bear-market indicator — it is a crisis-recovery indicator.**

- **2020 (COVID)**: Market crashed rapidly then rebounded → vol_of_vol spiked → overlay triggered 58 times → +18.5pp
- **2022 (Slow bear)**: Market declined gradually, volatility was elevated but STABLE → vol_of_vol stayed low → overlay triggered only 10 times → no improvement

The overlay only helps in **rapid crash + rapid recovery** scenarios (V-shaped bottoms).
It does NOT help in **slow grinding bear markets** (L-shaped or U-shaped bottoms).

### 2024 Degradation

In 2024 (bull regime), kurtosis_bull triggered but:
- Added +4.8pp MaxDD without commensurate return gain
- The kurtosis threshold (0.8) captures tail-risk events that sometimes precede dips
- Overlay boosts position before dips → increases drawdown

## Revised Conclusion

**dual_kurtosis overlay is a conditional tool, not a universal improvement.**

Effective in: V-shaped market crises (2020-style)
Ineffective in: Slow bears (2022-style), grinding recoveries (2024-style)

### Production Implications

Option 1: **Do not deploy** — overlay lacks walk-forward robustness
Option 2: **Conditional deployment** — only activate overlay when vol_of_vol_21 > historical p90
Option 3: **Regime-adaptive thresholds** — lower threshold in slow bears, higher in fast crashes
Option 4: **Accept limitation** — use overlay as a "crisis alpha booster" with known window dependency
