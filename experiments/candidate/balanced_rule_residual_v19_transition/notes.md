# balanced_rule_residual_v19_transition

## Motivation

The core feature stack is excellent at detecting *persistent stress* (drawdown, vol, regime_risk) but weak at detecting *stress release* and *restart transitions*.

Phase 1 academic factor mining failed because classic academic factors are already linearly captured by existing features.

This experiment tests a new direction: **transition-state features** that detect when selling pressure is exhausting and a restart/bounce is likely.

## Features tested and promoted

| Feature | Type | 5d IC | Standalone PF | Accuracy | Status |
|---------|------|-------|---------------|----------|--------|
| TRANS_VIX_DIVERGENCE | Continuous | +0.21 | 1.43 | 64.2% | ✅ Promoted |
| TRANS_SPRING_COMPRESSION | Binary | +0.12 | 1.41 | 58.3% | ✅ Promoted |
| TRANS_PANIC_EXHAUSTION | Binary | +0.08 | 1.46 | 60.8% | ✅ Promoted |
| TRANS_GAP_RECOVERY | Binary | +0.06 | 1.51 | 58.3% | ✅ Promoted |
| TRANS_VOL_TERM_ZSCORE | Continuous | +0.08 | 1.15 | 56.7% | ✅ Promoted |
| TRANS_VOV_TREND | Continuous | +0.13 | 1.28 | 57.1% | ✅ Promoted |
| TRANS_TE_ZSCORE | Continuous | +0.07 | 1.18 | 55.2% | ✅ Promoted |

All features verified orthogonal to existing CORE_FEATURE_COLUMNS (|rho| < 0.80).

## Hypothesis

RL can combine transition-state signals with existing regime features to:
1. Enter positions earlier during stress-release (higher returns)
2. Avoid false rebounds (better drawdown control)
3. Improve test_2022 performance (the critical weakness)

## Mechanism delta vs v19

- Feature pack: add `transition_signals` (7 features)
- All other hyperparameters identical to v19
- Feature count: 59 → 66 (+7)

## Results

### 20-iteration quick training (seed=12345, Phase 1 only)

| Metric | v19_transition (20 iters) | v19 baseline (80 iters) | Notes |
|--------|---------------------------|------------------------|-------|
| Valid return | 50.6% | ~55% | Comparable to v19 baseline |
| Valid Sharpe | 1.39 | ~1.42 | Slightly below |
| Valid MaxDD | 31.2% | ~28% | Slightly worse |
| Valid best_score | 74.58 | ~78-80 | Below baseline |
| **Test return** | **613.0%** | **~632.4%** | Within 3% of baseline |
| **Test Sharpe** | **0.86** | **0.91** | Lower |
| **Test MaxDD** | **57.6%** | **50.1%** | **+7.5pp worse** |
| Trades | 307 | ~290 | Slightly more |
| Promotion | **BLOCKED** | **PASSED** | MaxDD > 52% |

### 5-iteration smoke test (seed=20260423)

| Metric | v19_transition (5 iters) | v19 baseline (5 iters) |
|--------|--------------------------|------------------------|
| Test return | 448.3% | 469.5% |
| Test Sharpe | 0.85 | 0.84 |
| Test MaxDD | 46.9% | 49.4% |

## Analysis

1. **No material improvement over v19**: Test return 613% vs 632% is within noise. Sharpe 0.86 vs 0.91 is slightly worse.

2. **MaxDD regression is the critical issue**: 57.6% vs 50.1% (+7.5pp) is a significant degradation. The transition signals, despite having positive IC, do not help the RL agent control drawdown.

3. **Why positive IC did not translate to better RL performance**:
   - Transition features detect *potential* stress-release windows, but timing precision matters
   - The RL agent may be using these signals to enter too early during persistent stress, getting caught in continued declines
   - Binary features (SPRING_COMPRESSION, PANIC_EXHAUSTION, GAP_RECOVERY) provide sparse signals that may not align well with the agent's decision frequency
   - Additional feature dimensions (59→66) may increase model capacity requirements without proportional benefit at 20 iterations

4. **Phase 1 only**: Full staged training is 80 iterations. The 20-iter result may not reflect the full potential, but the MaxDD trajectory is concerning.

## Conclusion

**REJECTED as candidate for promotion.**

Transition-state features show positive standalone IC and are orthogonal to existing features, but they do not translate to improved RL performance. The MaxDD regression is the decisive factor.

## Next action

1. Do not proceed with full 80-iteration training
2. Archive this experiment as `ablated`
3. Transition feature pack remains available in `src/soxl_quant/data.py` but is not enabled in canonical profiles
4. Future experiments should test transition features in isolation (e.g., only 1-2 strongest features) rather than the full 7-feature pack
5. The `TRANS_VIX_DIVERGENCE` feature (IC +0.21) may warrant a separate minimal experiment
