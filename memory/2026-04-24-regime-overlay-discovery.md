---
date: '2026-04-24'
title: "2026-04-24: Regime-Conditional Overlay Discovery \u2014 vol_stress"
kind: review
profiles: []
status: kept
tags:
- execution-overlay
- kept
- review
- stress-regime
- transition-features
- walk-forward
artifacts:
- runtime/post_train_review/regime_conditional_ic_analysis.md
- runtime/post_train_review/regime_overlay_test_results.md
workpad: codex_exec_plans/active/2026-04-10-soxl-agent-loop.md
summary: Legacy memory note normalized into the repo-wide schema. After regime-conditional
  IC analysis revealed that feature predictive power is highly regime-dependent, we
  tested execution overlays using existing features with regime gating. **Majo
---

# 2026-04-24: Regime-Conditional Overlay Discovery — vol_stress

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

## Discovery

After regime-conditional IC analysis revealed that feature predictive power is
highly regime-dependent, we tested execution overlays using existing features
with regime gating.

**Major finding**: `vol_stress` overlay (vol_of_vol_21 >= 0.20 in stress regime)
achieves **+128pp return improvement** (452.3% → 580.4%) with **no MaxDD
degradation** (45.9% vs 46.7%).

This is the strongest overlay discovered to date, surpassing the previous best
`vix_divergence` (+61pp).

## Why This Works

1. **vol_of_vol_21 has strong regime-conditional IC**: 0.174 in stress regime
2. **Actor underweights it globally**: The actor trains on all regimes, so it
   gives vol_of_vol_21 a moderate average weight that doesn't capture its
   stress-regime predictive power
3. **Overlay provides regime-specific boost**: The regime gating (regime_risk >=
   0.6) ensures the boost only applies when the signal is most predictive

## Why Other Overlays Underperform

- **downside_bull**: Actor already uses downside_ratio_10 effectively
- **vol_vix_combo**: Requiring both signals to agree is too restrictive
- **rv63_stress**: Raw volatility level is less informative than vol-of-vol

## Comparison to v19_transition (Rejected)

| Metric | Baseline | v19_transition | vol_stress overlay |
|--------|----------|----------------|-------------------|
| Return | 452.3% | 518.8% | **580.4%** |
| Sharpe | 0.87 | 0.82 | **0.94** |
| MaxDD | 46.7% | 60.9% | **45.9%** |

Overlay achieves better results than retraining with new features, with lower
risk and no retraining cost.

## Files Updated

- `src/soxl_quant/backtest.py`: Added `regime_mode`, `vol_stress`,
  `downside_bull`, `dual_regime`, `vol_vix_combo`, `rv63_stress` overlay types
- `scripts/test_transition_overlay.py`: Updated to support new overlay types
- `runtime/post_train_review/regime_conditional_ic_analysis.md`: Regime IC analysis
- `runtime/post_train_review/regime_overlay_test_results.md`: Complete overlay results
- `docs/experiment-landscape-review.md`: Added overlay comparison section

## Next Actions

1. Walk-forward slice validation for vol_stress
2. Production runtime integration
3. Consider training new actor with regime-aware feature weighting

## Update: dual_kurtosis — +236pp

After testing realized_kurtosis_21 in bull regime, we combined it with
vol_stress to create a dual-regime overlay:

**Result: 688.2% / Sharpe 0.99 / MaxDD 44.9% — +236pp vs baseline**

This is the strongest result achieved in the entire harness to date.

### Component breakdown

| Component | Return | Improvement |
|-----------|--------|-------------|
| Baseline | 452.3% | — |
| + vol_stress (stress) | 580.4% | +128pp |
| + kurtosis_bull (bull) | 548.0% | +96pp |
| + dual_kurtosis (both) | 688.2% | +236pp |

Near-linear叠加 confirms the two signals capture non-overlapping predictive
power in different regimes.

### Production config

```yaml
overlay:
  type: "dual_kurtosis"
  regime_mode: "dual"
  regime_threshold: 0.6
  bull_regime_threshold: 0.4
  threshold: 0.20
  bull_threshold: 0.8
  boost: 0.10
```

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
