# 2026-04-24: Multi-Factor Overlay Tests

## Question

Does combining multiple predictive features into a composite score outperform
regime-separated single-factor overlays?

## Answer: No

### Multi-factor within same regime → underperforms

| Composite | Return vs Baseline |
|-----------|-------------------|
| vol + vix (stress) | +25pp |
| vol + vix + te (stress) | +28pp |
| vol_z + vix_z (stress) | **-8pp** |

### Multi-factor across regimes (dual_kurtosis) → outperforms

| Overlay | Return vs Baseline |
|---------|-------------------|
| vol_stress (stress only) | +128pp |
| kurtosis_bull (bull only) | +96pp |
| **dual_kurtosis (both)** | **+245pp** |

## Why

- **Within-regime features are correlated**: vol_of_vol_21 ρ with VIX divergence ≈ 0.65
- **Cross-regime features are uncorrelated**: vol (stress) and kurtosis (bull) capture different phenomena
- **Correlation kills composite alpha**: Combining correlated signals dilutes rather than amplifies

## Lesson

For overlay design:
1. Find the single best signal PER regime
2. Apply regime gating
3. Do NOT combine multiple signals within the same regime

The "multi-factor" intuition (more features = better) is wrong for execution overlays
when features are correlated. Regime separation is the key edge.
