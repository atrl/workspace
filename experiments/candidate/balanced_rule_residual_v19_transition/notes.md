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

All features verified orthogonal to existing CORE_FEATURE_COLUMNS (|rho| < 0.80).

## Hypothesis

RL can combine transition-state signals with existing regime features to:
1. Enter positions earlier during stress-release (higher returns)
2. Avoid false rebounds (better drawdown control)
3. Improve test_2022 performance (the critical weakness)

## Mechanism delta vs v19

- Feature pack: add `transition_signals` (4 features)
- All other hyperparameters identical to v19
- Feature count: 59 → 63 (+4)

## Acceptance criteria

- proxy return >= 500%
- proxy MaxDD <= 52%
- test_2022 MaxDD <= 35%
- Must beat v19 on at least 2 of: proxy return, proxy Sharpe, test_2022 MaxDD

## Expected trade-off

- If transition signals are useful: +50-100pp proxy return, tighter drawdown, especially in 2022-like stress-release windows
- If transition signals are noise: no improvement or slight degradation

## Failure mode

- No improvement over v19 → transition-state hypothesis rejected for SOXL
- Training instability due to mixed continuous/binary features
