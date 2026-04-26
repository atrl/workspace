# Structural MaxDD Reduction Analysis

## Date
2026-04-26

## Goal
Reduce 2022-style bear market MaxDD from 46-52% to <40% without erasing recovery participation.

## Canonical Baseline
balanced_rule_residual_v19 (2022 MaxDD 51.1%, return -47.0%)

## Direct Comparison Baseline
balanced_rule_residual_v19_reg (2022 MaxDD 46.2%, return -40.5%)

## Hypothesis
The 2022 MaxDD is driven by three factors:
1. **Bear min cap too high**: Current `bear_min_cap=0.3` means even in deep bear, strategy holds 30% position
2. **Late regime detection**: Actor may not reduce position fast enough at regime transition
3. **No stop-loss mechanism**: Strategy has no hard stop on single-day losses

Reducing `bear_min_cap` to 0.15-0.20 should reduce MaxDD by 5-10pp while preserving upside capture.

## Closest Historical Success
- v19_reg: MaxDD 46.2% (5pp better than v19) via dropout/weight decay regularization

## Closest Historical Failure
- v19_small: Network simplification made metrics worse (not a cap issue)

## Target Improvement Window
wf_2022: reduce MaxDD from 51% to <40% while keeping return > -50%

## Accepted Trade-off
- 2020 recovery may be slightly slower if bear cap is lower
- 2024 bull participation may decrease

## External Reference
- Position sizing literature: lower minimum caps in high-volatility regimes
- Kelly Criterion: optimal position scales with edge/volatility ratio

## Implementation Plan
1. ✅ Analyze v19 2022 daily position/return pattern to identify worst drawdown days
2. ✅ Test `min_cap` values: 0.40 (baseline), 0.35, 0.30, 0.25, 0.20
3. ✅ Run backtest_profile on all walk-forward windows
4. Select best balance of MaxDD reduction vs return preservation
5. If promising, create new profile and train actor with adjusted cap config

## Evidence

### 2022 Daily Pattern Analysis
- MaxDD -51.1% peaks in Oct, but **accumulated from Jan-Feb crash**
- Jan-Feb: -26.3% return, avg position 0.379 (too high for crash)
- Oct: +1.1% return, avg position 0.276 (already defensive)
- **Problem is early-year delay, not Oct persistence**
- Regime_risk threshold 0.65 is too high: Jan 5 regime=0.604 but cap still at 1.0

### min_cap Sensitivity (same v19 actor, varying cap config)

| min_cap | wf_2020 Return | wf_2022 Return | wf_2022 MaxDD | wf_2024 Return | wf_2024 MaxDD |
|---------|---------------|---------------|---------------|---------------|---------------|
| 0.40 | +113.4% | -47.0% | 51.1% | +28.8% | 34.4% |
| 0.35 | +113.4% | -46.7% | 50.9% | +28.8% | 34.4% |
| 0.30 | +113.4% | -46.1% | 49.9% | +28.5% | 34.4% |
| 0.25 | +110.1% | -42.1% | 48.4% | +27.9% | 34.0% |
| **0.20** | **+106.7%** | **-41.9%** | **46.7%** | **+29.8%** | **32.1%** |

**Key Finding**: `min_cap` is the ONLY effective lever. Lowering from 0.40 to 0.20:
- Reduces 2022 MaxDD by 4.4pp (51.1% → 46.7%)
- Improves 2022 return by 5.1pp (-47.0% → -41.9%)
- Improves 2024 return by 1.0pp (+28.8% → +29.8%)
- Costs only 6.7pp in 2020 bull (+113.4% → +106.7%)

**Surprise**: `bear_regime_threshold` (0.65→0.50), `start_drawdown`, `curve_power` have negligible effect.

## Next Action
Create new profile `balanced_rule_residual_v19_lowcap` with min_cap=0.20 and run full training.
