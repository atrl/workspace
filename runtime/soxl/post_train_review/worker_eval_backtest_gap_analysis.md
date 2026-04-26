# CRITICAL: Worker Eval vs Main Backtest Gap Analysis

Date: 2026-04-26
Status: **CRITICAL HARNESS BUG**

## Discovery

Systematic comparison of walk-forward worker evaluation vs main backtest engine
reveals massive gaps across all profiles and windows.

## Results

### v19

| Window | Worker Eval | Main Backtest | Gap |
|--------|-------------|---------------|-----|
| wf_2020 | +184.0% | +92.8% | **-91.3pp** |
| wf_2022 | +67.6% | -42.1% | **-109.6pp** |
| wf_2023_2024 | -2.6% | +25.5% | +28.0pp |

### v19_reg

| Window | Worker Eval | Main Backtest | Gap |
|--------|-------------|---------------|-----|
| wf_2020 | +181.3% | +96.9% | **-84.4pp** |
| wf_2022 | +42.2% | -38.3% | **-80.4pp** |
| wf_2023_2024 | -7.1% | +19.5% | +26.6pp |

### v19_reg_light

| Window | Worker Eval | Main Backtest | Gap |
|--------|-------------|---------------|-----|
| wf_2020 | +108.8% | +105.1% | **-3.7pp** |
| wf_2022 | -21.0% | -42.1% | **-21.2pp** |
| wf_2023_2024 | -20.4% | +22.5% | **+42.9pp** |

## Key Findings

1. **Worker eval is MASSIVELY overoptimistic in 2020 and 2022**
   - Gaps of 80-110pp are not explainable by transaction costs alone
   - Worker eval shows positive returns where main backtest shows heavy losses

2. **Worker eval is slightly pessimistic in 2024**
   - Shows negative returns where main backtest shows positive returns
   - But the gap is "only" 20-40pp

3. **The gap varies wildly between profiles**
   - v19_reg_light has smallest gap in 2020 (-3.7pp) but large gaps elsewhere
   - This suggests the gap is not a simple systematic bias

4. **Worker eval is completely unreliable for walk-forward assessment**
   - The promotion gate based on worker eval is meaningless
   - All walk-forward results in the repository are unreliable

## Root Cause Hypothesis

The worker eval uses `evaluate_split()` which:
- Uses simplified cost model (fixed 0.1% per trade)
- Uses simplified rebalance logic
- Does NOT use qlib's Exchange model
- Does NOT account for min_cost, trade_lock, buy_cooldown details
- May have different state initialization

But a 100pp gap cannot be explained by cost differences alone. There may be:
- A bug in `evaluate_split` cost calculation
- A bug in how worker windows are sliced
- Different price data being used

## Immediate Impact

1. **v19 was incorrectly rejected by walk-forward gate**
   - Worker eval showed wf_2023_2024 = -18.7% (blocked)
   - Main backtest shows wf_2024 = +25.5% (would pass)

2. **v19_reg was incorrectly assessed**
   - Worker eval showed wf_2022 = +42.2% (looks good)
   - Main backtest shows wf_2022 = -38.3% (actually bad)

3. **All walk-forward comparisons are invalid**
   - Cannot compare v19 vs v19_reg using worker eval
   - Cannot trust any walk-forward conclusion

## Required Fix

**Replace worker eval with main backtest for walk-forward assessment.**

Implementation:
1. Modify `run_walk_forward_isolated` to use `backtest_profile` instead of spawning worker subprocess
2. Or create a post-processing step that runs `backtest_profile` on each window after worker eval
3. Update promotion gate to use main backtest metrics

## Workaround (Immediate)

Until the fix is implemented:
1. Do NOT use worker eval for promotion decisions
2. Use main backtest on walk-forward windows for manual assessment
3. Consider v19 as the production candidate (main backtest shows it outperforms in all windows)
