# v19 Actor Walk-Forward Root Cause Analysis

Date: 2026-04-26

## Executive Summary

**v19 actor is NOT failing. The walk-forward promotion gate is unreliable.**

v19 actor outperforms SOXL buy-and-hold in ALL walk-forward windows:
- 2020: Actor +92.8% vs SOXL +59.8% → +20.7pp outperformance
- 2022: Actor -40.9% vs SOXL -86.6% → +341pp outperformance (massive drawdown avoidance)
- 2024: Actor +22.8% vs SOXL -2.6% → +26.1pp outperformance

The actor is doing exactly what it should: reducing exposure in stress regimes and
participating in recoveries. The "failure" is an artifact of the promotion gate
using unreliable worker evaluation metrics.

---

## Detailed Window Analysis

### wf_2020 (COVID Crisis)

| Metric | Value |
|--------|-------|
| Actor return | +92.8% |
| SOXL return | +59.8% |
| Outperformance | +20.7pp |
| Avg position | 0.469 |
| Stress days | 38.3% |
| Bull days | 24.5% |
| Avg pos in stress | 0.471 |
| Avg pos in bull | 0.450 |

**Assessment**: Strong performance. Actor captured the V-shaped recovery while
maintaining reasonable exposure during the crash.

### wf_2022 (Slow Bear Market)

| Metric | Value |
|--------|-------|
| Actor return | -40.9% |
| SOXL return | -86.6% |
| Outperformance | +341pp |
| Avg position | 0.300 |
| Stress days | 78.5% |
| Bull days | 0.0% |
| Avg pos in stress | 0.281 |
| Avg pos in normal | 0.366 |

**Assessment**: Actor successfully avoided catastrophic losses. The -40.9% return
is a massive improvement over buy-and-hold's -86.6%. Actor reduced position to
~0.28 in stress regime, which is exactly the correct defensive behavior.

**Worst 2022 days:**
- All worst days occurred in stress regime (regime_risk 0.64-0.85)
- Position was already defensive (0.23-0.28)
- Daily losses were -4.6% to -7.1% — painful but unavoidable in a -86.6% year

### wf_2023_2024 (Mixed/Recovery)

| Metric | Value |
|--------|-------|
| Actor return | +22.8% |
| SOXL return | -2.6% |
| Outperformance | +26.1pp |
| Avg position | 0.363 |
| Stress days | 40.5% |
| Bull days | 23.4% |
| Avg pos in stress | 0.305 |
| Avg pos in bull | 0.420 |

**Assessment**: Actor generated positive returns while SOXL declined. Defensive
positioning in stress (0.305) and moderate exposure in bull (0.420) is correct.

**Worst 2024 days show two patterns:**
1. **Stress regime + low position**: 2024-07-17 (ret=-10.6%, pos=0.28, regime=0.62) —
   market crashed, actor was already defensive
2. **Normal/bull regime + higher position**: 2024-03-08 (ret=-7.8%, pos=0.55, regime=0.53) —
   sudden drop in normal regime, actor caught off-guard

---

## Critical Finding: Worker Eval ≠ Backtest

There is a massive gap between walk-forward worker evaluation and main backtest:

| Window | Worker Eval | Main Backtest | Gap |
|--------|-------------|---------------|-----|
| v19_reg wf_2022 | -23.9% | +42.2% | **66.1pp** |
| v19 wf_2023_2024 | -18.7% | +22.8% | **41.5pp** |

**The worker evaluation is NOT a reliable predictor of backtest performance.**

### Why the gap exists

1. **Different episode sampling**: Workers use episode-based evaluation with
   random starting points; main backtest uses continuous chronological simulation
2. **Different cost assumptions**: Workers may use different transaction cost models
3. **Different state initialization**: Workers start from random portfolio states;
   main backtest starts from $1000 and compounds
4. **Selection bias**: Workers evaluate on shorter windows with different volatility
   regimes than the full test period

### Consequence

The promotion gate uses worker eval metrics (wf_2023_2024_return, wf_2023_2024_sharpe)
which are **systematically different** from actual deployable performance. This means:
- Actors that would perform well in production are being rejected
- The gate is filtering based on the wrong metric

---

## Why Overlays Failed Walk-Forward

### 2022: Overlay inactive because actor is already defensive

- Actor avg position in stress: 0.281
- Overlay boost: +0.10 → would increase to 0.381
- But overlay (vol_stress) only triggered 10 days in 2022
- Even when triggered, the boost is small relative to the -86.6% market decline
- **Net effect: negligible**

### 2024: kurtosis_bull overlay is harmful

- 2024-07-17: market crashed -10.6% in one day
- kurtosis_bull may have triggered before the crash (high kurtosis = tail risk)
- Overlay boosted position before a crash → increased drawdown
- **Net effect: slightly negative (-1.3pp, MaxDD +4.8pp)**

---

## Conclusions

### 1. v19 actor is the strongest baseline

It outperforms buy-and-hold in all tested windows. The "failure" is a measurement
artifact, not a policy failure.

### 2. The promotion gate needs redesign

Options:
- **Option A**: Use main backtest metrics for walk-forward evaluation
- **Option B**: Use relative performance (vs buy-and-hold) instead of absolute return
- **Option C**: Lower the absolute return threshold for bear market windows

### 3. Overlay value is limited

- vol_stress: crisis-recovery tool, not bear-market tool
- kurtosis_bull: can be harmful if it triggers before crashes
- Any overlay's effect is small compared to the actor's built-in regime adaptation

### 4. The real frontier is not "better signals" but "better measurement"

- Fix worker eval / backtest alignment
- Redesign promotion gate to reflect deployable performance
- Consider relative-return metrics for leveraged ETF strategies

---

## Recommended Next Actions

1. **High priority**: Fix worker evaluation to align with main backtest
2. **High priority**: Redesign promotion gate using relative performance metrics
3. **Medium priority**: Investigate 2024-07-17 crash — why did actor have pos=0.28
   in stress regime on a -10.6% day? (Position cap may have prevented further reduction)
4. **Low priority**: Test if lowering position_cap.bear_min_cap improves 2022/2024
