# 2026-04-16: A+B Breakthrough – High Threshold + Trade Lock + 80 Iters

## Hypothesis
The RL model was losing to the rule-based strategy primarily due to **friction from excessive micro-trades** (~1000+ vs 45). Two levers were identified:
- **A**: Drastically raise `rebalance_threshold` and add a `trade_lock` period to cut trade frequency.
- **B**: Increase training depth from 20 → 80 iters so the model converges under the new friction regime.

## Changes

### A. Rebalance threshold raised + Trade lock added
- `rebalance_threshold` baseline raised from `0.015` → `0.04` (quiet regime `0.03` → `0.08`).
- Added a **3-day trade lock** in `SoxlSimulator`: after any material trade (`turnover > 0.02`), the position is frozen for 3 days regardless of the actor's new action.
- Existing buy cooldown and hard regime routing remain active.

### B. Training depth increased
- `configs/soxl/balanced.yaml`: `max_iters` raised from `20` → `80`.

## Results: Multi-asset MoE + Hard Gate + Floor 0.30 + A+B

### SOXL (qlib backtest)
| Metric | A+B (80 iters) | Previous Best (v2, 20 iters) | Hard Gate Floor 0.30 (20 iters) |
|---|---|---|---|
| **Return** | **598.5%** | 414.2% | 354.6% |
| **Sharpe** | **0.895** | 0.80 | 0.78 |
| **MaxDD** | **54.7%** | 56.1% | 50.6% |
| **2022 MaxDD** | **42.25%** | 40.55% | 33.93% |
| **Trades** | **719** | 1075 | 1063 |

### Yearly Slices (qlib)
```
test_2020:      return=107.48%  sharpe=2.307   max_dd=20.06%
test_2022:      return=-38.34%  sharpe=-1.184  max_dd=42.25%
test_2023_2024: return=60.94%   sharpe=0.810   max_dd=34.04%
```

### Generalization (zero-shot on other 3x ETFs)

| ETF | Strategy (A+B) | Buy & Hold | Delta |
|---|---|---|---|
| SOXL | 544.2% | 154.3% | ✅ +389.9pp |
| TQQQ | 259.2% | 284.3% | ❌ -25.1pp |
| UPRO | 208.3% | 180.5% | ✅ +27.8pp |
| SPXL | 197.2% | 183.2% | ✅ +14.0pp |
| **TECL** | **296.8%** | **270.1%** | ✅ **+26.7pp** |

## Key Breakthroughs

1. **SOXL return jumped from 354.6% → 598.5%** (+244pp) simply by cutting friction and training longer.
2. **TECL beats B&H for the first time** (296.8% vs 270.1%). The model is now competitive on concentrated tech LETFs.
3. **Trade count dropped from 1063 → 719** (qlib) and **654** (generalization), confirming that the high-threshold + trade-lock mechanism successfully suppressed noise trading.
4. **Sharpe hit 0.895**, the highest ever recorded in this repo for a multi-asset SOXL policy.

## Trade-offs
- **2022 MaxDD worsened from 33.93% → 42.25%**, pushing the model back above the 35% promotion gate.
- The rule-based strategy still holds a 90pp advantage on SOXL (688.6% vs 598.5%), but the gap has closed from 274pp to **90pp**.
- **TQQQ still trails B&H by ~25pp**. The model is too cautious for the relentless Nasdaq-100 uptrend.

## Root-cause confirmation
The massive return jump (+244pp) from A+B validates the earlier diagnosis:
> **Friction massacre was the dominant headwind.** Cutting trade count and letting the model learn under higher threshold constraints unlocks far more alpha than architectural tweaks alone.

## Remaining gap to rule-based (90pp)
The rule-based strategy still out-earns the RL model on SOXL. The likely remaining reasons:
1. **Core position floor**: Rule-based keeps 50% minimum. RL floor is 0.15 (0.30 in bull trends). This guarantees the rule-based strategy captures more of the steepest rallies.
2. **Band granularity**: Rule-based uses clear 10%/30% steps. RL uses 11 discrete actions mapped through a floating band, which still produces some micro-wiggles even with trade lock.

## Next Options
1. **Raise the core floor to 0.40** while keeping trade lock. This will likely push returns toward 650-700% but also increase 2022 MaxDD toward 50%.
2. **Asset-specific floor**: SOXL/TECL get floor 0.40, TQQQ/UPRO/SPXL keep floor 0.15 or 0.30.
3. **Train even longer** (e.g. 200 iters). The loss curves in the log show the model is still improving slowly at iter 80.

## Artifacts
- Model: `artifacts/models/soxl_balanced_moe_actor.pt`
- Log: `log/balanced_multiasset_AB_20260416_225734.log`
- Generalization: `runtime/generalization/results.json`
