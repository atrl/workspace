# 2026-04-16: Hard Regime Routing + Floor 0.30

## Gate Optimization: Hard Regime Routing

### Problem
MoE ablation revealed that **Expert 2** (crisis/low-exposure expert) delivers the highest standalone return (**404.9%**), yet the natural gate only selects it **4.8%** of the time. The gate is over-reliant on the generic Expert 1 (~59% of decisions).

### Solution Implemented
Added **hard regime routing** to `RegimeMoENet`:
- **Low regime risk (< 0.35)** → force **Expert 0** (higher exposure)
- **High regime risk (> 0.65)** → force **Expert 2** (lower exposure)
- **Mid regime** → keep original learned top-1 gate

This is implemented as a runtime flag in `forward()`. When `--moe` is passed to `cli.py`, hard routing is now enabled for both training rollouts and inference.

### Fast-eval validation (existing model)
| Gate Mode | Return | Sharpe | MaxDD | Trades |
|---|---|---|---|---|
| Natural | 361.6% | 0.79 | 53.9% | 909 |
| **Hard routing** | **392.1%** | **0.81** | **54.0%** | **836** |

Hard routing alone gives **+30.5pp return** with **73 fewer trades** and virtually identical drawdown.

## Floor 0.30 Experiment

### Change
`action_band_from_obs` bullish floor lowered from **0.50** → **0.30** (healthy uptrend with `anchor_gap > 0.02` and `anchor_slope >= 0`). Baseline floor remains 0.15.

### Full Multi-Asset Training Results (seed=20260411, 20 iters)

| Metric | Hard Gate + Floor 0.30 | Multi-Asset v2 (Floor 0.50) | Multi-Asset v1 |
|---|---|---|---|
| **Return** | **354.6%** | 414.2% | 348.7% |
| **Sharpe** | **0.78** | 0.80 | 0.77 |
| **MaxDD** | **50.6%** | 56.1% | 52.4% |
| **2022 MaxDD** | **33.93%** ✅ | 40.55% ❌ | 35.33% ❌ |
| **Trades** | **1063** | 1075 | 1027 |

### Yearly Slices (Hard Gate + Floor 0.30)
```
test_2020:      return=92.11%   sharpe=2.254   max_dd=17.45%
test_2022:      return=-25.25%  sharpe=-0.813  max_dd=33.93%   <-- passes 35% gate
test_2023_2024: return=41.88%   sharpe=0.679   max_dd=34.97%
```

### Assessment
- **2022 MaxDD 33.93% is the first multi-asset configuration to pass the 35% promotion gate.**
- **Trade-off**: return dropped ~60pp vs floor 0.50, but risk control is materially better.
- Floor 0.30 is a viable middle ground: it provides more core exposure than the original 0.15 baseline, but doesn't trap the model in the 2022 crash like floor 0.50 did.

## Generalization Results (Hard Gate + Floor 0.30)

| ETF | Strategy | B&H | Delta vs B&H |
|---|---|---|---|
| SOXL | 349.85% | 154.26% | ✅ +195.59pp |
| TQQQ | 257.16% | 284.31% | ❌ -27.15pp |
| UPRO | 200.65% | 180.49% | ✅ +20.16pp |
| SPXL | 188.69% | 183.15% | ✅ +5.54pp |
| TECL | 244.36% | 270.08% | ❌ -25.72pp |

### vs Multi-Asset v2 (Floor 0.50)
- **TQQQ** dropped from 285.6% → 257.2%. Floor 0.30 + hard gate is too conservative for TQQQ's relentless bull trends.
- **UPRO/SPXL** still beat B&H, but with smaller margins.
- **SOXL** dropped from 414.2% → 354.6%.

## Conclusion

1. **Hard regime routing works.** It is the most effective single architectural fix for the MoE gate. It increases return, reduces trade count, and doesn't hurt drawdown.
2. **Floor 0.30 is viable but conservative.** It successfully brings 2022 MaxDD below the promotion gate at the cost of ~60pp total return on SOXL and losing TQQQ/TECL to B&H.
3. **The gap to the rule-based strategy (688.6%) remains.** Even with an optimized gate, the RL model still makes **1,000+ trades** vs the rule-based **45 trades**. Friction is the dominant headwind.

## Remaining Optimization Levers
If the goal is to close the 334pp gap to the rule-based strategy on SOXL, the highest-impact moves are:
1. **Raise rebalance threshold drastically** (0.015 → 0.05+) and/or add a 3-5 day trade-lock to cut trade count by 50-70%.
2. **Coarsen action space** from 11 levels to 5 levels, preventing micro-adjustments.
3. **Train much longer.** 20 iters for 67 pooled episodes is only ~3 epochs. The model hasn't converged.
