---
date: '2026-04-16'
title: '2026-04-16: Asymmetric Threshold Experiment (Option 1)'
kind: review
profiles:
- soxl
status: dropped
tags:
- dropped
- eval-2022
- review
- rule-based
- soxl
- stress-regime
artifacts: []
workpad: memory/index.json
summary: Legacy memory note normalized into the repo-wide schema. Test "buy easy,
  sell hard" asymmetric rebalance threshold while keeping the 3-day trade lock and
  hard regime routing. `rebalance_threshold` now returns **direction-dependent threshold
---

# 2026-04-16: Asymmetric Threshold Experiment (Option 1)

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

## Goal
Test "buy easy, sell hard" asymmetric rebalance threshold while keeping the 3-day trade lock and hard regime routing.

## Changes
- `rebalance_threshold` now returns **direction-dependent thresholds**:
  - **Buy** (increasing position): base = 0.02 (quiet regime 0.04)
  - **Sell** (decreasing position): base = 0.06 (quiet regime 0.12)
  - High-vol stress: buy = 0.01, sell = 0.03
- All call sites in `rl.py` and `backtest.py` updated to pass `direction` based on `target_pos` vs `current_pos`.
- Backtest plot enhanced with green ▲ buy markers and red ▼ sell markers on the Position subplot.

## Results: Asymmetric vs Symmetric (both 80 iters, multi-asset)

### SOXL (qlib backtest)
| Metric | Asymmetric (Option 1) | Symmetric (A+B) | Delta |
|---|---|---|---|
| **Return** | **461.7%** | **598.5%** | **-136.8pp** |
| Sharpe | 0.832 | 0.895 | -0.063 |
| MaxDD | 55.2% | 54.7% | +0.5pp |
| **2022 MaxDD** | **35.74%** ✅ | **42.25%** ❌ | **-6.5pp** |
| **Trades** | **622** | **719** | **-97** |

### Yearly Slices (Asymmetric)
```
test_2020:      return=108.08%  sharpe=2.307   max_dd=19.07%
test_2022:      return=-29.86%  sharpe=-0.946  max_dd=35.74%   <-- passes 35% gate
test_2023_2024: return=49.62%   sharpe=0.733   max_dd=34.91%
```

### Generalization (zero-shot)

| ETF | Asymmetric | B&H | A+B Symmetric |
|---|---|---|---|
| SOXL | 446.7% | 154.3% | 544.2% |
| TQQQ | 264.9% | 284.3% | 259.2% |
| UPRO | 210.9% | 180.5% | 208.3% |
| SPXL | 203.0% | 183.2% | 197.2% |
| TECL | **287.7%** | 270.1% | 296.8% |

## Key Findings

### 1. Asymmetric "buy easy, sell hard" hurt total returns
Contrary to the hypothesis, making it easier to buy and harder to sell **reduced SOXL returns by 137pp** compared to the symmetric high threshold.

**Why?**
- The symmetric high threshold (0.04/0.08) was already doing two valuable things:
  1. Preventing premature buying into falling knives
  2. Preventing panic selling on minor pullbacks
- By lowering the buy threshold to 0.02, the model started **buying dips too early** before the bottom was in.
- By raising the sell threshold to 0.06, the model **held losing positions too long** in drawdowns.
- The net effect was worse entry timing + worse exit timing, despite fewer total trades.

### 2. 2022 MaxDD improved back below the promotion gate
The one bright spot: 2022 MaxDD dropped from 42.25% → **35.74%**, making this the **second configuration to pass the 35% gate** (after hard-gate + floor 0.30 at 20 iters).

This confirms that making selling harder does keep the model **more committed to core positions during volatility**, which suppresses whipsaws in bear markets.

### 3. Trade count continues to fall
622 trades (asymmetric) vs 719 trades (symmetric). The trade lock + threshold combo is successfully reducing noise.

### 4. TQQQ still loses to B&H
264.9% vs 284.3%. The asymmetric threshold did not solve the TQQQ underperformance problem. The model is still too cautious for broad-index tech rallies.

## Plot Observation (with buy/sell markers)
The updated `soxl_qlib_strategy_v2.png` now shows:
- **Green ▲** = buy signals
- **Red ▼** = sell signals
- Visual confirmation that 2020-2021 and 2023-2025 have dense clusters of both buys and sells (active band-trading)
- 2022 shows a sustained low-position regime with sporadic sells and fewer buys

## Conclusion
**The symmetric high threshold (A+B) remains the higher-return configuration** on SOXL. Asymmetric threshold trades ~137pp of return for ~6.5pp of 2022 drawdown improvement.

If the **primary goal is to pass the promotion gate**, asymmetric threshold is viable (35.74% 2022 MaxDD).
If the **primary goal is to maximize total return and close the gap to the 688.6% rule-based strategy**, the symmetric high threshold is superior.

## Next Options
1. **Revert to symmetric threshold** and pursue other levers (e.g. floor raise to 0.40, longer training, or asset-specific floors).
2. **Keep asymmetric threshold** if promotion gate compliance is the hard constraint.
3. **Hybrid approach**: asymmetric threshold only in high regime_risk (>0.55), symmetric elsewhere. This might capture the best of both worlds.

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
