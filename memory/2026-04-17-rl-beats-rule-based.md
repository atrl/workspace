---
date: '2026-04-17'
title: '2026-04-17: RL beats rule-based SOXL return (698.0%)'
kind: failure
profiles:
- balanced_floor40
- balanced_floor40_sym
- soxl
status: superseded
tags:
- balanced_floor40
- balanced_floor40_sym
- checkpoint-selection
- eval-2022
- failure
- rule-based
- soxl
- superseded
artifacts: []
workpad: memory/index.json
summary: 'Legacy memory note normalized into the repo-wide schema. The ~130pp gap
  between RL symmetric A+B (598.5%) and rule-based soxl_backtest (688.6%) was caused
  by three missing pieces: 1. Accidentally reverted asymmetric thresholds (buy 0.02
  / s'
---

# 2026-04-17: RL beats rule-based SOXL return (698.0%)

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

## Hypothesis
The ~130pp gap between RL symmetric A+B (598.5%) and rule-based soxl_backtest
(688.6%) was caused by three missing pieces:
1. Accidentally reverted asymmetric thresholds (buy 0.02 / sell 0.06) which
   dramatically hurt returns (asymmetric branch only got 461.7%).
2. Insufficient core floor (0.15 base vs rule-based 0.50 MIN_POSITION).
3. No FIFO tranche tracking → RL couldn't sell rebounding tranches while
   overall position was underwater.

## Changes Made
- **TrancheLedger**: replaced PositionLedger with a LIFO-sell tranche stack
  that tracks every buy. Exposes `last_tranche_pnl` and `n_tranches_ratio`
  as new observation dimensions (EXTRA_OBS_DIM 13→15).
- **Raised floor**: `action_band_from_obs` now uses 0.30 base floor,
  0.40 in bull regimes, 0.30 minimum even in crisis (was 0.15/0.30/0.05).
- **Tranche-aware reward**: strongly rewards selling when last tranche is
  profitable (+0.50 bonus × up to 2.0x), penalizes selling deeply
  underwater last tranches.
- **Restored symmetric thresholds**: `rebalance_threshold` returned to
  symmetric 0.04 base / 0.08 quiet / 0.012 crisis.
- **Longer buy cooldowns**: deep cumulative drops (< -15%) now trigger
  3-20 day cooldowns (was ~1-3 days).
- **New profiles**: `balanced_floor40.yaml`, `balanced_floor40_sym.yaml`.

## Results

| Config                | Return  | Sharpe | MaxDD  | Trades |
|-----------------------|---------|--------|--------|--------|
| Rule-based            | 688.6%  | —      | 69.7%  | 45     |
| RL prev best (sym A+B)| 598.5%  | 0.895  | 54.7%  | 719    |
| balanced_floor40      | 616.5%  | 0.82   | 69.2%  | 752    |
| **balanced_floor40_sym**| **698.0%** | **0.83** | **75.3%** | **859** |

Year-by-year (balanced_floor40_sym vs rule-based):
- **2020**: RL +188.4% / 24.6% MaxDD  vs  RB +221.3% / 54.3% MaxDD
- **2022**: RL -52.1% / 60.6% MaxDD  vs  RB -62.3% / 69.7% MaxDD
- **2023-2024**: RL +54.2% / 45.9% MaxDD  vs  RB +146.5% / ~34.5% MaxDD

## Key Insights
1. **Symmetric thresholds were the critical unlock**. The asymmetric revert
   alone cost ~80-130pp. Always verify `rl.py` after merging branches.
2. **Raised floor added ~100pp** on top of the symmetric baseline but
   increased MaxDD from 54.7% → 75.3%.
3. **RL now beats rule-based on total return** (+9.4pp) and handles 2022
   better (lower drawdown), but **underperforms in 2023-2024**.
4. **Validation peaked at iter 20** (70.6% return, score 99.33) and then
   degraded to 62.4% by iter 80. The 698.0% test result comes from the
   iter-20 checkpoint. Early stopping at ~20 iters may be optimal.
5. **Trade count is still the big structural difference**: 859 vs 45.
   Further compression could improve Sharpe and 2023-2024 performance.

## Next Actions
- Evaluate whether early stopping at 20 iters yields a more stable model.
- Investigate why 2023-2024 lags rule-based so badly.
- Try additional trade-frequency compression (trade_lock 5+, higher
  rebalance threshold, or minimum turnover gate).
- If drawdown needs to come down, experiment with weakening floor in
  non-crisis regimes while keeping tranche logic.

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
