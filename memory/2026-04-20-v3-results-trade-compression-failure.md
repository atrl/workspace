---
date: '2026-04-20'
title: "v3 Results: Trade Compression Failed \u2014 Root Cause Analysis"
kind: failure
profiles: []
status: dropped
tags:
- checkpoint-selection
- dropped
- eval-2022
- execution-overlay
- failure
artifacts: []
workpad: memory/index.json
summary: "Legacy memory note normalized into the repo-wide schema. | Metric | v2 |\
  \ v3 Target | v3 Actual | \u0394 | |--------|-----|-----------|-----------|---|\
  \ | Return | 521.3% | 600-650% | **569.7%** | +48.4pp | | Sharpe | 0.86 | 0.85-0.90\
  \ | **0.86** |"
---

# v3 Results and Trade Compression Failure Analysis

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

## v3 Actual Results

| Metric | v2 | v3 Target | v3 Actual | Δ |
|--------|-----|-----------|-----------|---|
| Return | 521.3% | 600-650% | **569.7%** | +48.4pp |
| Sharpe | 0.86 | 0.85-0.90 | **0.86** | = |
| MaxDD | 50.4% | 55-60% | **53.6%** | +3.2pp |
| Trades | 316 | 150-220 | **314** | -2 |
| Valid best | 104.61 @40 | — | **101.07 @60** | -3.54 |

Promotion: **BLOCKED** (valid score regression + trade target missed)

### Sub-periods
| Period | Return | Sharpe | MaxDD |
|--------|--------|--------|-------|
| 2020 | +99.68% | 2.289 | 19.27% |
| 2022 | -23.82% | -0.885 | 32.27% |
| 2023-2024 | +53.87% | 0.780 | 33.14% |

## Why Trade Compression Failed (314 → 314)

### Finding 1: `base_threshold` < Minimum Discrete Step
- Action space with hybrid overlay: `delta_levels = [-0.10, -0.05, 0.0, 0.05, 0.10]`
- Minimum non-zero step = 0.05
- v3 `base_threshold = 0.06`
- **A 0.05 delta is below threshold and should be filtered.** BUT:

### Finding 2: Base Position Drift Defeats Threshold
- Hybrid overlay: `target_pos = rule_based_position(obs) + delta`
- `rule_based_position = (floor + cap) / 2.0` — changes daily as floor/cap shift
- Example: base drifts 0.60 → 0.62 (+0.02), agent adds delta=+0.05 → total change = 0.07 > 0.06 threshold → **trade happens despite small delta**
- **Conclusion: threshold filters individual delta, not total position drift**

### Finding 3: Reward Incentive >> Penalty
- Kelly-log alpha reward at alpha=1% in bull: `log(1.01) * 100 * 3.9 ≈ 3.88`
- Short-hold penalty for 0.10 turnover at age=0: `(1-0) * 0.10 * 2.25 = 0.225`
- Turnover penalty: `0.10 * 0.05 = 0.005`
- **Alpha reward (3.88) >> total penalty (~0.23)** — agent is strongly incentivized to trade

### Finding 4: `micro_rebalance_penalty` Never Triggered
- `micro_rebalance_threshold = 0.08`, but with hybrid overlay the agent can use 0.10 deltas to avoid it
- Agent learned to avoid the 0.05 level (penalized + filtered) but uses 0.10 freely

### Finding 5: `trade_lock_days = 5` Insufficient
- 314 trades over 1569 days = average 5.0 day interval
- Lock of 5 days barely overlaps with natural trade frequency
- Most trades happen just after lock expires

## What Actually Worked in v3

1. **Return +48pp**: Floor raise (0.20→0.40) and regime-alpha conditioning did improve returns
2. **MaxDD 53.6%**: Within tolerance, slightly worse than v2 but acceptable
3. **Valid @60 score 101.07**: Close to v2's 104.61, not a major regression

## What Did NOT Work

1. **Trade compression**: Complete failure. All mechanisms (threshold, lock, penalty) were insufficient
2. **Valid score**: Best checkpoint (@60) is worse than v2 (@40)
3. **2022 performance**: Still -23.8%, similar to v2. Floor raise didn't help bear market

## v4 Design Directions

### Option A: Flat Trade Penalty (Recommended)
Add a fixed cost per trade in reward.py, regardless of turnover size:
```python
if turnover > 0:
    reward -= cfg.flat_trade_penalty  # e.g., 0.5-1.0
```
This is realistic (commissions, slippage, bid-ask) and directly penalizes frequent trading.

### Option B: Dramatically Longer trade_lock
`trade_lock_days: 15` or even `20`. Forces minimum 2-3 week hold periods.
Risk: Misses regime shifts.

### Option C: Coarser Action Space
`delta_levels: [-0.15, 0.0, 0.15]` — fewer opportunities to trade, larger moves only.
Risk: Reduced fine-tuning capability.

### Option D: Reduce alpha_scale
Lower `alpha_scale` from 3.0 to 1.5-2.0 so small alphas aren't worth chasing.
Risk: May reduce overall return.

## Recommended v4 Plan
Combine **Option A + B + D**:
1. Add `flat_trade_penalty: 0.75` to reward.py
2. Increase `trade_lock_days: 10` (double from v3)
3. Reduce `alpha_scale: 2.0` (from 3.0)
4. Keep floor raise and regime-alpha from v3
5. Target: trades 120-180, return 580-620%, Sharpe 0.85+

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
