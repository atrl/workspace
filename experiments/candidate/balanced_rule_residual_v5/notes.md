---
date: '2026-04-20'
title: 'balanced_rule_residual_v5: massively stronger flat trade penalty + longer lock'
kind: experiment
status: candidate
parent: balanced_rule_residual_v4
baseline: balanced_rule_residual_v4
---

# balanced_rule_residual_v5

## Hypothesis

v4 improved return dramatically (+161pp to 730.8%) but **trade compression failed for a second time** — trades increased from 314 to 327. Root cause: the 0.75 flat penalty per trade is economically trivial relative to 730% returns. The penalty is a cost of doing business, not a deterrent.

v5 makes the penalty genuinely hurt:

1. **Flat trade penalty 3.00** — 4x increase from v4's 0.75. At ~50 trades per episode, this is -150 penalty per episode, enough to make marginal trades unprofitable.
2. **Trade lock 12 days** — 20% longer than v4's 10, forcing minimum ~2.5 week hold periods.

## Delta from v4

| Parameter | v4 | v5 | Rationale |
|-----------|-----|-----|-----------|
| `flat_trade_penalty` | 0.75 | **3.00** | 4x stronger — must hurt to deter |
| `trade_lock_days` | 10 | **12** | Even longer minimum holds |

All other parameters identical to v4.

## Expected Impact

| Metric | v3 | v4 | v5 (target) |
|--------|-----|-----|-------------|
| Return | 569.7% | 730.8% | 550-700% |
| Sharpe | 0.86 | 0.90 | 0.85-0.90 |
| MaxDD | 53.6% | 55.7% | 55-60% |
| Trades | 314 | 327 | **150-220** |
| Valid best | 101.07 | 106.03 | 95-110 |

## Risk

- Penalty may be too strong — agent could stop trading entirely (< 50 trades)
- Return could collapse if agent becomes pathologically passive
- Long lock may miss rapid regime shifts
- If trades < 100, penalty is too high and should be reduced to 2.0 or 1.5

## Promotion Criteria

- Trades 150-220 AND return >= v3 (569.7%)
- Valid score >= 95
- 2022 MaxDD <= 35%
