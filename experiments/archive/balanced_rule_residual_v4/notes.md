---
date: '2026-04-20'
title: 'balanced_rule_residual_v4: flat trade penalty + longer lock + reduced alpha'
kind: experiment
status: candidate
parent: balanced_rule_residual_v3
baseline: balanced_rule_residual_v2
---

# balanced_rule_residual_v4

## Hypothesis
v3 improved return (+48pp to 569.7%) but **trade compression completely failed** (314 trades, target 150-220). Root cause analysis identified:

1. **Alpha reward >> trade penalties** — agent is strongly incentivized to chase small alphas
2. **Base position drift defeats threshold filtering** — hybrid overlay base shifts with regime changes
3. **trade_lock=5 is too short** — average trade interval is already ~5 days
4. **No flat cost per trade** — all penalties are proportional to turnover

v4 attacks these directly:

1. **Flat trade penalty (0.75)** — fixed cost per trade regardless of size, mimicking real commissions/slippage
2. **Longer trade lock (10 days)** — forces minimum 2-week hold periods
3. **Reduced alpha_scale (2.0)** — makes small alphas not worth the trade cost

## Delta from v3

| Parameter | v3 | v4 | Rationale |
|-----------|-----|-----|-----------|
| `flat_trade_penalty` | 0.0 | **1.20** | Fixed cost per trade |
| `trade_lock_days` | 5 | **8** | Longer hold period |
| `alpha_scale` | 3.0 | **2.0** | Less eager to chase small alphas |

All other parameters identical to v3.

## Expected Impact

| Metric | v2 | v3 | v4 (target) | v4 (actual) |
|--------|-----|-----|-------------|-------------|
| Return | 521.3% | 569.7% | 580-620% | **730.8%** |
| Sharpe | 0.86 | 0.86 | 0.85-0.90 | **0.90** |
| MaxDD | 50.4% | 53.6% | 55-60% | **55.7%** |
| Trades | 316 | 314 | **120-180** | **327** ❌ |
| Valid best | 104.61 | 101.07 | 100-105 | **106.03** |

## Actual Results (2026-04-20)

- **Best checkpoint**: @80 (valid score 106.03)
- **Test**: Return 730.8% | Sharpe 0.90 | MaxDD 55.7% | Trades 327
- **test_2020**: 116.70% / Sharpe 2.45 / MaxDD 18.39%
- **test_2022**: -25.02% / Sharpe -0.91 / MaxDD 33.24%
- **test_2023_2024**: 69.00% / Sharpe 0.89 / MaxDD 31.17%
- **Promotion**: BLOCKED (trades 327 >> target 150-220)

## Key Finding: Trade Compression Failed Again

Trades **increased** from 314 (v3) to 327 (v4) despite:
- Flat penalty of 1.20 per trade
- Longer lock (5→8 days)
- Reduced alpha_scale (3.0→2.0)

**Root cause hypothesis**: The agent found significantly more alpha (730.8% vs 569.7%), making the 1.20 penalty per trade economically trivial. The strategy is highly profitable — it can afford the penalty. The penalty is a cost of doing business, not a deterrent.

## Risk
- Flat penalty may cause agent to become too passive, missing legitimate opportunities
- Longer lock may miss rapid regime shifts (e.g., March 2020 crash)
- Lower alpha_scale may reduce return if agent stops taking profitable trades
- If trades drop below 100, strategy may be too sluggish

## Promotion Criteria
- Trades 120-180 AND return >= v3 (569.7%)
- Valid score >= 100
- 2022 MaxDD <= 35%
