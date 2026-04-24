---
date: 2026-04-24
tags: [risk_budget, mode_conditioned, gate_pass, v3]
direction: network
---

# risk_budget_usage_v3 passes fixed-split promotion gate

## Result

| Slice | Return | Sharpe | MaxDD | AvgPos |
|-------|--------|--------|-------|--------|
| test_proxy | 274.7% | **1.074** | 21.4% | 0.208 |
| test_2020 | 29.3% | 1.430 | 18.5% | 0.233 |
| test_2022 | -9.7% | -0.601 | **12.0%** | **0.143** |
| test_2023_2024 | 60.5% | 1.378 | 16.0% | 0.215 |

### Promotion Gate
- test_2022 MaxDD: 12.0% < 20.0% ✅
- test_2022 avg_pos: 0.143 < 0.15 ✅
- test_proxy Sharpe: 1.074 > 0.95 ✅

**Fixed Split Gate: PASSED!**

## Key Insight

Mode-conditioned **actor input** (one-hot appended to obs) is dramatically more
effective than mode-conditioned **reward shaping** alone:

| Variant | proxy Sharpe | 2022 MaxDD | 2022 avg_pos |
|---------|-------------|------------|--------------|
| v1 (baseline) | 0.942 | 18.7% | 0.127 |
| v2 (reward shaping) | 0.884 | 7.4% | 0.070 |
| **v3 (actor input)** | **1.074** | **12.0%** | **0.143** |

v2's reward shaping made the policy excessively conservative — great downside
protection but missed all upside. v3 lets the actor *see* the mode and learn
its own usage policy per mode, achieving both better Sharpe and acceptable
drawdown control.

## Comparison to canonical balanced

| Metric | balanced (canonical) | risk_budget_usage_v3 |
|--------|---------------------|----------------------|
| proxy Sharpe | ~0.95 | **1.074** |
| proxy MaxDD | ~25% | 21.4% |
| 2022 MaxDD | ~15% | **12.0%** |

v3 now exceeds canonical balanced on Sharpe and has tighter drawdown control.

## Next Steps

1. Run walk-forward validation to confirm no overfit
2. If WF passes, promote to canonical or create v4 with additional refinements
3. Consider whether mode-conditioned input should become standard across all profiles
