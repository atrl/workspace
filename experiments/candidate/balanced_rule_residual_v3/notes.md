---
date: '2026-04-20'
title: 'balanced_rule_residual_v3: floor raise + trade compression + regime-alpha'
kind: experiment
status: candidate
parent: balanced_rule_residual_v2
baseline: balanced_rule_residual_v2
---

# balanced_rule_residual_v3

## Hypothesis
v2 (521.3%) still trails rule-based (688.6%) by 167pp. The gap analysis identified
five root causes. v3 attacks the top three simultaneously:

1. **Floor too low** (20% vs rule-based 50%) → raise to 40%
2. **Too many trades** (316 vs 45) → compress via threshold + lock + penalty
3. **No regime-conditioned alpha** → scale alpha reward by regime

## Delta from v2

| Layer | v2 | v3 | Rationale |
|-------|-----|-----|-----------|
| `hard_floor` | 0.20 | **0.40** | Match rule-based core position |
| bull floor | 0.40 | **0.50** | Capture 2023-2024 bull run |
| stress floor | 0.25 | **0.30** | Don't over-de-risk |
| crisis floor | 0.15 | **0.25** | Maintain minimum exposure |
| `base_threshold` | 0.04 | **0.06** | Fewer micro-rebalances |
| `trade_lock_days` | 3 | **5** | Longer hold periods |
| `micro_rebalance_penalty` | 0.35 | **0.50** | Punish small trades harder |
| `micro_rebalance_threshold` | 0.05 | **0.08** | Wider no-penalty zone |
| `min_cap` | 0.40 | **0.50** | Drawdown cap floor aligned |
| `bull_min_cap` | 0.65 | **0.70** | Higher bull cap floor |
| `bear_min_cap` | 0.30 | **0.35** | Higher bear floor |
| alpha_scale (bull) | 3.0 | **3.9** (1.3×) | Reward outperformance more |
| alpha_scale (stress) | 3.0 | **2.1** (0.7×) | Reduce risk-seeking |

## Expected Impact

| Metric | v2 | v3 (target) |
|--------|-----|-------------|
| Return | 521.3% | 600-650% |
| Sharpe | 0.86 | 0.85-0.90 |
| MaxDD | 50.4% | 55-60% |
| Trades | 316 | 150-220 |

## Results

| Metric | v2 | v3 (target) | v3 (actual) | Δ |
|--------|-----|-------------|-------------|---|
| Return | 521.3% | 600-650% | **569.7%** | +48.4pp |
| Sharpe | 0.86 | 0.85-0.90 | **0.86** | = |
| MaxDD | 50.4% | 55-60% | **53.6%** | +3.2pp |
| Trades | 316 | 150-220 | **314** | -2 |
| Valid best | 104.61 @40 | — | **101.07 @60** | -3.54 |

### Sub-period breakdown
| Period | Return | Sharpe | MaxDD |
|--------|--------|--------|-------|
| 2020 | +99.68% | 2.289 | 19.27% |
| 2022 | -23.82% | -0.885 | 32.27% |
| 2023-2024 | +53.87% | 0.780 | 33.14% |

### Assessment
- **Return**: Modest improvement (+48pp) but fell short of 600-650% target.
- **Trade compression**: COMPLETE FAILURE. 314 trades vs target 150-220. The threshold/lock/penalty changes had **zero effect**. This is the #1 issue to diagnose.
- **Valid score**: 101.07 @60 is **worse** than v2's 104.61 @40. Promotion blocked.
- **MaxDD**: 53.6% is within tolerance but slightly worse than v2.
- **2022 performance**: Still poor (-23.8%), similar to v2. Floor raise did not help bear market.

## Root Cause Analysis (Post-Hoc)

1. **Trade compression failed**: The `base_threshold`, `trade_lock_days`, `micro_rebalance_penalty`, and `micro_rebalance_threshold` parameters were changed but trades only dropped from 316→314. This suggests either:
   - The execution layer is not respecting these parameters
   - The RL agent is learning to trade at the boundary exactly to avoid penalties
   - The trade count metric counts something different than expected

2. **Valid score regression**: Best valid @60 (101.07) < v2 @40 (104.61). The regime-alpha conditioning may have introduced instability in later stages.

## Risk
- Higher floor → higher MaxDD (could breach 60%)
- Fewer trades → slower adaptation to regime shifts
- If 2022 MaxDD exceeds 35%, promotion gate may block

## Next Action
**Diagnose trade compression failure** before attempting v4. The 167pp gap to rule-based (688.6%) is still dominated by excessive trading (314 vs 45).
