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

_(Trimmed by prune_experiments.py)_
