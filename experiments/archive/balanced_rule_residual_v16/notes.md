# balanced_rule_residual_v16 — More Temporal Context

## Hypothesis

v12/v13 proved GRU + terminal penalty (2.0-4.0) controls drawdown (48.8% MaxDD).
But return dropped to 492.3% — lower than MLP baseline (521%).

The GRU may be under-utilized with only 3 days of history.
More temporal context (5 days) could help the GRU:
- Better anticipate regime transitions
- Smoother position adjustments
- Potentially recover some return without losing drawdown control

## Changes from v13

- `observation_window: 5` (was 3)
- `episode_drawdown_penalty: 2.0` (proven effective strength)
- All other config identical

## Comparison baselines

| Profile | Window | Penalty | Return | Sharpe | MaxDD |
|---------|--------|---------|--------|--------|-------|
| v12 | 3 | 4.0 | 492.3% | 0.85 | 48.8% |
| v13 | 3 | 2.0 | 492.3% | 0.85 | 48.8% |
| **v16** | **5** | **2.0** | **?** | **?** | **?** |

## Expected outcomes

- If v16 > v12/v13 on return with similar MaxDD: more context helps
- If v16 ≈ v12/v13: 3-day window is already sufficient
- If v16 worse: too much context causes overfitting or confusion
