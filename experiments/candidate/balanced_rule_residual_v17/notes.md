# balanced_rule_residual_v17 — Extended Temporal Context (10 days)

## Hypothesis

v16 proved window=5 is a big win over window=3 (+140pp return, better Sharpe).
Does more context continue to help, or is there a diminishing return?

Testing window=10 to find the optimal context length.

## Changes from v16

- `observation_window: 10` (was 5)
- All other config identical to v16

## Expected outcomes

- If v17 > v16: more context is always better (up to 10 days)
- If v17 ≈ v16: 5 days is the sweet spot
- If v17 worse: too much context causes overfitting or confusion
