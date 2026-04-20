# balanced_rule_residual_v15 — Weak Terminal Penalty Sweep

## Hypothesis

v12/v13 showed penalty=2.0 and 4.0 produce identical results (492.3%/48.8%).
v14 showed penalty=0.0 reverts to aggressive behavior (663.2%/56.6%).
The penalty strength might have a non-monotonic effect: below some threshold
the agent ignores it, above some threshold it fully engages.

Testing penalty=1.0 to see if there is a sweet spot between:
- v14 (no penalty): too aggressive, MaxDD 56.6%
- v12/v13 (penalty 2.0-4.0): too conservative, return 492.3%

## Changes from v14

- `episode_drawdown_penalty: 1.0` (was 0.0)
- All other config identical to v14

## Comparison baselines

| Profile | Penalty | Return | Sharpe | MaxDD |
|---------|---------|--------|--------|-------|
| v14 | 0.0 | 663.2% | 0.89 | 56.6% |
| v15 | 1.0 | ? | ? | ? |
| v12 | 4.0 | 492.3% | 0.85 | 48.8% |
| v13 | 2.0 | 492.3% | 0.85 | 48.8% |

## Expected outcomes

- If v15 ≈ v12/v13: penalty has a sharp threshold (anything ≥2.0 is "on")
- If v15 is between v14 and v12: penalty strength is a continuous dial
- If v15 ≈ v14: penalty needs to be >1.0 to have effect
