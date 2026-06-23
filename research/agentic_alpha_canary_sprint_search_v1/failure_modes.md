# Failure Modes

- canary_sprint_short_hold: short holding and VIX filters found high-return tails, but high-return settings still crossed crash-level drawdowns; drawdown-gated settings scaled down below the production baseline.
- dip_sprint_short_hold: dip entries produced the best drawdown-gated sprint return, but the return remained far below the production strategy.
- guarded_sprint_budget: stateful drawdown and volatility budget controls reduced risk, but the remaining exposure budget was too small to beat the production baseline.
