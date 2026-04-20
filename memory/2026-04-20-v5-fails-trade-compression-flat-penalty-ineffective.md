# v5 Results: Flat Penalty 4x Stronger — Trade Compression Still Fails

Date: 2026-04-20

## Results
| Metric | v4 | v5 (4x penalty) | Delta |
|--------|-----|-----------------|-------|
| Return | 730.8% | 780.7% | +50pp |
| Sharpe | 0.90 | 0.93 | +0.03 |
| MaxDD | 55.7% | 55.9% | +0.2pp |
| **Trades** | **327** | **326** | **-1** |

## Critical Finding
A 4x increase in flat_trade_penalty (0.75→3.0) combined with longer trade_lock (10→12) had **virtually zero effect on trade count**. Rewards were ~30% more negative (~-400 vs ~-300), confirming the penalty IS applied, but behavior did not change.

## Root Cause Analysis
1. **Flat penalty is a soft cost**: The value function absorbs it as "cost of doing business." Alpha rewards (alpha_scale=2.0, Kelly log scaling, return_scale=1.15) are so large that a 3.0 per-trade penalty is irrelevant.
2. **Trade count is structurally driven**: Trade count has been invariant (314→327→326) across penalties of 0.5→0.75→3.0. This means trades are determined by execution thresholds and market dynamics, not by penalty strength.
3. **Inactivity penalty is counterproductive**: `inactivity_penalty: 0.80` actively PUNISHES the agent for holding positions >4 days, creating pressure to trade that counteracts trade compression efforts.
4. **Rule-based benchmark trades 45×**: The RL agent makes 7× more trades than the rule it tracks, suggesting excessive micro-rebalancing.

## Key Insight
**Execution thresholds are the real lever.** The `base_threshold: 0.06` means any position change ≥6% triggers a trade. The agent makes many small oscillating adjustments around the rule position, each counting as a trade. Raising the threshold is a HARD constraint that physically prevents small trades.

## v6 Design
- `base_threshold`: 0.06 → 0.10 (hard filter for ≥10% changes)
- `quiet_threshold`: 0.08 → 0.12
- `inactivity_penalty`: 0.80 → 0.0 (stop punishing holding)
- `alpha_scale`: 2.0 → 1.5 (reduce trading incentive)
- Keep `flat_trade_penalty: 3.0` as additional cost

## Decision
- v5 → **dropped** (trade compression failed, keep summary only)
- v6 → launched, targeting 120-180 trades
