# v8 Design Notes

## Hypothesis

v2 baseline (hard_floor=0.20, [128,64], alpha_scale=3.0) achieved the best test drawdown so far (50.4%). v5 and v7 both degraded drawdown by raising hard_floor to 0.40 and adding tighter caps. v8 reverts to v2's proven architecture and attempts to improve drawdown control purely via reward shaping.

## Key Changes from v2

1. **drawdown_penalty: 0.75** (was 0.28) — 2.7x stronger step-level drawdown penalty
2. **downside_penalty: 0.40** (was 0.20) — 2x stronger
3. **risk_budget_penalty: 0.40** (was 0.24) — 1.7x stronger
4. **high_exposure_penalty: 0.15** (was 0.08) — 1.9x stronger

## Key Restorations from v2

1. **hard_floor: 0.20** — allows effective bear-market de-risking (v5/v7's 0.40 trapped strategy at 27% avg_pos in 2022 vs v2's 18%)
2. **hidden_sizes: [128, 64]** — avoids overfitting that plagued v7's [256,128]
3. **position_cap: v2 settings** — looser caps that don't limit upside without protecting downside
4. **alpha_scale: 3.0** — higher sensitivity to negative alpha forces faster position adjustments
5. **micro_rebalance: v2 settings** — lighter threshold (0.05) and penalty (0.35)

## Expected Outcome

- Test MaxDD should improve toward or below v2's 50.4% (target: <50%, ideally <45%)
- Test return may decrease vs v5's 780% (trade-off acceptable per user direction)
- Test Sharpe should stay competitive (>0.80)
- 2022 bear-market avg_pos should drop toward v2's 0.18 level

## Risk

If step-level penalties are still too small relative to alpha reward, drawdown may not improve. Next step would be code-level reward redesign (episode-level max drawdown penalty or portfolio-path-dependent cap).
