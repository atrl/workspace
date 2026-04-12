# SOXL Optimization Notes

Current baseline after the refactor:

- `balanced`: `360.55% / Sharpe 0.799 / MaxDD 53.57% / 95 trades`

Why the refactor underperformed the old best runs:

- The reward moved from hand-shaped threshold rules to a continuous objective, so the policy is no longer being manually pushed into panic-buy / hard de-risk behavior.
- The current reward is more conservative on drawdown, downside volatility, and exposure in high-risk regimes.
- Training budget is still at `200 x 10`; the old best `721%` run came from a much larger manual Tianshou budget and is not a like-for-like comparison.

Highest-priority improvements:

1. Regime-stratified episode sampling
   - Oversample 2020 crash, 2022 bear, and 2023 rebound windows in training.
   - Keep normal periods in the mix so the policy does not become permanently defensive.

2. Two-stage reward schedule
   - Stage 1: higher drawdown and downside penalties so the agent first learns survival.
   - Stage 2: relax penalties slightly and increase return weight to recover upside.

3. Action-space redesign
   - Replace fixed 11-level target positions with a hybrid action:
   - coarse target position + small delta adjustment, or fewer actions biased around `0.0/0.25/0.5/0.75/1.0`.
   - This should reduce noise trading while keeping regime response.

Daily-data limitation:

- The current framework is daily only.
- "Do T more aggressively" can only mean more active daily rebalancing, not true intraday T.
- To do real T, the active path would need minute bars and a different execution model.

4. Regime-aware reward budget
   - The current `risk_budget_penalty` is linear.
   - Try a sigmoid budget using `regime_risk`, so penalties stay mild in normal periods and rise sharply in bear regimes.

8. Do not reward turnover unconditionally
   - High turnover alone increased drawdown in the daily setup.
   - Turnover should only be rewarded when volatility is elevated or the regime is changing.
   - In low-edge periods, turnover is mostly cost and whipsaw.

5. Add explicit persistence features
   - Add `drawdown_streak`, `rolling_new_low_ratio`, `below_ma63_days`, and `negative_gap_frequency`.
   - 2022 was persistent stress, not a single panic spike.

6. Separate train and model-selection metrics
   - Select checkpoints on a weighted score:
   - `0.45 * Sharpe + 0.30 * return - 0.25 * max_dd`
   - Also require a minimum 2022 exposure-control score.

7. Save checkpoints by profile
   - Persist actor weights every N iterations and compare 100/150/200/250 checkpoints.
   - The best policy may not be the last policy.

Recommended next experiments:

- `balanced_regime`: stronger 2022 oversampling, same reward shape.
- `balanced_two_stage`: same features, staged reward schedule.
- `balanced_coarse_action`: keep reward, simplify action space.
- `aggressive_regime`: lighter drawdown penalty, but keep regime budget.
