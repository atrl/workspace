# Release16 Drawdown-Drift Reward Recipe v1

Decision: `release16_drawdown_drift_reward_recipe_ready_for_short_training_screen`.

- Selected recipe: `dd1.5_thr18_pos0.05_tf1_floor0.25_lb21`.
- Training allowed next: `True` for short fold-local screen only.
- Drawdown failure capture: `90.32%`.
- Trend-underparticipation capture: `100.00%`.
- Overreach share: `0.00%`.
- Current source path still fails retention: min relative `12.29` pp and worst MaxDD `26.57%`.
- Next mechanism: `short_foldlocal_bc_or_ppo_training_screen`.

Conclusion: use this reward patch for a short fold-local BC/PPO training screen, then rerun official three-fold replay before any actor retention decision. This advances the model path beyond same-family BC weighting while keeping SOXLBacktest qlib as the comparison baseline.
