# Drawdown-Drift Reward Screen

## Decision

- decision: `drawdown_drift_reward_recipe_ready_for_short_training_screen`
- selected_recipe: `dd1.5_thr18_pos0.15_tf1_floor0.25_lb21`
- training_allowed_next: `True`
- next_mechanism: `short_foldlocal_bc_or_ppo_training_screen`

## Candidate Ranking

| recipe | ready | loss | dd capture | trend capture | overreach | active rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| dd1.5_thr18_pos0.15_tf1_floor0.25_lb21 | True | 1.51 | 1.00 | 1.00 | 0.00 | 74 |
| dd1.5_thr18_pos0.1_tf1_floor0.25_lb21 | True | 1.53 | 1.00 | 1.00 | 0.00 | 74 |
| dd1.5_thr18_pos0.05_tf1_floor0.25_lb21 | True | 1.56 | 1.00 | 1.00 | 0.00 | 74 |
| dd2_thr18_pos0.15_tf1_floor0.25_lb21 | True | 1.60 | 1.00 | 1.00 | 0.00 | 74 |
| dd2_thr18_pos0.1_tf1_floor0.25_lb21 | True | 1.63 | 1.00 | 1.00 | 0.00 | 74 |
| dd2_thr18_pos0.05_tf1_floor0.25_lb21 | True | 1.67 | 1.00 | 1.00 | 0.00 | 74 |
| dd3_thr18_pos0.15_tf1_floor0.25_lb21 | True | 1.78 | 1.00 | 1.00 | 0.00 | 74 |
| dd3_thr18_pos0.1_tf1_floor0.25_lb21 | True | 1.83 | 1.00 | 1.00 | 0.00 | 74 |
| dd3_thr18_pos0.05_tf1_floor0.25_lb21 | True | 1.88 | 1.00 | 1.00 | 0.00 | 74 |
| dd1.5_thr18_pos0.15_tf1_floor0.35_lb21 | True | 1.91 | 1.00 | 1.00 | 0.00 | 111 |

## Readout

- This is a reward-objective readiness screen, not a performance backtest.
- A ready recipe may enter a short training screen, but cannot promote an actor or runtime overlay.
