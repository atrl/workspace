# Drawdown-Drift Reward Screen

## Decision

- decision: `drawdown_drift_reward_recipe_ready_for_short_training_screen`
- selected_recipe: `dd1.5_thr18_pos0.15_tf1_floor0.25_lb21`
- training_allowed_next: `True`
- next_mechanism: `short_foldlocal_bc_or_ppo_training_screen`

## Candidate Ranking

| recipe | ready | loss | dd capture | trend capture | overreach | active rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| dd1.5_thr18_pos0.15_tf1_floor0.25_lb21 | True | 1.31 | 0.68 | 1.00 | 0.00 | 107 |
| dd2_thr18_pos0.15_tf1_floor0.25_lb21 | True | 1.40 | 0.68 | 1.00 | 0.00 | 107 |
| dd1.5_thr18_pos0.1_tf1_floor0.25_lb21 | True | 1.48 | 0.78 | 1.00 | 0.00 | 112 |
| dd3_thr18_pos0.15_tf1_floor0.25_lb21 | True | 1.59 | 0.68 | 1.00 | 0.00 | 107 |
| dd1.5_thr18_pos0.05_tf1_floor0.25_lb21 | True | 1.60 | 1.00 | 1.00 | 0.00 | 123 |
| dd2_thr18_pos0.1_tf1_floor0.25_lb21 | True | 1.65 | 0.78 | 1.00 | 0.00 | 112 |
| dd1.5_thr15_pos0.15_tf1_floor0.25_lb21 | True | 1.75 | 0.68 | 1.00 | 0.00 | 107 |
| dd2_thr18_pos0.05_tf1_floor0.25_lb21 | True | 1.84 | 1.00 | 1.00 | 0.00 | 123 |
| dd1.5_thr18_pos0.15_tf1_floor0.35_lb21 | True | 1.87 | 0.68 | 1.00 | 0.00 | 137 |
| dd2_thr18_pos0.15_tf1_floor0.35_lb21 | True | 1.94 | 0.68 | 1.00 | 0.00 | 137 |
| dd3_thr18_pos0.1_tf1_floor0.25_lb21 | True | 1.98 | 0.78 | 1.00 | 0.00 | 112 |
| dd1.5_thr18_pos0.1_tf1_floor0.35_lb21 | True | 1.99 | 0.78 | 1.00 | 0.00 | 142 |

## Readout

- This is a reward-objective readiness screen, not a performance backtest.
- A ready recipe may enter a short training screen, but cannot promote an actor or runtime overlay.
