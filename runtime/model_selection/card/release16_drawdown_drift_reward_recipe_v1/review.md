# Drawdown-Drift Reward Screen

## Decision

- decision: `drawdown_drift_reward_recipe_ready_for_short_training_screen`
- selected_recipe: `dd1.5_thr18_pos0.05_tf1_floor0.25_lb21`
- training_allowed_next: `True`
- next_mechanism: `short_foldlocal_bc_or_ppo_training_screen`

## Candidate Ranking

| recipe | ready | loss | dd capture | trend capture | overreach | active rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| dd1.5_thr18_pos0.05_tf1_floor0.25_lb21 | True | 1.72 | 0.90 | 1.00 | 0.00 | 121 |
| dd1.5_thr18_pos0.05_tf1_floor0.45_lb21 | True | 1.83 | 0.90 | 1.00 | 0.00 | 252 |
| dd1.5_thr18_pos0.05_tf1_floor0.35_lb21 | True | 1.83 | 0.90 | 1.00 | 0.00 | 173 |
| dd1.5_thr18_pos0.1_tf1_floor0.25_lb21 | True | 1.83 | 0.50 | 1.00 | 0.00 | 96 |
| dd1.5_thr18_pos0.1_tf1_floor0.45_lb21 | True | 1.89 | 0.50 | 1.00 | 0.00 | 227 |
| dd1.5_thr18_pos0.1_tf1_floor0.35_lb21 | True | 1.92 | 0.50 | 1.00 | 0.00 | 148 |
| dd2_thr18_pos0.05_tf1_floor0.45_lb21 | True | 1.99 | 0.90 | 1.00 | 0.00 | 252 |
| dd2_thr18_pos0.1_tf1_floor0.45_lb21 | True | 2.02 | 0.50 | 1.00 | 0.00 | 227 |
| dd2_thr18_pos0.05_tf1_floor0.25_lb21 | True | 2.06 | 0.90 | 1.00 | 0.00 | 121 |
| dd2_thr18_pos0.05_tf1_floor0.35_lb21 | True | 2.07 | 0.90 | 1.00 | 0.00 | 173 |
| dd2_thr18_pos0.1_tf1_floor0.35_lb21 | True | 2.13 | 0.50 | 1.00 | 0.00 | 148 |
| dd2_thr18_pos0.1_tf1_floor0.25_lb21 | True | 2.15 | 0.50 | 1.00 | 0.00 | 96 |

## Readout

- This is a reward-objective readiness screen, not a performance backtest.
- A ready recipe may enter a short training screen, but cannot promote an actor or runtime overlay.
