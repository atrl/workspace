# Drawdown-Drift Reward Screen

## Decision

- decision: `drawdown_drift_reward_recipe_ready_for_short_training_screen`
- selected_recipe: `dd1.5_thr15_pos0.05_tf1_floor0.25_lb63`
- training_allowed_next: `True`
- next_mechanism: `short_foldlocal_bc_or_ppo_training_screen`

## Candidate Ranking

| recipe | ready | loss | dd capture | trend capture | overreach | active rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| dd1.5_thr15_pos0.05_tf1_floor0.25_lb63 | True | 0.72 | 0.87 | 1.00 | 0.00 | 169 |
| dd2_thr15_pos0.05_tf1_floor0.25_lb63 | True | 0.82 | 0.87 | 1.00 | 0.00 | 169 |
| dd1.5_thr15_pos0.05_tf1_floor0.25_lb21 | True | 1.01 | 0.87 | 1.00 | 0.00 | 189 |
| dd3_thr15_pos0.05_tf1_floor0.25_lb63 | True | 1.03 | 0.87 | 1.00 | 0.00 | 169 |
| dd1.5_thr15_pos0.05_tf1_floor0.35_lb63 | True | 1.10 | 0.87 | 1.00 | 0.00 | 180 |
| dd2_thr15_pos0.05_tf1_floor0.25_lb21 | True | 1.11 | 0.87 | 1.00 | 0.00 | 189 |
| dd1.5_thr15_pos0.05_tf2_floor0.25_lb63 | True | 1.12 | 0.87 | 1.00 | 0.00 | 169 |
| dd2_thr15_pos0.05_tf1_floor0.35_lb63 | True | 1.20 | 0.87 | 1.00 | 0.00 | 180 |
| dd2_thr15_pos0.05_tf2_floor0.25_lb63 | True | 1.22 | 0.87 | 1.00 | 0.00 | 169 |
| dd3_thr15_pos0.05_tf1_floor0.25_lb21 | True | 1.30 | 0.87 | 1.00 | 0.00 | 189 |

## Readout

- This is a reward-objective readiness screen, not a performance backtest.
- A ready recipe may enter a short training screen, but cannot promote an actor or runtime overlay.
