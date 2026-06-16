# Drawdown-Drift Reward Screen

## Decision

- decision: `drawdown_drift_reward_recipe_ready_for_short_training_screen`
- selected_recipe: `dd1.5_thr18_pos0.15_tf1_floor0.35_lb21`
- training_allowed_next: `True`
- next_mechanism: `short_foldlocal_bc_or_ppo_training_screen`

## Candidate Ranking

| recipe | ready | loss | dd capture | trend capture | overreach | active rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| dd1.5_thr18_pos0.15_tf1_floor0.35_lb21 | True | 2.36 | 0.87 | 1.00 | 0.00 | 226 |
| dd1.5_thr18_pos0.15_tf1_floor0.25_lb21 | True | 2.48 | 0.87 | 1.00 | 0.00 | 153 |
| dd1.5_thr18_pos0.1_tf1_floor0.35_lb21 | True | 2.52 | 0.87 | 1.00 | 0.00 | 226 |
| dd2_thr18_pos0.15_tf1_floor0.35_lb21 | True | 2.61 | 0.87 | 1.00 | 0.00 | 226 |
| dd1.5_thr18_pos0.05_tf1_floor0.35_lb21 | True | 2.68 | 0.87 | 1.00 | 0.00 | 226 |
| dd1.5_thr18_pos0.1_tf1_floor0.25_lb21 | True | 2.72 | 0.87 | 1.00 | 0.00 | 153 |
| dd2_thr18_pos0.1_tf1_floor0.35_lb21 | True | 2.82 | 0.87 | 1.00 | 0.00 | 226 |
| dd2_thr18_pos0.15_tf1_floor0.25_lb21 | True | 2.85 | 0.87 | 1.00 | 0.00 | 153 |
| dd1.5_thr15_pos0.15_tf1_floor0.35_lb21 | True | 2.93 | 1.00 | 1.00 | 0.00 | 235 |
| dd1.5_thr18_pos0.05_tf1_floor0.25_lb21 | True | 2.96 | 0.87 | 1.00 | 0.00 | 153 |
| dd2_thr18_pos0.05_tf1_floor0.35_lb21 | True | 3.04 | 0.87 | 1.00 | 0.00 | 226 |
| dd3_thr18_pos0.15_tf1_floor0.35_lb21 | True | 3.11 | 0.87 | 1.00 | 0.00 | 226 |

## Readout

- This is a reward-objective readiness screen, not a performance backtest.
- A ready recipe may enter a short training screen, but cannot promote an actor or runtime overlay.
