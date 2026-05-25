# Gate-Gap Failure Attribution

## External References

- [Microsoft Qlib](https://github.com/microsoft/qlib): repo workflow and reproducible quant platform boundary
- [Qlib arXiv](https://arxiv.org/abs/2009.11189): AI-oriented quantitative investment workflow
- [Deflated Sharpe Ratio](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551): multiple-testing / selection-bias control
- [Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-overfitting.pdf): walk-forward and PBO-style anti-overfit guard

## Profile Summary

| profile | windows | failures | dominant type | worst window | severity | next mechanism |
| --- | ---: | ---: | --- | --- | ---: | --- |
| mode_conditioned_rebound_reward_train_v2 | 8 | 8 | joint_tradeoff | test_proxy | 2429.81 | walk_forward_weighted_multi_objective_selection |
| drawdown_rebound_reward_train_v1 | 1 | 1 | joint_tradeoff | main | 2283.21 | walk_forward_weighted_multi_objective_selection |
| state_conditioned_drawdown_rebound_reward_train_v1 | 4 | 4 | joint_tradeoff | test_proxy | 1785.68 | walk_forward_weighted_multi_objective_selection |
| drawdown_rebound_checkpoint_pbo_v1 | 4 | 4 | drawdown_control | strict_oos | 395.50 | drawdown_controller_or_sequence_risk_feature |
| state_conditioned_actor_selector_v1 | 4 | 4 | drawdown_control | wf_2022 | 347.75 | drawdown_controller_or_sequence_risk_feature |

## Window Attribution

| profile | window | rel return | max dd | avg pos | type | path |
| --- | --- | ---: | ---: | ---: | --- | --- |
| mode_conditioned_rebound_reward_train_v2 | test_proxy | -2107.47pp | 44.23% | 0.268 | joint_tradeoff | walk_forward_weighted_multi_objective_selection |
| drawdown_rebound_reward_train_v1 | main | -2026.44pp | 37.68% | 0.281 | joint_tradeoff | walk_forward_weighted_multi_objective_selection |
| state_conditioned_drawdown_rebound_reward_train_v1 | test_proxy | -1470.03pp | 43.57% | 0.383 | joint_tradeoff | walk_forward_weighted_multi_objective_selection |
| mode_conditioned_rebound_reward_train_v2 | wf_2020 | -304.71pp | 23.09% | 0.297 | joint_tradeoff | walk_forward_weighted_multi_objective_selection |
| state_conditioned_drawdown_rebound_reward_train_v1 | wf_2020 | -275.09pp | 24.41% | 0.432 | joint_tradeoff | walk_forward_weighted_multi_objective_selection |
| mode_conditioned_rebound_reward_train_v2 | test_2020 | -304.71pp | 21.43% | 0.297 | joint_tradeoff | walk_forward_weighted_multi_objective_selection |
| drawdown_rebound_checkpoint_pbo_v1 | strict_oos | 610.30pp | 54.55% | 0.000 | drawdown_control | drawdown_controller_or_sequence_risk_feature |
| state_conditioned_actor_selector_v1 | wf_2022 | 39.33pp | 49.77% | 0.273 | drawdown_control | drawdown_controller_or_sequence_risk_feature |
| mode_conditioned_rebound_reward_train_v2 | test_2022 | 65.02pp | 43.91% | 0.131 | drawdown_control | drawdown_controller_or_sequence_risk_feature |
| state_conditioned_actor_selector_v1 | strict_oos | 851.29pp | 43.53% | 0.450 | drawdown_control | drawdown_controller_or_sequence_risk_feature |
| drawdown_rebound_checkpoint_pbo_v1 | wf_2023_2024 | -15.81pp | 31.46% | 0.000 | joint_tradeoff | walk_forward_weighted_multi_objective_selection |
| mode_conditioned_rebound_reward_train_v2 | valid | -77.91pp | 21.22% | 0.307 | joint_tradeoff | walk_forward_weighted_multi_objective_selection |
| state_conditioned_actor_selector_v1 | wf_2023_2024 | 20.43pp | 30.71% | 0.469 | joint_tradeoff | walk_forward_weighted_multi_objective_selection |
| drawdown_rebound_checkpoint_pbo_v1 | wf_2022 | 65.58pp | 30.72% | 0.000 | drawdown_control | drawdown_controller_or_sequence_risk_feature |
| state_conditioned_actor_selector_v1 | wf_2020 | 11.36pp | 28.64% | 0.387 | joint_tradeoff | walk_forward_weighted_multi_objective_selection |
| mode_conditioned_rebound_reward_train_v2 | wf_2023_2024 | 42.44pp | 30.49% | 0.255 | drawdown_control | drawdown_controller_or_sequence_risk_feature |
| state_conditioned_drawdown_rebound_reward_train_v1 | wf_2022 | 45.13pp | 29.15% | 0.244 | drawdown_control | drawdown_controller_or_sequence_risk_feature |
| mode_conditioned_rebound_reward_train_v2 | test_2023_2024 | -10.06pp | 22.18% | 0.261 | joint_tradeoff | walk_forward_weighted_multi_objective_selection |
| drawdown_rebound_checkpoint_pbo_v1 | wf_2020 | 53.55pp | 22.46% | 0.000 | drawdown_control | drawdown_controller_or_sequence_risk_feature |
| mode_conditioned_rebound_reward_train_v2 | wf_2022 | 61.69pp | 15.90% | 0.141 | near_miss_drawdown | cap_calibration_or_execution_friction |
| state_conditioned_drawdown_rebound_reward_train_v1 | wf_2023_2024 | 36.99pp | 15.23% | 0.163 | near_miss_drawdown | cap_calibration_or_execution_friction |

## Readout

- Treat joint trade-off rows as evidence against another single-coefficient reward sweep.
- Use near-miss drawdown rows for cap calibration or execution friction checks before retraining.
- Use underparticipation or trend-capture rows to justify participation-aware objectives or bull guards.
