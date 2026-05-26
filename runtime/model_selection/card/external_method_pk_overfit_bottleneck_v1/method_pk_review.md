# External Method PK

## Decision

- decision: `method_pk_next_batch_selected`
- primary_next: `constrained_checkpoint_selector`
- secondary_next: `reward_constrained_short_training_screen`
- parallel_research_next: `single_factor_soxx_soxl_spread_ic`

## Ranking

| rank | method | category | score | fit | data | overfit ctrl | cost | first experiment |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | constrained_checkpoint_selector | model_selection | 4.56 | 5 | 4 | 5 | 2 | runtime/model_selection/card/<constrained_checkpoint_selector>.ipynb over existing checkpoint_objective_scores + new short-training checkpoints. |
| 2 | reward_constrained_short_training_screen | reward | 4.56 | 5 | 5 | 4 | 3 | runtime/reward/card/<short_training_screen>.ipynb using reward_objective_patch.json, then official wf_2022/wf_2023_2024 replay. |
| 3 | single_factor_soxx_soxl_spread_ic | single_factor | 4.46 | 4 | 5 | 5 | 2 | runtime/factor/card/<spread_ic_screen>.ipynb with IC, signal accuracy, single-signal WF gate. |
| 4 | multi_factor_state_guard | multi_factor | 4.22 | 5 | 4 | 4 | 3 | runtime/model_selection/card/<state_guard_screen>.ipynb with fold-local thresholds and no runtime promotion. |
| 5 | tft_or_gated_attention_feature_selector | network | 3.02 | 3 | 3 | 3 | 4 | runtime/network/card/<gated_feature_selector>.ipynb with tiny capacity and walk-forward ablation vs MLP/GRU. |
| 6 | decision_transformer_offline_bc | model | 2.68 | 3 | 3 | 2 | 5 | runtime/model_selection/card/<decision_transformer_bc_probe>.ipynb on frozen trajectories and small context windows. |
| 7 | tdqn_or_q_learning_action_value_probe | model | 2.58 | 2 | 3 | 3 | 4 | runtime/model_selection/card/<discrete_value_probe>.ipynb with fixed buckets and purged walk-forward. |
| 8 | twitter_x_idea_watchlist_only | external_watchlist | 1.40 | 1 | 1 | 1 | 1 | None until an idea is backed by a paper, GitHub project, dataset, or reproducible feature definition. |

## Conclusion

- Co-primary next: pair the constrained checkpoint selector with the constrained reward short-training screen; the selector is the gate and the short training screen generates the candidate.
- Operational order: run the short training screen, then accept or reject it through the constrained checkpoint selector.
- Parallel low-risk research: screen SOXX/SOXL spread and related volatility/trend features as single factors before feeding them into RL.
- Network/model upgrades such as TFT-style gating or Decision Transformer are deferred until the reward/constraint signal is proven, because they carry higher overfit and engineering risk.

## Do Not Do Next

- `large_transformer_actor_before_reward_constraint_screen`
- `runtime_overlay_promotion_from_objective_screen`
- `twitter_x_idea_without_paper_code_data_gate`
