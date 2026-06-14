# Retraining Execution Handoff Current

Decision: `constrained_release16_state_head_training_handoff_no_promotion`.

Training branch allowed: `True` for `release16_rulebase_reversal_drawdown_upside_state_head_actor_probe_v1`. This is an actor-probe handoff only; promotion and actor retention remain `False`.

Latest replay gaps:
- wf_2020: rel -0.43pp, MaxDD 19.33%, gap 30.43pp, action `repair_drawdown_cap_without_suppressing_reentry; add state-head reward for rebound participation`.
- wf_2022: rel 71.71pp, MaxDD 15.62%, gap 0.00pp, action `guard_only_keep_target_unchanged`.
- wf_2023_2024: rel 2.99pp, MaxDD 13.62%, gap 27.01pp, action `raise sustained-upside participation transfer; add trend/reversal state-head reward and checkpoint penalty`.

Required source target: `runtime/model_selection/card/release16_rulebase_reversal_drawdown_upside_target_v1/drawdown_upside_bc_target_contract.csv`; scope clean `True`, changed eval/test rows `0`.

Primary feature handoff:
- rulebase_reversal/RB_SELL_COMPOSITE: test effect 0.490, train effect 0.712.
- compact_state_head/mom_5: test effect 0.467, train effect 0.408.
- momentum_drawdown/mom_5: test effect 0.467, train effect 0.408.
- rulebase_reversal/RB_RSI6_OVERSOLD: test effect -0.400, train effect -0.360.
- compact_state_head/RB_TREND_REVERSAL: test effect 0.357, train effect 0.678.
- rulebase_reversal/RB_TREND_REVERSAL: test effect 0.357, train effect 0.678.

Hard gates after any training:
- fixed split plus `wf_2020`, `wf_2022`, `wf_2023_2024` on official daily-path metrics.
- every walk-forward fold relative return >=30pp and MaxDD <20%.
- YTD return must clear SOXLBacktest qlib reference and current-best-alignment floors.
- replay export must make the profile runnable in dashboard v2 before retention review.

Risk note: do not rerun the same checkpoint/post-hoc overlay and call it a retained model. The next branch must add the state-head / representation objective and still pass full replay gates.

Artifacts: `evidence_chain.csv`, `fold_gap_to_training_targets.csv`, `feature_handoff.csv`, `top_state_transfer_features.csv`, `current_training_contract.json`, `summary.json`.
