# release16_nonconvex_checkpoint_objective_score_handoff_v1 Conclusion

## Decision

- decision: `nonconvex_checkpoint_objective_scored_training_contract_no_actor_retention`
- promotion_allowed: `False`
- actor_retention_allowed: `False`
- next_mechanism: `train_fold_local_nonconvex_checkpoint_actor_with_relative_floor_drawdown_drift_stop_and_sparse_slice_guard_then_official_three_fold_replay`

## Key Metrics

- objective_pass_candidate_count: `0`; runnable_objective_pass_candidate_count: `0`.
- best runnable reference: `release16_bull_reentry_participation_objective_replay_v1::bull65_hold3` with min relative `14.30pp`, max MaxDD `21.11%`, objective loss `45.63`.
- best diagnostic training target: `release16_nonconvex_checkpoint_objective_readiness_v1::per_fold_teacher_mix` with min relative `21.33pp`, max MaxDD `16.65%`, objective loss `35.88`.
- sparse slice gate: `False`; failed folds: `wf_2020,wf_2022,wf_2023_2024`.

## Risk Notes

- The best diagnostic row is a per-fold teacher mix, not a replayable actor. It cannot be promoted or retained.
- The best runnable row still fails relative-return floors and breaches the 20% drawdown gate in wf_2020.
- The next run must train against this objective and then pass official wf_2020/wf_2022/wf_2023_2024 replay before retention.

## Artifacts

- `objective_candidate_scores.csv`
- `objective_window_terms.csv`
- `training_objective_handoff.json` / `training_objective_handoff.md`
- `objective_contract.json`
- `summary.json` / `selection.json`
