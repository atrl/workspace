# Release16 Post-Short-Training Reward/Checkpoint Contract

## Decision

- decision: `release16_post_short_training_contract_no_training`
- selected candidate: `bc_initialization`
- promotion_allowed: `False`
- actor_retention_allowed: `False`
- training_allowed_next: `False`
- primary blocker: `bc_candidate_or_slice_gate_failed`

## Gate Readout

- Required gate: relative return >= `30.00` pp and MaxDD < `20.00%` on `wf_2020, wf_2022, wf_2023_2024`.
- Selected candidate min relative return: `-264.73` pp.
- Selected candidate worst MaxDD: `26.83%`.
- Failed folds: `wf_2020, wf_2022, wf_2023_2024`.

## Objective Terms

- `checkpoint_objective_loss`
- `drawdown_drift_penalty`
- `trend_preservation_floor`

## BC Slice Blocker

The reward/checkpoint objective terms are expressible in the current reward/config code, but the short-training BC holdout slice gate is still false on the selected folds. Re-running longer PPO from the same weak BC target is not allowed by the review rules.

Next action: `rebuild_fold_local_joint_target_with_holdout_slice_gate_then_reward_checkpoint_screen`.

Artifacts: `gate_gaps.csv`, `reward_checkpoint_scores.csv`, `bc_holdout_slice_readout.csv`, `reward_checkpoint_review.md`, `training_contract.json`, `summary.json`.
