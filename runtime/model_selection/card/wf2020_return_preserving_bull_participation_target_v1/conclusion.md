# wf2020_return_preserving_bull_participation_target_v1 conclusion

Decision: `return_preserving_bull_participation_target_ready_for_official_checkpoint_replay_no_promotion`.

What changed:
- risk/conflict v2 rows are now projected into `is_downtrend_state=True` so BC consumes `risk_target_position`;
- recovery-only v2 rows are now projected into `is_rebound_cluster=True` so BC consumes `rebound_target_position`;
- non-v2 rows keep source state flags and targets.

Evidence:
- Source mismatch rows: `226`.
- Projected mismatch rows: `0`.
- State-flag changed rows: `300`.
- Target delta rows: `949`.
- Eval leakage rows: `0`.

Output target contract: `runtime/model_selection/card/wf2020_return_preserving_bull_participation_target_v1/return_preserving_bull_participation_three_fold_target_contract.csv`.

Promotion remains blocked. Next action: official-daily three-fold checkpoint replay with this contract.
