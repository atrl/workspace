# Release16 Rulebase Reversal Drawdown/Upside Target

Decision: `release16_rulebase_reversal_drawdown_upside_target_ready_for_checkpoint_replay_no_promotion`.

Source target: `runtime/model_selection/card/release16_rulebase_reversal_checkpoint_readiness_v1/rulebase_reversal_bc_target_contract.csv` from `release16_rulebase_reversal_checkpoint_readiness_v1`.
Reference baseline: `soxlbacktest_qlib_walkforward_baseline_v1:wf_pass_ytd18_15_15_02`.

Target changes:
- wf_2020: `92` high-drawdown rulebase-reversal train rows projected to risk/downtrend target `0.35`.
- wf_2023_2024: `151` train-only upside rows projected to rebound target floor `0.55`.
- wf_2022 guard fold changed rows: `0`.

Scope/mismatch checks: changed eval/test rows `0`, consumed-target mismatches `0`.

Risk note: this card writes only a target contract. It does not train, promote, or retain an actor, and it does not relax the walk-forward gates.

Next action: `rerun_rulebase_reversal_drawdown_upside_checkpoint_metric_replay`.

Artifacts: `drawdown_upside_bc_target_contract.csv`, `drawdown_upside_adjusted_rows.csv`, `target_scope_audit.csv`, `target_distribution_before_after.csv`, `target_bucket_delta.csv`, `target_feature_contrast.csv`, `consumed_target_audit.csv`, `selection.json`, `summary.json`.
