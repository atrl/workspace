# Retention YTD Training Objective Handoff v1 Conclusion

Decision: `retention_ytd_training_objective_contract_ready_no_promotion`.

## Current Evidence

- Closest retained-model candidate remains `existing_model_retention_gate::mode_conditioned_rebound_reward_train_v2`; it is diagnostic only and not promotable.
- Best low-drawdown reference remains `soxlbacktest_qlib_reference::wf_pass_ytd18_15_15_02` with YTD `18.31%`, far below the required current-best alignment floor `213.14%`.
- SOXLBacktest qlib buy/sell reference floor is `132.08%`; it is a reference baseline, not runnable inference.
- Current dashboard replay source reports safe_backtest_end `2026-06-11`.

## Training Contract

- Next branch/profile candidate: `state_conditioned_multi_objective_ytd_gate_v1`.
- Candidate screening must use official replay metrics for all WF windows plus dashboard/YTD replay through safe_backtest_end.
- Hard reject any checkpoint with missing WF evidence, worst WF MaxDD >=20%, or YTD return below 213.143462%.
- Keep the stricter <=15% MaxDD margin as a training/checkpoint objective, because prior warm-start candidates near 16-31% MaxDD do not leave enough retention margin.
- Do not increase reentry/trend participation floors unless risk-state coverage and drawdown interval behavior improve together.

## Stop Rules

- `missing_official_replay_evidence` -> reject: candidate must have official replay or dashboard replay YTD plus all required WF windows
- `wf_maxdd_ge_20pct` -> reject: retained model cannot breach requested walk-forward drawdown gate
- `ytd_below_current_best_alignment` -> reject: YTD must be >= 213.143462%
- `ytd_below_soxlbacktest_reference` -> reject_or_downrank_reference: YTD below 132.077978% cannot beat SOXLBacktest reference floor
- `ordinary_vs_official_gate_flip` -> reject_training_handoff: do not use ordinary split metrics to justify PPO/retention
- `risk_state_reentry_drift` -> stop_floor_sweep: if reentry/trend floor improves return but worsens drawdown, revise state representation before longer training

## Sidecars

- `candidate_gap_table.csv`
- `training_objective_recipe.csv`
- `stop_rules.csv`
- `training_contract.json` / `selection.json` / `summary.json`
