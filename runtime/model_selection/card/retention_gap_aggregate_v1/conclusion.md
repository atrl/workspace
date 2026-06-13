# Retention Gap Aggregate v1 Conclusion

Decision: `retention_gap_aggregate_no_promotion_train_state_conditioned_multi_objective`.

Scored `61` candidate/source rows from `9` evidence inputs. Full retained-model gate pass count: `0`.

## Best Current Evidence
- Closest aggregate candidate: `existing_model_retention_gate::mode_conditioned_rebound_reward_train_v2` with objective loss `53.89`; bottleneck `joint_drawdown_and_return_gap`; failures `worst_wf_max_drawdown_ge_20;ytd_return_below_current_best_alignment`.
- Best candidate that satisfies WF drawdown by YTD return: `soxlbacktest_qlib_reference::wf_pass_ytd18_15_15_02`; YTD `18.31%`, worst WF MaxDD `18.51%`; still fails `ytd_return_below_soxlbacktest_reference;ytd_return_below_current_best_alignment`.
- No complete candidate with usable WF evidence satisfies the current-best YTD alignment gate.

## Mechanism Readout
- Low-drawdown paths exist, but they underparticipate and miss both SOXLBacktest and current-best YTD return floors.
- High-YTD existing models exist, but their complete walk-forward evidence still breaches the 20% MaxDD gate or is missing required windows.
- SOXLBacktest qlib buy/sell reference remains useful as a return/timing baseline, but its raw teacher path is not promotable because WF drawdown is too high; risk-capped teacher variants lose too much return.
- QuantBench current ensemble is the clearest low-DD training-target style, but its YTD return is far below the reference floors, so it is a teacher component, not a retained model.

## Next Action
- Route the next training branch to `train_state_conditioned_multi_objective_with_ytd_participation_and_drawdown_gate`.
- The next actor screen must use official replay metrics as the selection source, include all three WF windows plus YTD through safe replay end, and reject any checkpoint with WF MaxDD >=20% or YTD <213.14%.
- Do not retain or promote any current candidate from this aggregate card.

## Sidecars
- `candidate_gate_summary.csv`
- `candidate_window_metrics.csv`
- `mechanism_gap_summary.csv`
- `failure_counts.csv`
- `selection.json` / `summary.json`