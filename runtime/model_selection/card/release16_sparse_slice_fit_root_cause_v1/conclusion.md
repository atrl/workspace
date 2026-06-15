# release16 sparse slice fit root-cause diagnostic

## Decision

`release16_sparse_slice_target_repair_required_no_training_no_promotion`. No actor retention, no promotion, and no PPO handoff from this card.

## Evidence

- Selected official replay gate count remains `0/3`; min relative return is `-40.5841056381` pp and worst MaxDD is `35.85099103981264`%.
- Missing target slice coverage rows: `3`; sparse under-sample rows: `3`; action-fit failures: `2`.
- `drawdown_control_state` has zero target rows in `wf_2022` and `wf_2023_2024`.
- `release_gate_state` has only 8/11 selected BC samples in `wf_2022`/`wf_2023_2024`, below the 20-row gate.
- Existing feature groups can separate these labels for: `is_downtrend_state, release_gate_state`.
- State-head lift stability gate: `False` with unstable effective lift rate `0.279`.

## Next Action

Build a fold-local joint drawdown/release target contract and auxiliary state/action-head screen before another BC official replay. Keep the guard-fold state-head slice as source-scope coverage, not a standalone learned-slice failure, but retain release/rebound/high-participation gates.
