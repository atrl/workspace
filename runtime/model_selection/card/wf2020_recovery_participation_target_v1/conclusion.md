# wf2020_recovery_participation_target_v1 conclusion

Decision: `wf2020_recovery_participation_target_ready_for_checkpoint_replay_no_promotion`.

Selected recipe: `dd12_cap15_recovery65_bridge35_mom02` with risk-only cap `0.15`, recovery floor `0.65`, and conflict bridge `0.35`.

Preview metrics are diagnostic only: return `73.07%`, MaxDD `15.35%`, surrogate gate `True`. Base near-miss was return `61.32%`, MaxDD `20.92%`.

The non-leaking target contract is `runtime/model_selection/card/wf2020_recovery_participation_target_v1/recovery_participation_three_fold_target_contract.csv` with `3731` rows and `949` modified rows. Eval leakage rows: `0`.

Next action: rerun three-fold checkpoint replay with this target contract. Promotion and actor retention remain blocked until official replay passes `wf_2020`, `wf_2022`, and `wf_2023_2024`.
