# Release16 Joint Target Holdout Slice Repair

Decision: `release16_joint_target_holdout_slice_repair_contract_no_training`.

The fold-local joint target contract is still target-side ready: scope gate `True`, lift gate `True`, and train label readiness `True`. That does not authorize training because the latest post-short holdout slice gate is `False` across required folds `wf_2020, wf_2022, wf_2023_2024`.

Worst selected MaxDD remains `26.83%` versus target `< 20.00%`, and minimum selected relative return remains `-264.73pp` versus target `>= 30.00pp`. Failed folds: `wf_2020, wf_2022, wf_2023_2024`.

No actor is retained. No PPO or full reward-checkpoint replay should start from this contract until a state-conditioned action head, auxiliary state loss, or equivalent target enrichment passes the holdout slice gate before official three-fold replay.
