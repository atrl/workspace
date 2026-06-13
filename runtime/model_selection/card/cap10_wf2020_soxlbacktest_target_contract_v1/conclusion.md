# cap10_wf2020_soxlbacktest_target_contract_v1

- Decision: `wf2020_soxlbacktest_target_contract_ready_for_bc_screen_no_promotion`.
- `wf_2020` target rows: `2470` from qlib-aligned SOXLBacktest replay, with target dates `2010-03-11` to `2019-12-31`.
- Leakage audit: `eval_overlap_rows=0`, `outside_train_eval_rows=0`; no 2020 rows enter the target contract.
- Label health: `rebound_rows=42`, `downtrend_rows=781`, `unique_actions=18`, dominant action share `0.369`.
- Coverage effect: target-contract coverage now exists for `wf_2020`, `wf_2022`, and `wf_2023_2024`, but `wf_2020` still lacks BC slice metrics and checkpoint selection.
- Retention remains blocked: no actor, PPO checkpoint, runtime overlay, or production model is promoted from this notebook.
- Next action: `run_three_fold_bc_slice_screen_before_any_ppo` with drift stop `PPO-vs-BC MaxDD delta <= 2.0pp` and relative delta `>= -5.0pp` before any full retraining continuation.
