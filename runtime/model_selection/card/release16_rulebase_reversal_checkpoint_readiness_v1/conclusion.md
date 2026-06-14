# release16_rulebase_reversal_checkpoint_readiness_v1

- Decision: `rulebase_reversal_training_contract_ready_for_checkpoint_replay_no_promotion`.
- Overlay rows: `569`; target contract rows: `3731`.
- Failed-fold coverage: `wf_2020=424`, `wf_2023_2024=145`.
- Mean absolute consumed target lift: `0.188`; consumed target mismatches: `0`.
- Gates: target contract `True`, BC readiness `True`, BC/checkpoint screen allowed next `True`.
- Next action: `run_rulebase_reversal_bc_checkpoint_metric_replay_with_soxlbacktest_baseline_and_walkforward_gates`.

No actor is promoted or retained. The next card must run fresh BC/checkpoint replay metrics against the SOXLBacktest qlib buy/sell baseline and the walk-forward retention gates.
