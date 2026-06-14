# release16_checkpoint_distillation_bc_repair_screen_v1

Decision: `bc_repair_weighting_ready_for_checkpoint_replay_no_promotion`.
Selected config: `release128_cb995_obj000`.
Next mechanism: `wire_repaired_bc_weighting_into_checkpoint_metric_replay_before_ppo`.

No actor checkpoint is retained or promoted from this screen.

Key evidence:
- Source failure: `wf_2023_2024 release_gate_state` accuracy 0.3636 on 11 samples with one predicted action.
- Best release-gate pass count: 3 / 3.
- Selected `wf_2023_2024 release_gate_state` accuracy: 0.8181818181818182.
- Selected all-fit gate: True.
