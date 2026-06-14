# WF2020 Recovery Checkpoint Failure Attribution v1

Decision: `wf2020_recovery_target_semantics_mismatch_blocks_checkpoint_replay_no_promotion`. No actor is promoted.

What ran:
- compared latest recovery checkpoint replay against prior bull-reentry checkpoint replay;
- audited the recovery target contract against the current `state_objective_target_position_from_row` target chooser;
- refreshed sidecar evidence in this card.

Key metrics:
- Strict selected replay pass count: 2/3 folds.
- `wf_2020` BC return changed from 55.597960% to 28.136997% (-27.460963 pp).
- `wf_2020` BC relative return delta vs prior: -27.460963 pp.
- `wf_2020` BC MaxDD delta vs prior: -0.626254 pp.
- `wf_2020_train` changed-target rows: 949 of 2470.
- `wf_2020_train` rows where recovery target differs from consumed BC target: 226 (0.091498).

Root cause:
`recovery_participation_target_position` is not a consumed target surface for current state-objective BC labels. The current target chooser selects `risk_target_position` for downtrend rows, `rebound_target_position` for rebound rows, and `teacher_position` otherwise. Any recovery semantics that are only represented in the new recovery column are therefore diagnostic, not training behavior.

Risk notes:
- Continuing checkpoint replay on this contract would test a partially mismatched label surface.
- A shared BC target-priority change is higher blast radius and needs unit tests before use.

Next action:
Project recovery semantics into the existing consumed columns/state flags, or add a tested explicit target-priority interface, then rerun the three-fold checkpoint metric replay.

Artifacts:
- `candidate_delta_vs_prior.csv`
- `target_semantics_audit.csv`
- `ignored_recovery_target_rows.csv`
- `selection_by_fold_sources.csv`
- `summary.json`
- `selection.json`
