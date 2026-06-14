# return_preserving_hybrid_target_bc_replay_v1

Decision: `return_preserving_hybrid_bc_replay_no_promotion`.

Selected candidate: `hybrid_blend_nonrisk_50__riskcap_default_weight`.

Fold gates passed: `0/3`. MaxDD gates passed: `2/3`. Relative gates passed: `2/3`.

- Selected global candidate `hybrid_blend_nonrisk_50__riskcap_default_weight` from 6 candidate groups.
- wf_2020: return=12.79%, relative=-45.31pp (target -263.73pp), MaxDD=14.83% (<20.00% gate=True), avg_pos=0.182, fold_gate=False.
- wf_2022: return=-21.05%, relative=65.54pp (target 30.00pp), MaxDD=21.74% (<20.00% gate=False), avg_pos=0.135, fold_gate=False.
- wf_2023_2024: return=14.03%, relative=14.38pp (target 30.00pp), MaxDD=14.65% (<20.00% gate=True), avg_pos=0.168, fold_gate=False.
- No runtime promotion: the selected hybrid still fails at least one required fold gate.

Next action: `inspect_selected_hybrid_failure_and_repair_participation_weight_or_target_before_retention_training`.

Promotion note: `promotion_allowed=false`, `actor_retention_allowed=false`; this card is a diagnostic replay, not a retained model.
