# wf2020_riskcap_full_bc_replay_v1

Decision: `wf2020_riskcap_full_bc_underparticipation_no_promotion`.

Fold gates passed: `1/3`. MaxDD gates passed: `3/3`. Relative gates passed: `2/3`.

- wf_2020: return=14.06%, relative=-44.04pp (target -263.73pp), MaxDD=14.87% (<20.00% gate=True), avg_pos=0.170, fold_gate=False.
- wf_2022: return=-10.71%, relative=75.88pp (target 30.00pp), MaxDD=19.10% (<20.00% gate=True), avg_pos=0.080, fold_gate=True.
- wf_2023_2024: return=15.26%, relative=15.61pp (target 30.00pp), MaxDD=16.06% (<20.00% gate=True), avg_pos=0.179, fold_gate=False.
- No runtime promotion: at least one required fold still fails the replay gate; use this evidence to design the next target rather than retaining an actor.

Next action: `design_return_preserving_hybrid_target_with_riskcap_floor_then_rerun_three_fold_bc_replay`.

Promotion note: `promotion_allowed=false`, `actor_retention_allowed=false`; this card is a diagnostic replay, not a retained model.
