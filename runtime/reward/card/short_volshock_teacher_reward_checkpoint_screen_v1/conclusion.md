# short_volshock_teacher_reward_checkpoint_screen_v1 Conclusion

Decision: `reward_checkpoint_screen_no_promotion`.

Selected candidate `teacher_short_volshock_cap05_hit2` has evaluated-window min relative return 29.82pp and max MaxDD 15.11%.
Coverage-scored min relative return is 0.00pp because missing windows are explicit failure rows.
Missing final windows: wf_2020.
Required terms: checkpoint_objective_loss, drawdown_drift_penalty, short_volshock_state_deleverage_consistency, trend_preservation_floor.

No promotion, no actor retention, and no longer PPO are allowed from this card. The teacher is usable only as checkpoint-scoring or fold-local target-generation reference after a target-scope leakage audit.

Next action: `implement_consumed_reward_or_checkpoint_recipe_then_fold_local_target_scope_audit`.

Contract sidecar: `reward_checkpoint_recipe_contract.json` keeps training blocked until the recipe is consumed in code, fold-local target scope is audited, `wf_2020` official replay coverage is added, and strict final walk-forward retention is rerun.
