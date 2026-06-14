# Conclusion

- decision: `release16_guarded_reward_short_ppo_drift_failed_no_promotion`
- selected_gate_pass_count_20dd: `1/3`
- bc_gate_pass_count_20dd: `1/3`
- promotion_allowed: `False`
- actor_retention_allowed: `False`

## Readout

- wf_2020: BC rel=22.28pp / MaxDD=21.11%; reward PPO rel=-15.26pp / MaxDD=21.11%; delta rel=-37.54pp, delta MaxDD=0.01pp, fold_gate=False.
- wf_2022: BC rel=72.45pp / MaxDD=15.07%; reward PPO rel=72.45pp / MaxDD=15.07%; delta rel=0.00pp, delta MaxDD=0.00pp, fold_gate=True.
- wf_2023_2024: BC rel=14.30pp / MaxDD=13.96%; reward PPO rel=14.30pp / MaxDD=13.96%; delta rel=0.00pp, delta MaxDD=0.00pp, fold_gate=False.
- Selected reward PPO passes 1/3 primary 20% gates and 0/3 strict 15% gates; BC initialization passes 1/3 primary gates.
- At least one fold fails the PPO-vs-BC drift guard, so longer PPO from this reward patch is blocked.
- No actor is promoted or retained from this notebook.

## Next Action

- `do_not_launch_longer_ppo_from_this_reward_patch; inspect_failed_fold_action_distribution`
