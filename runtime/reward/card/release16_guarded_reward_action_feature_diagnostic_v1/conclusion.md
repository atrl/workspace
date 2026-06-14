# Conclusion

- decision: `reward_ppo_wf2020_underparticipation_action_drift_no_promotion`
- reward_ppo_gate_count_20dd: `1/3`
- bc_gate_count_20dd: `1/3`
- promotion_allowed: `False`
- actor_retention_allowed: `False`

## Readout

- Reward PPO remains non-promotable: 1/3 primary 20% gates, versus BC 1/3.
- wf_2020 is the critical regression: relative return delta is -37.54pp while MaxDD is roughly unchanged (21.11% -> 21.11%).
- wf_2020 average position changed by -0.027; action_changed_rate=10.7%.
- Worst 21-day PPO-vs-BC window: 2020-11-05 to 2020-12-04, delta=-14.36pp, avg position delta=0.000.
- No actor is promoted or retained; the next step should change state/action objective before another reward PPO run.

## Next Action

- `state_conditioned_trend_participation_checkpoint_or_auxiliary_action_head_before_reward_ppo`
