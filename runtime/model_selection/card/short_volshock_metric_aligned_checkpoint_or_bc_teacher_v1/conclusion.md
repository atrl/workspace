## Conclusion

Decision: `short_volshock_metric_aligned_teacher_ready_no_promotion`.

`short_volshock_cap05_hit2` is preserved only as a metric-aligned teacher/checkpoint reference. It passes `wf_2022` but remains a near-miss in `wf_2023_2024` with relative gap `0.183pp` and MaxDD gap `0.107%` against the `+30pp / <=15%` official replay target. It also lacks final `wf_2020` coverage, so actor retention remains blocked.

The trace evidence supports moving this signal into checkpoint/target design rather than adding another runtime overlay threshold: selected cap order absorption is low, with cap orders much smaller than cap-changed days and high min-rebalance blocking.

Next action: run `short_volshock_teacher_reward_checkpoint_screen_v1` using this recipe as a checkpoint/reward/BC-target contract. The next screen must generate fold-local targets or checkpoint terms, add `wf_2020`, and keep strict retention gates closed until MaxDD `<20%` and current-best return alignment are proven.
