# return_preserving_hybrid_failure_repair_screen_v1

Decision: `hybrid_failure_repair_screen_no_three_fold_pass_no_actor_promotion`.

The best path-level repair is `scale1.00_cap0.55_stress0.15_dd0.06_mom-0.01_bull0.55_mom0.10_dd0.20` and passes 2/3 fold gates; no actor promotion is allowed.

- wf_2020: return 46.92%, relative -11.18pp, MaxDD 16.42%; still misses the 69.60% floor by 22.69pp.
- wf_2022: return -14.82%, relative 71.77pp, MaxDD 18.76%; stress cap repairs the drawdown gate.
- wf_2023_2024: return 33.69%, relative 34.05pp, MaxDD 17.46%; bull participation repairs the relative-return gate.

Next action: do not train actor from the current hybrid repair; build a stronger wf2020 bull/recovery target before rerunning three-fold BC/checkpoint replay.
