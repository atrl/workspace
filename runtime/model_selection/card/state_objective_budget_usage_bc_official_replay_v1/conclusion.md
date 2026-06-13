# Conclusion

Decision: `budget_usage_bc_official_replay_no_promotion`.

Official replayed `risk_budget_usage_v1` BC warm-start actor from `runtime/model/card/risk_budget_usage_v1/soxl_risk_budget_usage_v1_state_objective_budget_usage_bc_warm_start_v1_actor.pt` through actual safe end `2026-06-09`.
Freshness file safe end `2026-06-11` / latest `2026-06-12`; current qlib data safe end `2026-06-09` / latest `2026-06-10`; mismatch `True`.
Walk-forward readout: wf_2020 return 16.24%, MaxDD 21.35%; wf_2022 return -12.14%, MaxDD 14.80%; wf_2023_2024 return 24.28%, MaxDD 15.51%.
YTD return `27.19%`, MaxDD `17.82%`; required SOXLBacktest baseline `132.08%`, current-best alignment `213.14%`.
Failures: `wf_max_drawdown_ge_20;ytd_return_below_soxlbacktest_baseline;ytd_return_below_best_alignment`.

This card does not promote or retain an actor. It decides whether the existing budget-usage BC actor is strong enough to justify a fold-local PPO continuation under the current retention-style gates.
