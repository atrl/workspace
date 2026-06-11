## Conclusion

- Scheduler/data fix status: LaunchAgent paths now point at this repo and scheduled preopen/postclose runs use the current `v19reg_v22_budget_usage_bull_guard_v1` runtime composer. The manual postclose run refreshed data/dashboard successfully; Futu OpenD was unreachable on `127.0.0.1:11111`, so the scheduler retried as a local paper dry-run and still produced the current signal artifact.
- Data status: canonical SOXL/SOXX/VIX panel is fresh through `2026-06-10`; dashboard model replays are safe through `2026-06-09`. The model-vs-SOXL comparison below is therefore apples-to-apples through `2026-06-09`.
- YTD 2026 common window (`2026-01-02` to `2026-06-09`): SOXL buy-and-hold `326.93%`; current model `3.25%`; excess `-323.68pp`; model max drawdown `21.09%` vs SOXL `43.47%`.
- June 2026 common window (`2026-06-01` to `2026-06-09`): SOXL buy-and-hold `-11.17%`; current model `0.00%`; excess `11.17pp`; model max drawdown `0.00%` vs SOXL `34.93%`.
- Market-only latest readout through `2026-06-10`: SOXL buy-and-hold is `282.41%` YTD and `-20.43%` for June.
- Latest current signal artifact: `2026-06-10` profile `v19reg_v22_budget_usage_bull_guard_v1`, dry_run=`True`, executed=`False`, target_position=`0.205`, risk_gate_approved=`True`.
- Decision: scheduler/data/dashboard path is unblocked for local evidence generation, but real Futu execution proof still needs OpenD back online. In the common windows, the current model severely lagged SOXL YTD because SOXL rallied while the model stayed defensive, but it protected the June selloff by staying flat.
