# soxlbacktest_qlib_walkforward_baseline_v1

- Decision: `soxlbacktest_reference_qlib_aligned_not_promotable`.
- `legacy_default` is qlib-calendar aligned and remains the buy/sell reference baseline, but fails the walk-forward drawdown gate.
- `legacy_default` worst WF MaxDD is `84.92%` with YTD return `132.08%`.
- `wf_pass_ytd18_15_15_02` is the strongest low-position probe found here that passes the <20% WF drawdown gate, but its YTD return is still too low for the SOXLBacktest/current-best return gates.
- `wf_pass_ytd18_15_15_02` worst WF MaxDD is `18.51%` with YTD return `18.31%`.
- No screened SOXLBacktest parameter variant is promotable. Use these sidecars as reference/baseline evidence for trained candidate selection, not as final retained actors.

## Sidecars
- `runtime/model_selection/card/soxlbacktest_qlib_walkforward_baseline_v1/candidate_metrics.csv`
- `runtime/model_selection/card/soxlbacktest_qlib_walkforward_baseline_v1/candidate_gate_summary.csv`
- `runtime/model_selection/card/soxlbacktest_qlib_walkforward_baseline_v1/candidate_daily.csv`
- `runtime/model_selection/card/soxlbacktest_qlib_walkforward_baseline_v1/candidate_trades.csv`
- `runtime/model_selection/card/soxlbacktest_qlib_walkforward_baseline_v1/summary.json`
- `runtime/model_selection/card/soxlbacktest_qlib_walkforward_baseline_v1/conclusion.md`
