# state_conditioned_overlay_gap_to_20_v1

- Decision: `simple_position_throttles_do_not_satisfy_both_drawdown_and_return_gates`.
- Original `state_conditioned_drawdown_rebound_reward_train_v1` path has worst WF MaxDD `29.15%` and YTD return `196.37%`.
- WF-drawdown-pass variants: `static_cap_0_20, risk_floor_plus_0_10, risk_floor_plus_0_00`.
- YTD SOXLBacktest-return-pass variants: `original_reported, dd18_cap_0_30, dd15_cap_0_20, dd15_cap_0_30`.
- No tested simple cap, risk-floor throttle, or drawdown-trigger cap satisfies both WF drawdown and return gates.
- Use this as branch evidence: a post-hoc simple cap is not enough; next work should change training/selection so exposure is lower before the bad WF drawdown intervals while preserving current YTD upside.

## Sidecars
- `runtime/model_selection/card/state_conditioned_overlay_gap_to_20_v1/overlay_window_metrics.csv`
- `runtime/model_selection/card/state_conditioned_overlay_gap_to_20_v1/overlay_gate_summary.csv`
- `runtime/model_selection/card/state_conditioned_overlay_gap_to_20_v1/drawdown_intervals.csv`
- `runtime/model_selection/card/state_conditioned_overlay_gap_to_20_v1/variant_daily.csv`
- `runtime/model_selection/card/state_conditioned_overlay_gap_to_20_v1/selection.json`
- `runtime/model_selection/card/state_conditioned_overlay_gap_to_20_v1/summary.json`
- `runtime/model_selection/card/state_conditioned_overlay_gap_to_20_v1/conclusion.md`
