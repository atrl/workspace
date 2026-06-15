# release16_wf2020_trend_participation_gap_v1

- Decision: `release16_wf2020_trend_participation_gap_no_promotion`.
- Release16 `wf_2020/dd_control_checkpoint` remains below the relative-return floor: `-5.60pp` vs target `+30pp`, while MaxDD is `16.63%` under the user `<20%` gate.
- Counterfactual trend-floor pass counts: lagged `16`, same-day `3786`. Same-day rows are treated as leakage-risk diagnostics, not runnable evidence.
- Primary blocker: `official_replay_required_for_lagged_counterfactual`.
- Next: convert the visible trend-participation gap to a fold-local checkpoint/target objective and rerun full three-fold official replay; do not promote this card.

Artifacts:
- `runtime/model_selection/card/release16_wf2020_trend_participation_gap_v1/wf2020_gap_contribution.csv`
- `runtime/model_selection/card/release16_wf2020_trend_participation_gap_v1/wf2020_monthly_gap_contribution.csv`
- `runtime/model_selection/card/release16_wf2020_trend_participation_gap_v1/wf2020_drawdown_intervals.csv`
- `runtime/model_selection/card/release16_wf2020_trend_participation_gap_v1/trend_overlay_grid.csv`
- `runtime/model_selection/card/release16_wf2020_trend_participation_gap_v1/trend_overlay_top_daily.csv`
- `runtime/model_selection/card/release16_wf2020_trend_participation_gap_v1/feature_state_summary.csv`
- `runtime/model_selection/card/release16_wf2020_trend_participation_gap_v1/selection.json`
- `runtime/model_selection/card/release16_wf2020_trend_participation_gap_v1/summary.json`
- `runtime/model_selection/card/release16_wf2020_trend_participation_gap_v1/conclusion.md`