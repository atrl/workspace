# Model Retention Frontier

- Rows scanned: `93`.
- Completed handoffs: `63`.
- Recommended next source: `cap10_wf2020_bull_participation_diagnostic_v1`.
- Recommended next action: `Rerun checkpoint selection with backtest_profile bnh_return and relative_outperformance from the same qlib replay source before any new training.`
- Primary blocker: `missing_walkforward_evidence`.
- Retention gate: MaxDD `< 20.0` and YTD return floor `213.143462` versus SOXLBacktest reference `132.077978`.

This card ranks model-selection evidence for the next training run. It does not promote or retain actors.
