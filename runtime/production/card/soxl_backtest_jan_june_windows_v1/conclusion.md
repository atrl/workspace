## Conclusion

- Purpose: this notebook packages the SOXL backtest data for the two requested 2026 windows, using rebased window-level NAV so the January and June slices are directly usable.
- Model comparison data source: `runtime/production/card/current_model_ytd_june_comparison_v1/current_model_daily.csv` for the current production composer, plus `runtime/production/card/current_model_ytd_june_comparison_v1/all_model_daily_nav.csv` for dashboard replay peers.
- Date rule: model comparisons stop at `2026-06-09` because dashboard replays are safe only through that date; market-only SOXL data is fresh through `2026-06-10`.
- January/YTD common window: requested `2026-01-01`, actual first trading row `2026-01-02`, end `2026-06-09`, rows `109`. SOXL buy-and-hold returned `326.93%`; current production model returned `3.25%`; excess `-323.68pp`. Model max drawdown `21.09%` vs SOXL `43.47%`.
- June common window: requested `2026-06-01`, actual first trading row `2026-06-01`, end `2026-06-09`, rows `7`. SOXL buy-and-hold returned `-11.17%`; current production model returned `0.00%`; excess `11.17pp`. Model max drawdown `0.00%` vs SOXL `34.93%`.
- Legacy `soxl_backtest` YTD fresh-start run: `2026-01-02` to `2026-06-09`, rulebase returned `132.08%` vs SOXL buy-and-hold `326.93%`, excess `-194.85pp`, trades `3`, end position `15.91%`.
- Legacy `soxl_backtest` June fresh-start run: `2026-06-01` to `2026-06-09`, rulebase returned `3.14%` vs SOXL buy-and-hold `-11.17%`, excess `14.30pp`, trades `1`, end position `29.94%`. This June run starts a fresh rulebase account on June 1, so use it as a standalone scenario rather than a continuation of the January run.
- All-model YTD leader: `smoke` returned `224.36%` with max drawdown `30.35%`. YTD laggard: `v19reg_v22_budget_usage_bull_guard_v1` returned `3.25%`.
- All-model June leader: `v19reg_v22_budget_usage_bull_guard_v1` returned `0.00%` with max drawdown `0.00%`. June laggard: `state_conditioned_drawdown_rebound_reward_train_v1` returned `-9.80%`.
- Market-only latest through `2026-06-10`: SOXL buy-and-hold returned `282.41%` YTD and `-20.43%` for June. Do not compare these directly to model rows because the model replay common end is `2026-06-09`.
- Primary reusable outputs: `current_model_ytd_2026_common_daily.csv`, `current_model_june_2026_common_daily.csv`, `current_model_window_metrics.csv`, `all_model_window_metrics.csv`, `legacy_soxl_backtest_window_metrics.csv`, `legacy_soxl_backtest_window_daily.csv`, and `legacy_soxl_backtest_trades.csv` in this card directory.

