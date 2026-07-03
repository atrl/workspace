# 结论

- what_ran: 执行 risk_hardened_open2open_v2，用 qlib `D.features`/Alpha158、SOXX 成分股 qlib store、LGB/XGB 和 open-to-open return 重新训练 risk-hardened position selector。
- key_metrics: selected_return=183.4342%，maxDD=-24.9027%，max_daily_loss=-14.0086%，CVaR5=-7.9389%。
- base_relative: base_return=278.1140%，return_gap=-94.6798%，risk_fallback_pass=True。
- regime/window: factor_ablation 最优组为 base_cogalpha_soxx；training_window 最优为 expanding_2019。
- execution: intraday_status=ok，coverage_days=123，open/VWAP1/VWAP5/VWAP15 stress 已写入 sidecar。
- latest_signal: as_of=2026-07-02，action=HOLD，target_position=0.0500%。
- promotion_decision: not_promoted_under_base_constrained_ref。
- risks: SOXX holdings 仍有 survivorship bias；分钟验证是执行压力测试，不是实盘成交保证；不构成买卖建议。
- next_action: 若未晋升，优先补 PIT SOXX holdings 和更长 1m/order-flow 数据，再放宽或重训风险预算模型。recorder=5f735be3c21c4382b833d9f12e1c571a。
