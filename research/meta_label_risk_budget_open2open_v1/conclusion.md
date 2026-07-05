# 结论

- what_ran: 执行 meta_label_risk_budget_open2open_v1，用 qlib `D.features`/Alpha158、SOXX 成分股、15m lagged intraday features、LGB/XGB dual meta-label，验证 base opportunity + risk budget 架构。
- key_metrics: selected_return=1.6038%，maxDD=-2.0065%，max_daily_loss=-1.1582%，CVaR5=-0.7796%。
- base_relative: base_return=278.1140%，return_gap=-276.5102%，risk_fallback_pass=True。
- walk_forward: folds=4，all_pass=False，median_return=9.9873%，worst_daily_loss=-5.2009%。
- execution: intraday_status=ok，coverage_days=123，15m 用作 lagged execution cap，1m cache 用作 open/VWAP stress。
- latest_signal: as_of=2026-07-02，action=score_next_open_target_position，target_position=0.0000%，tail_prob=0.2399。
- promotion_decision: not_promoted_failed_walk_forward_risk_gates。
- decision: 旧的 base_constrained_ref + ML probability boost + risk gate 不再作为主方案；本卡把它降级为 opportunity source，并用 tail/vol/drawdown/intraday 预算决定仓位。
- risks: 仍需 PIT holdings、更长分钟/订单级执行样本和实盘 paper 验证；不构成买卖建议。
- next_action: 若未晋升，先修 walk-forward/tail gate 失败项，再考虑 regime specialist 或盘中独立执行模型。recorder=4fbc224c4fb84ba2832ffdae2796048f。
