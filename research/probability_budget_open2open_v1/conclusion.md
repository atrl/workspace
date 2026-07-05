# 结论

- what_ran: 执行 probability_budget_open2open_v1，用 qlib `D.features`、Alpha158、SOXX 成分股特征和 LGB/XGB consensus，验证纯 probability budget 仓位。
- key_metrics: selected_return=9.8608%，maxDD=-1.8830%，max_daily_loss=-1.0992%，CVaR5=-0.8835%。
- walk_forward: folds=4，risk_all_pass=True，return_all_pass=True，min_return=3.2041%。
- architecture: 不使用 base_constrained_ref 作为 base，不使用 base_plus_probability，不使用 risk_gate_col；base_constrained_ref 只作为 benchmark，对比差距=-268.2532%。
- latest_signal: as_of=2026-07-02，action=HOLD，target_position=5.1998%。
- promotion_decision: candidate_for_walk_forward_validated_probability_budget_under_base_constrained_ref。
- risks: 该卡解决 walk-forward 风险/收益门，但收益低于 base；不构成买卖建议，实盘前需 paper/sim。
- next_action: 若目标包含跑赢 base，需要在不破坏 walk-forward 门的前提下重做收益目标或引入独立 regime sleeve。recorder=77501a073c524aec89765eacf916c290。
