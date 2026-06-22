# 中文结论

- 做了什么：固定使用 research/qlib_validated_signal_weight_v1 验证出的 linear_ic / qlib_ic_weighted_linear，执行 qlib Dataset、R.start、SignalRecord、SigAnaRecord、PortAnaRecord，并生成 paper/sim target-position intent。
- 固定参数：train=2012-01-03 到 2017-12-29，valid=2018-01-02 到 2018-12-31，test=2019-01-02 到 2026-06-09；model_selection_metric=objective_score；fixed_model_name=linear_ic。
- 核心指标：qlib paper/sim 策略 test 收益 6570.02%，legacy soxl_backtest rulebase test 收益 2148.77%，超额 4421.26%；recorder_id=43fa6114dcb9420ea1e42b8d48014e01。
- regime/window 差异：regime_metrics.csv、window_metrics.csv、backtest_window_metrics.csv 和 risk_analysis.csv 为 qlib recorder 派生证据。
- 是否晋升：晋升到 production paper/sim 固定策略；仍不做真实 broker 下单。
- 风险：该卡固定的是验证集选择结果，后续仍需 rolling/walk-forward 复验和 schedule 接入。
- 下一步：让 schedule/dashboard 仅读取 recorder 登记的 production_summary 和 sidecars。
