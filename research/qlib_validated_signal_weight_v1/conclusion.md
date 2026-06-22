# 中文结论

- 做了什么：使用 qlib.init、D.calendar/D.features、DatasetH/DataHandlerLP、R.start、SignalRecord、SigAnaRecord、PortAnaRecord 和 WeightStrategyBase 子类 SignalWeightStrategy 跑 SOXL 验证集选择实验。
- 切分方式：train=2012-01-03 到 2017-12-29，valid=2018-01-02 到 2018-12-31 用于模型选择，test=2019-01-02 到 2026-06-09 只用于报告和 legacy 比较。
- 模型逻辑：Alpha158 叠加旧 SOXL 特征，候选模型 6 个，按 selection_objective_score 选中 linear_ic / qlib_ic_weighted_linear。
- 核心指标：qlib 策略 test 收益 6570.02%，legacy soxl_backtest rulebase test 收益 2148.77%，超额 4421.26%；feature_count=166，recorder_id=fa5df7f7aff44697924f641bc4589361。
- regime/window 差异：regime_metrics.csv、window_metrics.csv、backtest_window_metrics.csv 和 risk_analysis.csv 为 qlib recorder 派生证据；模型选择指标与 test 指标分别在 model_selection.csv 中记录。
- 是否晋升：研究层面通过 legacy soxl_backtest 收益门槛；但仍需滚动验证后才可晋升生产。
- 风险：模型由 2018 validation 段选择，测试收益只用于报告；仍可能对单一年份切分敏感。
- 下一步：若通过，补 rolling/walk-forward 固定参数复验并同步 production paper/sim；若未通过，继续扩展 SPEC 模型或切分方案。
