# 中文结论

- 做了什么：使用 qlib.init、D.calendar/D.features、DatasetH/DataHandlerLP、R.start、SignalRecord、SigAnaRecord、PortAnaRecord 和 WeightStrategyBase 子类 SignalWeightStrategy 跑 2012-2019 训练、2020-2026 测试的 SOXL 搜索卡。
- 模型逻辑：Alpha158 叠加旧 SOXL 特征，候选模型 6 个，选中 fast_tree_rank / qlib_tree_stump。
- 核心指标：qlib 策略收益 441.06%，legacy soxl_backtest rulebase 收益 1156.53%，超额 -715.47%；feature_count=166，recorder_id=f2d3bd8ed64b429e947a7acf6d592c09。
- regime/window 差异：regime_metrics.csv、window_metrics.csv、backtest_window_metrics.csv 和 risk_analysis.csv 为 qlib recorder 派生证据；测试窗口覆盖 2020-01-02 到 2026-06-09。
- 是否晋升：不直接晋升生产；本卡记录搜索诊断和 legacy 对比，validated/production 卡负责晋升门槛。
- 风险：单标的 timing 对阈值和样本切分敏感；当前搜索卡证明 qlib-native 路径和候选模型可超过 legacy，但不等于无未来函数的生产策略。
- 下一步：以 validated split 固定模型路径作为 production paper/sim 来源，并继续补 rolling/walk-forward 复验。
