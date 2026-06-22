# 中文结论

- 做了什么：使用 qlib.init、D.calendar/D.features、DatasetH/DataHandlerLP、R.start、SignalRecord、SigAnaRecord、PortAnaRecord 和 WeightStrategyBase 子类 SignalWeightStrategy 跑 2012-2019 训练、2020-2026 测试的 SOXL 搜索卡。
- 模型逻辑：Alpha158 叠加旧 SOXL 特征，候选模型 6 个，选中 lstm_mse / qlib_window_sequence。
- 核心指标：qlib 策略收益 2754.37%，legacy soxl_backtest rulebase 收益 1156.53%，超额 1597.84%；feature_count=166，recorder_id=70519d460cb3428491669619d4a1f599。
- regime/window 差异：regime_metrics.csv、window_metrics.csv、backtest_window_metrics.csv 和 risk_analysis.csv 为 qlib recorder 派生证据；测试窗口覆盖 2020-01-02 到 2026-06-09。
- 是否晋升：研究层面通过 legacy soxl_backtest 收益门槛；暂不直接晋升生产，因为当前模型选择仍是搜索流程，需要独立验证/滚动验证后再固定参数。
- 风险：单标的 timing 对阈值和样本切分敏感；当前搜索卡证明 qlib-native 路径和候选模型可超过 legacy，但不等于无未来函数的生产策略。
- 下一步：补 walk-forward/valid split，避免测试集选择偏差，再把通过的固定策略写入 production paper/sim notebook。
