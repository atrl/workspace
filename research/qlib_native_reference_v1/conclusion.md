# 中文结论

- 做了什么：通过 qlib.init、D.features、DatasetH/DataHandlerLP、R.start、SignalRecord、SigAnaRecord 和 PortAnaRecord 跑通 SOXL qlib-native reference，并在 qlib Dataset 上完成多模型选择。
- 模型逻辑：候选模型 6 个，选中 linear_ic / qlib_ic_weighted_linear，结果见 model_selection.csv、all_model_predictions.csv 和 selected_model.json。
- 核心指标：已生成 qlib recorder、pred.pkl、label.pkl、sig_analysis、portfolio_analysis、risk_analysis.csv、backtest_window_metrics.csv、backtest_walk_forward_metrics.csv、backtest_benchmark_comparison.csv 和 model_comparison.csv。
- regime 差异：regime_metrics.csv 基于 qlib recorder 和 qlib Data API 的 VIX、趋势、回撤、波动、成交量和市场状态分层，短窗口仅用于验收。
- 是否晋升：不晋升；这是 qlib 原生工作流验收卡。
- 风险：样本窗口短，IC 可能为 NaN；不能作为交易信号。
- 下一步：在 production paper/sim notebook 复用 recorder id c134e7b4476f411283ea8ab7b0820e90。
