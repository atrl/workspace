# 结论

- 执行内容：只用SOXL收盘前可知的价格路径，区分趋势许可、快速湍流和持续高位震荡；不用预测模型。
- 最新状态（2026-07-02）：252日区间位置=0.569，20日有向效率=-0.159，波动率排名=1.000，趋势许可=0.000，快速湍流风险=0.000，持续震荡年龄=0。
- fast/development: score样本=2935, 事件日=58, episode=24, AUC=0.504, Q5-Q1=2.0%, price分层条件增量=2.2%。
- fast/validation: score样本=496, 事件日=5, episode=2, AUC=0.503, Q5-Q1=2.7%, price分层条件增量=-0.5%。
- fast/test: score样本=369, 事件日=31, episode=5, AUC=0.504, Q5-Q1=0.2%, price分层条件增量=-3.4%。
- structural/development: score样本=2920, 事件日=125, episode=23, AUC=0.495, Q5-Q1=-2.2%, price分层条件增量=2.4%。
- structural/validation: score样本=481, 事件日=7, episode=1, AUC=0.542, Q5-Q1=28.0%, price分层条件增量=20.0%。
- structural/test: score样本=354, 事件日=9, episode=5, AUC=0.545, Q5-Q1=17.2%, price分层条件增量=8.9%。
- fast: diagnostic_rare_event_support
- structural: diagnostic_rare_event_support
- 晋升决定：这是因子诊断卡，不是交易策略晋升；验证与测试必须同向、事件支持充分，并在price-only分层内都有正增量才保留。
- 风险：日线无法观察盘中路径；5日和20日标签重叠；末端样本已按窗口purge；事件episode比逐日样本更接近独立证据。
- 下一步：只把保留因子接入分层降仓回测；失败因子停止调模型，改查盘中/事件数据。
