# 直接净值缓冲目标检验结论

固定0.70峰值参考线、4倍缓冲，0次拟合、18个原生账户。目标暴露与恒定1.2beta、SOXX比较；长期连续与2023现金起点都保留。

|窗口|策略|滑点bp|净收益%|监测回撤%|相对SOXX财富|
|---|---|---:|---:|---:|---:|
|continuous_2018|cushion_target|2|196.55|-29.01|0.2778|
|continuous_2018|constant_beta|2|1126.77|-55.35|1.1494|
|continuous_2018|soxx|2|967.31|-47.41|1.0000|
|cash_start_2023|cushion_target|2|198.15|-28.53|0.6113|
|cash_start_2023|constant_beta|2|467.66|-51.68|1.1639|
|cash_start_2023|soxx|2|387.73|-43.39|1.0000|

决定：`not_promoted_cushion_target_gates_failed`。门槛：{"net_excess_all_costs_and_starts": false, "drawdown_30pct": true, "cvar_no_worse_than_soxx": true, "rolling_positive_70pct": false, "bootstrap_positive": false, "qualified_forward": false}。年度和滚动差异完整保留。

这是直接分配预算的无预测控制，不是旧预测仓位的缩小上限。离散跳空、交易费、ETF相对3倍SOXX跟踪偏差与整数股数都可侵蚀参考线，不能声称未来回撤保证。原始价格/分红现金流与真实成交边界继承来源；历史复用不能代替独立前向。

失败即结束该固定系数/映射，不另挑倍率或起点。原模拟盘未切换，目标继续active。
