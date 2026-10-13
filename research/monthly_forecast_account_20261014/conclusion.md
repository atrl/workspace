# 月度更新完整账户检验

0新增拟合，24个新增原生账户、9个已验证对照复用。初始10万美元，IBKR Pro Tiered，分别运行0/2/5bp。

|窗口|策略|净收益%|监测回撤%|相对SOXX财富|
|---|---|---:|---:|---:|
|continuous_2021|frozen|-0.56|-28.80|0.2168|
|continuous_2021|monthly|0.06|-28.64|0.2182|
|continuous_2021|cushion_target|73.56|-29.01|0.3785|
|continuous_2021|soxx|358.59|-47.29|1.0000|
|continuous_2021|matched_mean_beta|87.96|-19.47|0.4099|
|cash_start_2023|frozen|43.51|-27.74|0.2942|
|cash_start_2023|monthly|12.03|-28.13|0.2297|
|cash_start_2023|cushion_target|198.15|-28.53|0.6113|
|cash_start_2023|soxx|387.73|-43.39|1.0000|
|cash_start_2023|matched_mean_beta|101.42|-19.64|0.4130|
|cash_start_2023|current_anchor|279.67|-27.07|0.7784|

决定：not_promoted_monthly_account_gates_failed。门槛：{"net_excess_all_costs_starts": false, "drawdown_30pct": true, "cvar_no_worse_soxx": true, "positive_rolling_70pct": false, "positive_years_70pct": false, "positive_bootstrap": false, "matched_controls": false, "qualified_forward": false}。

冻结与月度模型使用同一目标映射、风险预算、费用估算和执行时点。HOLD确实不下单；预估成本按已知价格和第一费率档，实际账户按当月累计股数结算。恒定平均暴露对照为事后诊断，不可部署；current_anchor仅为同执行时钟的历史策略参考，不等于实际模拟盘成交。

回撤依据日度原生收益与已知/收盘净值点，不宣称全分钟最大回撤或未来保证；价格/分红现金流及真实成交资格继承来源限制。校准改善不能替代收益和风险门。原生产SIM未改，目标未完成；失败则封存固定映射与更新规则，不改频率、风险倍率或成本假设寻找胜出。

2bp下，2023现金起点月度候选平均现金66.57%，收益12.03%；同平均暴露对照收益101.42%、监测回撤19.64%，均优于候选。零滑点时月度候选收益28.45%，仍远低于SOXX，因此费用不是全部差距。各滑点情景独立决策及结算，路径会变化，收益不必随滑点单调。

本结果只拒绝该月度Ridge与固定风险/权重映射组合，不能推断所有重训机制无效。当前原策略参考表现更好，不替换生产模型。下一步需要关注原策略的决策增量和现金参与度，不能继续把较低MSE当作收益改善。固定方案停止，不在本结果上调频率、风险倍数或阈值。
