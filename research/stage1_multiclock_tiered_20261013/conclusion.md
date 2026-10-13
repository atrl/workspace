# 多时点完整Stage1费用检验

冻结分钟模型与预测复用，新增拟合0次；18个实际Tiered/0-2-5bp账户通过Qlib逐笔费用、现金、股数及净值对账。原首小时主候选保留，其他两组不事后重选。

|模型|滑点bp|净收益%|15分钟监测回撤%|相对SOXX财富|
|---|---:|---:|---:|---:|
|current|0|67.20|-30.38|0.3591|
|neutral_first_hour|0|201.15|-28.67|0.6467|
|first_hour|0|177.00|-31.82|0.5949|
|full_day|0|103.84|-33.59|0.4378|
|full_day_context|0|113.93|-32.07|0.4594|
|soxx|0|365.66|-43.40|1.0000|
|current|2|32.17|-32.31|0.2838|
|neutral_first_hour|2|228.11|-29.25|0.7046|
|first_hour|2|172.03|-32.51|0.5842|
|full_day|2|120.48|-32.37|0.4735|
|full_day_context|2|167.68|-30.58|0.5749|
|soxx|2|365.64|-43.41|1.0000|
|current|5|113.00|-29.13|0.4575|
|neutral_first_hour|5|196.31|-30.70|0.6364|
|first_hour|5|188.25|-30.81|0.6191|
|full_day|5|90.29|-32.33|0.4087|
|full_day_context|5|63.35|-33.23|0.3508|
|soxx|5|365.61|-43.41|1.0000|

门槛：{"net_soxx_excess_all_costs": false, "monitored_drawdown_within_30": false, "cvar_no_worse_than_soxx": true, "minute_increment_all_costs": false, "rolling_win_70pct": false, "positive_years_at_least_three_all_costs": false, "paired_bootstrap_excess_and_increment": false, "qualified_independent_forward": false}。原开盘动作、日级风险状态与恢复许可保留，盘中经济选择及中性对照共用机会集。原始生产时钟与本轮延迟close时钟不同，不能把跨时钟收益差称作alpha。

历史仍为复用诊断，真实首收/成交/股息资格未取得；逐笔费用为当前费表投影，滑点为不利执行准备金。监测回撤使用15分钟报价，不是每秒保证。结论不晋升，目标保持active；本固定组合结束后不修改源模型、窗口、阈值或重选最佳特征组。

归因：首小时分钟数值在0/2/5bp三种情景均落后于同机会集的数值中性对照。改善原控制器的部分来自恢复权限/时段结构，不能认作分钟预测alpha。原生持仓默认近似清仓会丢失微小正余额，已复用QlibContinuousPosition，费用/价格不改、阈值不放宽；当前整段账户对账相对误差约4.4e-16。

只读Futu探测保存了盘前报价和本地接收时间，但未获得常规时段执行校准资格；原始探测仅在NAS Recorder保存，未下单。稳定超额目标保持active。
