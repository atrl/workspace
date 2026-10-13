# 完整路径扣费优化结论

已执行 27 次严格成熟训练和 66 个 Qlib 原生账户，完整持仓、现金与逐笔费用独立对账。输入从2018真实分钟起点扩充；所有种子保留。

|窗口|模型|滑点bp|净收益%|最大回撤%|相对SOXX财富|
|---|---|---:|---:|---:|---:|
|development_2021|no_market|2|38.68|-10.53|0.9441|
|development_2021|daily|2|18.83|-15.39|0.8089|
|development_2021|minute|2|35.46|-14.11|0.9222|
|development_2021|soxx|2|46.89|-12.44|1.0000|
|development_2022|no_market|2|-34.35|-44.59|1.0251|
|development_2022|daily|2|-49.51|-61.61|0.7883|
|development_2022|minute|2|-39.95|-52.94|0.9377|
|development_2022|soxx|2|-35.96|-47.08|1.0000|
|reused_outer|no_market|2|223.57|-33.25|0.6634|
|reused_outer|daily|2|444.96|-40.90|1.1173|
|reused_outer|minute|2|130.96|-52.28|0.4735|
|reused_outer|soxx|2|387.73|-42.87|1.0000|
|reused_outer|anchor|2|279.67|-24.17|0.7784|

决定：`not_promoted_full_path_gates_failed`。门槛：{"net_excess_every_fold_and_cost": false, "drawdown_30pct_every_fold_and_cost": false, "cvar_no_worse_than_soxx": false, "rolling_win_fraction_at_least_70pct": false, "bootstrap_excess_and_minute_increment": false, "independent_forward_qualified": false}。年度/滚动与区块差异见 sidecars，所有费用和 seed 原生证据保留在 NAS Recorder。

风险：历史复用不构成新留出期，真实首收/成交/股息资格仍缺；整数梯度、训练片段初始现金与软回撤处罚都有限制。模型中先读已知价格/过去状态，再使用未来成交价结算，没有把当前成交价作为当前 NN 输入。仓位目标约束不保证持有漂移后的约束。

目标保持 active；本实验结束不等于稳定超额达成。下一步：qualify frozen independent forward identity if historical gates pass; otherwise close this fixed route and qualify independent release-time semiconductor capacity data。原模拟盘 release 不变。

训练/跨期差异：2023冻结的三只 minute 模型训练区块 log 超额均值约0.49–0.51，训练区块最大回撤约21.6–23.1%；复用外窗却相对SOXX财富仅0.474、监测回撤52.7%。日级模型复用外窗相对财富1.117，但开发期两个年份均落后，外窗监测回撤41.4%。这显示全路径目标和成本修正不能制造可泛化的信息；不以外窗收益挑日级模型晋升，不再搜索本固定路线的罚系数、epoch或seed。
