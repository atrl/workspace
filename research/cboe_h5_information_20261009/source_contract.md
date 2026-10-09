# Cboe 三指标的来源与时点合同

本卡检验 BASE25 与 BASE25+CBOE3 的 H5 收益预测和固定分配器增量。三项特征、共同训练起点、年度 Ridge10、HOLD5、成本与门禁全部由 [plan.json](plan.json) 冻结；本文不增加参数或记录未经执行的收益结论。

## 双数据截止与原始证据

官方完整下载保留到 `store_as_of=2026-10-08`，新 store 对其全部原始历史建档，不预先截短文件。行情和研究评价保持 `market_as_of=2026-09-25`；这两个截止分别记录，不把后续已下载的 Cboe 观察混入截至 09-25 的信号、训练或评价。读取 Qlib 后按已审计的 benchmark 原生 calendar 选择研究窗口，并报告未用于研究的记录数量。store builder 仍须在自身 `as_of` 上执行未来行拒绝检查。

| 指标 | 官方原始文件 | 当前文件首值 | 发布边界与限制 |
| --- | --- | --- | --- |
| VIX9D | [VIX9D_History.csv](https://cdn-api.cboe.com/api/global/us_indices/daily_prices/VIX9D_History.csv) | 2011-01-04 | [SEC 原始文件](https://www.sec.gov/rules/sro/cboe/2014/34-71764.pdf)确认 VXST 于 2013-10-01 开始发布；2011 起点不代表当时可见 |
| VIX3M | [VIX3M_History.csv](https://cdn-api.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv) | 2009-09-18 | [现行 Cboe 方法](https://cdn.cboe.com/api/global/us_indices/governance/Volatility_Index_Methodology_Selected_SPX_Target_Expected_Volatility_Term_Indices.pdf)列 launch 为 2013-10；不能仅因旧 VXV 更早存在就认证当前早期回溯序列 |
| VVIX | [VVIX_History.csv](https://cdn-api.cboe.com/api/global/us_indices/daily_prices/VVIX_History.csv) | 2006-03-06 | [Cboe 首发稿](https://www.prnewswire.com/news-releases/cboe-introduces-vix-of-vix-benchmark-index-provides-a-gauge-for-measuring-volatility-of-the-vix-index-itself-142607166.html)说明 2012-03-14 开始发布，但[现行方法](https://cdn.cboe.com/api/global/us_indices/governance/Volatility_Index_Methodology_Selected_Broad_Based_Index_Equity_and_ETF_Volatility_Indices.pdf)列 2014-02-01；冲突未解决，研究采用较晚边界 |

三指标的共同保守 source 起点是 **2014-02-03**，匹配训练信号从下一原生会话 **2014-02-04** 开始。完整下载的 SHA256、原下载 URL、获取时间、日期范围和旧 raw 文件 hash 保存在 intake manifest；完整性以新 provider 原生读回审计为准。

SKEW 在任何模型拟合前被排除：官方文件缺少开发期原生会话 2017-09-14、2018-12-03、2019-07-05，另缺 2024-11-29。旧 Taleb builder 的前向填充不能补足真实观察。本卡不填值、不删除评价行，不把“有非空字段”当作来源合格。

## 特征与可用时刻

固定家族为上一原生会话的 `log(VIX3M/100)`、`log(VIX9D/VIX3M)`、`log(VVIX/100)`。它们分别表示较长期的隐含波动水平、近远端期限结构、波动率自身的不确定性；这只是三项原始观察的低维表达，不增加独立数据，也不预设收益方向。训练折内做标准化，避免额外滚动窗口减少样本；不因结果不佳追加 z-score 或阈值搜索。

现行 Cboe 方法给出的三指标常规发布时段均为 09:31–16:15 ET。本轮进一步采用 source session 17:00 ET 可用的保守假设：signal session 16:15 ET 决策只读取**上一原生会话**值，再于下一会话开盘执行。lag 必须先在完整 benchmark calendar 上对齐，再按会话移动；不能在稀疏 raw 行上 shift 后 inner join。逐行保留 `source_date`、可用时间和 signal date，并验证严格先后关系。

可用时刻是假设，完整历史文件也不是各历史日期当时归档的版本。首发前回溯值被屏蔽、lag1 与旧快照数值一致，都不能升级为严格 historical-vintage/PIT 认证。

## 增量比较与 fail-closed 检查

- BASE 和 full 使用同一匹配起点、完整成熟行、H5 标签、年度边界、训练内 scaler、正则、分配器、费用及执行日历。不能将 2011 起训练的旧 BASE 与较短训练的 full 直接解释为信息增量。
- 原日期唯一，三项值有限且为正；研究原生会话出现缺值或重复时拒绝继续，不前向填充、不取最后重复行、不删除缺失日。读取字段使用明确名单，不把旧 provider 的未来标签作为特征。
- 训练标签必须在拟合截止前成熟且共同样本不少于计划门槛；预测诊断另设显式 aware `evaluation_cutoff`，只消费 `label_available_at < cutoff`。末端未成熟标签后缀保留审计，内部缺值直接失败。
- MSE、IC、校准与误差偏差/方差分解描述固定预测。校准系数不用于重写预测；不翻转符号、不挑年份。paired bootstrap 若使用，信息诊断按每窗口两个资产乘两个比较基准明确计为四项，不与组合门的候选数混淆，也不修复此前历史搜索偏差。

旧 `taleb_fragility` 已实现 VIX9D/VIX3M 的滚动 z、VVIX/VIX、SKEW 和短期风险溢价；旧角色分离和 dynamic 工作流还纳入变化率、原始水平与比率。因此本卡是来源、时点和匹配样本更清晰的增量复核，不能称作从未研究的新因子。全部评价历史仍已复用；来源合格、预测改善、组合稳定超额和前向晋升必须分别判定。
