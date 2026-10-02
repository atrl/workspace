# 冻结数据基准审计

本审计只核验数据，不训练模型、不运行新组合、不修复旧provider。状态：`needs_data_qualification_before_new_basis_rebuild`。

来源 Qlib recorder：`b94059c2e0754b0ab102e4f62534eefb`，as_of2026-09-25。专属审计 recorder：`f77655b5e36e4771a7ead1343c287e37`。后续重执行从该审计recorder读取冻结输入并核对哈希，不把主目录中持续更新的行情当成冻结数据。

## 已核验事实

- 通过Qlib原生接口读取8324条标的交易日记录，区间2010-03-11至2026-09-25；价格源和成交量源日期逐项一致。
- 执行provider字段为OHLC，没有volume；primary的volume完整且正值。原生执行O/C与原recorder的冻结执行价格最大误差为0。
- 保存的soxx_ret1与primary收盘收益重建误差为0.0。
- 3956条主特征记录中，2974条收盘收益与执行源差异超过1e-7；最大差异80.4532bp。
- 2014-12-24，primary收盘从31.5766678到31.3700008，收益-0.6545%；执行源从27.8728161到27.9146347，收益0.1500%。
- OHLC物理检查发现172条low高于open/close、190条high低于open/close。按原生float32数值转float64计算相对幅度：>1e-6为190条，>1e-5为1条。

唯一超过1e-5的物理异常为SOXL，2023-06-05：执行源O=21.3147888、H=21.0109921、L=20.2465992、C=20.5503960，开盘高于最高价1.4459%。同日primary原报价O=21.7500000、H=21.7500000、L=20.6599998、C=20.9699993。冻结Futu旁证及其原文件哈希也已归档；未归档原响应的外部复查只标记为审阅说明，不当作可重放机器证据。

## 来源解释与未解决事项

原代码的Yahoo quote OHLC路径没有应用adjclose/close分红调整；执行provider则显式应用该比例。当前top-level manifest不能证明primary全历史每行都来自同一调整方式。两条SOXX价格流的基准不同，不是同一个SOXX基金在同日跟踪两个指数。

旧Yahoo/Nasdaq拼接按OHLC字段分别估计缩放比例，可能造成细小蜡烛边界破坏。单个2023-06-05 high错误明显更大，初始原始payload尚未完整找回，不能靠high=max(open,close)等补丁捏造价格。旧source代码文本、provider manifest及原执行源manifest随冻结输入归档。

## 同一因子调整PRIMARY的可行性检查

PRIMARY自身物理检查：low高于open/close为0条，high低于open/close为0条，最大相对违反0。

只在内存中构造候选：对PRIMARY的O/H/L/C使用同一正因子execution_close/primary_close，volume完全不变；没有修改执行O/C或重新计算旧P/L。候选收盘与执行收盘最大浮点差为5.684341886e-14。

调整后开盘与冻结执行开盘逐项比较：[{"instrument": "SOXL", "rows": 4162, "maximum_absolute_price_difference": 0.24093352642916344, "maximum_absolute_relative_difference": 0.02663912674976654, "worst_date": "2017-10-30", "above_relative_threshold_counts": {"1e-07": 1034, "1e-06": 894, "1e-05": 248, "0.0001": 84, "0.001": 4}}, {"instrument": "SOXX", "rows": 4162, "maximum_absolute_price_difference": 0.5656698781834635, "maximum_absolute_relative_difference": 0.00931496245429353, "worst_date": "2018-03-13", "above_relative_threshold_counts": {"1e-07": 1394, "1e-06": 271, "1e-05": 80, "0.0001": 6, "0.001": 3}}]。因此相同收盘基准不自动证明开盘也相同；是否仍有两个来源通道、差异是否可接受及来源需要单独审定。候选整表和开盘差异都归档为数据审计产物，不代表已经批准新的canonical数据。

## 对历史研究的影响边界

原阶段组合结算读取执行O/C，没有直接消费这条错误high；该日期也属于2023外层诊断。原特征使用另外的primary OHLC。Phase3 prior_h5只依据执行收益标签和开盘协方差，不能把其七个开发年份落后全部归咎于特征复权差异。

但特征和标签的价格基准未统一，学习模型及PIT辅助信息的经济解释需在明确重建后复查。影响方向和P/L变化本次没有测量。原数据上的数值可复现，不等于外部行情已验证、数据基准已经合格或存在稳定超额。

## 官方指数历史：另一个应分开的因素

SOXX前身于2010-10-15从S&P North American Technology-Semiconductors改为PHLX，[发行人文件](https://www.sec.gov/Archives/edgar/data/1100663/000119312510241547/d497.htm)有记录。SOXX于2021-06-21改为ICE；ICE于2023-11-03更名NYSE，[iShares招募书](https://www.ishares.com/us/literature/prospectus/p-ishares-phlx-semiconductor-etf-3-31.pdf)说明历史切换。SOXL则于2021-08-25开盘生效，[NYSE监管公告](https://www.nyse.com/publicdocs/nyse/markets/nyse-arca/rule-interpretations/2021/NYSE%20Arca%20Equities%20RB-21-075.pdf)确认。

因此2021-06-21至8/24存在指数错位，名义3倍不是ETF收益恒等式；这也不能单独解释全部年份。现有ETF实际收益上没有再扣一笔虚构的融资成本。

## 下一步

先按来源和物理一致性确定新、独立版本的统一OHLCV基准，必要的分红调整必须对同一bar四个价格用相同因子，并明确volume来源/拆股口径。若无法恢复原高低价，不将执行源OHLC自动当作干净替代品。重建选择不得依据哪个旧/新回测更漂亮。

数据验收后，再以相同Ridge10、相同特征定义和成熟时序重建OOF头；旧模型、provider和证据保持原身份。现金收益仍需独立的真实可得来源合同，不能临时填入收益率使候选过门。稳定超额目标尚未达成。

完整原生frames和明细位于本审计recorder的`data_quality/`；卡根只保留本说明和data_quality_summary.json。
