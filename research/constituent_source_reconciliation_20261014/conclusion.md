# 补充来源衔接：纠正遗漏，不重开失败实验

本轮0拟合、0回测、0未来收益筛选。此前只核对主成分provider，遗漏了已存在的WOLF补充来源和已完成的广度实验，导致把工作区总数据误说成仍缺WOLF历史。本卡更正这个判断；主provider单独覆盖统计仍准确，不能扩大为全部来源可用性。

已存在的data/snapshots/semis_wolf_recovery_20241126_v1保存2019-06-03至2024-11-26共1383日Futu NONE历史、2019 CREE与2024 WOLF两份SEC价格身份锚点，以及Qlib价格库。原清单记载取回时间2026-10-05。所有原始文件、清单和close.bin哈希已核对，未拼入2025重组新证券。当前Yahoo对2023 WOLF查询失败不代表这个既有来源不存在。

新增data/constituent_price_supplements.json显式登记WOLF/CREE旧证券和上一轮BRKS更名候选来源。读取器逐文件校验哈希、通过Qlib取数，只填原缺失，重叠有冲突时报错，绝不覆盖已有价格。CREE/WOLF别名表示持有旧披露篮子期间的发行人更名连续性，不声称当时官方指数名单。

在相同1657个有披露篮子的交易日上，20日窗口完整性从398日恢复到1027日；2023年250日、2024年252日均完整，2023Q4为63/63。2022年完整150日。2019–2021仍受其他退市成员影响；2026价格仍只到7月2日，不能解释整个2026Q3。旧provider与模型未覆盖。

研究去重：research/semiconductor_breadth_20261005已使用WOLF补充来源，实际特征期2022-05-27至2026-07-02。该固定实验测试上涨比例、等权减披露权重收益、横截面标准差、60日平均绝对相关性，6次Ridge拟合；2024–2026共627行评价，ΔIC=-0.098808，MSE增量为负，结论end_route_no_further_search。不能因补数重新包装成未做过的新广度实验。本轮仅引用其既有证据，没有重新拟合。

这也不等于所有成分级信息均被证伪；新低扩散或下跌集中度若另行研究，必须说明相对旧四特征的实质机制差异，并遵守当前不拟合约束。不能仅改名称或周期重开旧路线。

决定：supplement_sources_reconciled_closed_breadth_route_not_reopened。新增的是来源可发现性、显式读取和纠错，非新价格发现或策略超额。稳定净超额未完成。

[覆盖对照](coverage_comparison.csv) · [补充来源读回](supplement_readback.csv) · [来源注册](../../data/constituent_price_supplements.json) · [已关闭旧广度实验](../semiconductor_breadth_20261005/conclusion.md)

Recorder `50ec34c660a545aeab8e33ff7964c079`。
