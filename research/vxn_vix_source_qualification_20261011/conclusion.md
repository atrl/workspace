# VXN/VIX 首次发布来源资格

已实际取得匿名官方 Initial Release Only ZIP，并通过 Qlib 原生信息 Store 读回；未训练、未回测、不晋升。

原始 VXN3243/VIX4072条，无重复；完整3956个价格交易日保留，3126日有可用信息、830日未知。配对必须同观察日，公开日期保守映射到下一真实交易日09:31。2018-10-08真实市场日存在提前发布日期标记，已隔离，未编造原因或改写原记录。缺值不被零或旧值填补。

2016外层的内层2015截止前只有177个已知来源日（尚未扣除标签成熟尾部），不能满足252行；必须明确SOXX退回。后续比较的新信息模型与价格对照必须共享同一来源有效样本，不能将训练跨度差异归因于信息增量。

下一步预登记一个固定log(VXN/VIX)信息增量，保持费用/回撤/年度/滚动/置信区间风险门。来源合格不代表alpha，H20及已失败广义CBOE路线保持关闭。

官方来源：[ALFRED VXN](https://alfred.stlouisfed.org/series?seid=VXNCLS)、[首次发布格式](https://alfred.stlouisfed.org/help/downloaddata)、[CBOE 方法](https://cdn.cboe.com/api/global/us_indices/governance/Volatility_Index_Methodology_Selected_Broad_Based_Index_Equity_and_ETF_Volatility_Indices.pdf)。
