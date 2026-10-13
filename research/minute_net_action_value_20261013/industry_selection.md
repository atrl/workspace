# 公开业界研究如何选择，以及本次取舍

来源核查日：2026-10-13。下列为机构研究者和原作者的公开研究，不代表各机构当前专有策略，也不保证迁移到 SOXL/SOXX 后有效。

|原始研究|选择原则|现有模型差距|本次落实|
|---|---|---|---|
|[Gârleanu/Pedersen，AQR/Journal of Finance，2013](https://www.aqr.com/Insights/Research/Journal-Article/Dynamic-Trading-With-Predictable-Returns-and-Transactions-Costs)|信号衰减期限、风险和成本共同决定目标；调整可以只走向目标的一部分。|日级训练效用、H5 辅助标签和盘中恢复决策并非同一持有期限。|固定半步恢复；标签从真实延迟执行到次日共同重置，完整收取分支费用，不搜索半步比例。|
|[Boyd 等，Multi-Period Trading via Convex Optimization，2017](https://web.stanford.edu/~boyd/papers/cvx_portfolio.html)|交易选择同时考虑持仓、预期收益、风险、交易/持有成本；多期计划只执行第一步，随后重新评估。论文没有解决预测质量问题。|把更好的预测分数当作更好的账户表现，缺少费用与状态传导验证。|现有持仓和风控状态作为条件；选择调整或 HOLD；净价值减均值估计误差后为正才动；每个成本情景独立结算和回测。不是完整多期凸优化的复现。|
|[Frazzini/Israel/Moskowitz，AQR，Trading Costs，2018](https://www.aqr.com/Insights/Research/Working-Paper/Trading-Costs)|用真实成交数据校准交易类型、规模、时间、股票和交易所带来的成本差异。|当前滑点为假设，尚无真实报价/场所/回执校准。|保留 IBKR Tiered 逐笔真实价格股数费表与 0/2/5 bp 压力；不能从最好结果倒推滑点应为零。实际成交校准仍是缺失信息。|
|[Bailey/López de Prado，Deflated Sharpe Ratio，2014](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)|多次试验、样本长度与收益分布会抬高获胜回测分数；应保留失败试验和选择偏差。|2023–2026 已被多轮研究，不能因改一次模型就称为新样本外。|预先限定 6 次年度训练、12 个 native 账户、一个分钟候选和状态对照；不使用外窗选阈值；报告历史复用。区块置信界仅作路径诊断，不声称消除了历次选择偏差或计算了完整 DSR。|

选择顺序：信息时点与标签期限成立 → 在状态对照之外有分钟增量 → 完整账户在所有成本情景下有净 SOXX 超额 → 年度/滚动稳定与回撤/CVaR → 独立前向及成交实现性。约 30% 回撤是本次研究门槛，不能解释成未来保证。

本次选分钟净执行价值，而不增加网络复杂度或调风控触发阈值。依据：现有分钟信息没有进入风险/alpha 头；费率核对后，极端收益差主要来自费用引起的风控路径变化；先用实际持有期限检验分钟信号能否产生有效调整，再决定是否值得开展另一种控制器研究。

本次不改变 09:30 原开盘控制器及仓位约束，只替换已有盘中恢复机会的价值评分。新增分钟特征为已完成首小时的收益、下行方差、末段方差/成交量占比、路径效率、回撤/反弹、SOXL 对 SOXX 的跟踪残差及方向一致性。全日和隔夜信息只进入次日到期标签。此前前一完整交易日分钟特征的 MLP 实验保留原负结果，不重训或挑选最好种子。

原草案的“60 分钟收益＋次日重置费用”不一致，本次改为“10:32 到次日 09:30 净收益＋两分支共同重置费用”。共享重置目标只用作未来标签，不作为预测输入；它来自固定参考账户，仍然只是一步反事实代理，不是完整未来控制树。

使用 Qlib 官方 [Data](https://qlib.readthedocs.io/en/latest/component/data.html)、[Model](https://qlib.readthedocs.io/en/latest/component/model.html)、[Strategy/Backtest](https://qlib.readthedocs.io/en/latest/component/strategy.html)、[Recorder](https://qlib.readthedocs.io/en/latest/component/recorder.html)。模型采用 DatasetH/DataHandlerLP/Model，价格与日历由 Qlib 管理，交易与费用通过 native Exchange/Position 对账。稀疏 1min 日历是观察到的执行事件序列，不声称是完整分钟 K 线；特征从原完整 1min store 读取。
