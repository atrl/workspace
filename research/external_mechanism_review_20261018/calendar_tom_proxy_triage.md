# 固定四日 TOM：交易日历代理资格

2026-10-18。结论：**可以提出一个事前固定的 exchange-calendar proxy**；窗口定义已具外部依据，历史输入仍需限定的 `calendar_knowledge_audit`。不要求先证明养老金流量或真实现金结算截止日的因果解释。本轮仅文献/源码/既有结论去重，0本项目行情收益读取、0特征、0拟合、0账户；不授权采集或训练。

[QuantConnect 示例](https://www.quantconnect.com/research/15377/turn-of-the-month-in-equity-indexes/)只作为入口。决定窗口的依据是 McConnell 与 Xu（2008）[《Equity Returns at the Turn of the Month》，作者官网全文](https://business.purdue.edu/faculty/mcconnell/publications/Equity-Returns-at-the-Turn-of-the-Month.pdf)：印刷第50页/PDF第3页明确为与既有研究保持一致，采用 Day−1 至 Day+3，即使图形可能暗示更早起点也没有把本表窗口改为−2。第51页表注定义 **四个交易日 `{−1,+1,+2,+3}`**：前月最后一个、次月前三个，**没有 Day0**。这给本项目提供了在查看 SOXX 结果之前可锁定的唯一窗口；不声称原作者进行了现代意义的预注册，也不搜索更优起止日。

此 proxy 与已 parked 的 [现金结算截止日](month_end_settlement_triage.md) 是不同声明。后者试图从机构支付要求、股票结算制度和银行/CNS营业日反推卖出期限；本候选仅编码公开交易日历位置，不称已经观测到401(k)入金、养老金买盘、机构资金流、具体支付期限或精确结算缓冲。因此 DTC/银行支付日历未合格不自动阻止这一较窄的日历假说。

反证必须保留：

- McConnell/Xu 本身未发现交易量或股票基金净流入足以解释该效应；论文仍将成因视为未解决问题。
- Choi、Hong、Lou、Mukherjee [2021-12作者稿](https://www.aeaweb.org/conference/2022/preliminary/paper/iQikdBKZ)（第6页；[2018机构条目](https://researchportal.hkust.edu.hk/en/publications/turn-of-the-month-effect-and-retirement-account-flows-evidence-fr-2/)）未发现401(k)月底支付暴露与TOM效应有预期关系。这否定把工资/退休金流入当成既定解释；并不等价于证明四日日历指标没有任何预测性。
- Maberly/Waggoner 的 [Atlanta Fed WP2000-11](https://fraser.stlouisfed.org/title/working-papers-federal-reserve-bank-atlanta-8586/closing-question-continuation-turn-month-effects-656698)报告其标普500期货及现货样本在1990年后效应消失；McConnell/Xu 在不同市场指数与时期得到不同结论。样本、微观结构、ETF/期货套利及结算制度变化使稳定性不能外推至当前半导体ETF。固定proxy不因T+3/T+2/T+1变化自动移窗，变化只能作为后来预登记的制度分层描述，不能择优救援。

限定仓库检索未找到等价“最后一个＋最初三个交易日”输入或策略已经执行的证据：检查 campaign/SOP、研究结论、`src`/`tests`，并补查已跟踪路径。`funding_information_20261014` 的股票月末仅为SOFR信息检验的粗分组控制；月度估值的月末调度也不是该四日proxy实验。保留这些既有失败，不把“未找到”夸大为全历史检索证明。

唯一待资格输入可命名 `exchange_tom_4d_proxy`，在原决策时点、纽约交易日期 d 上定义为：

`1{ d 是本月截至决策时已确认开市序列的第1/2/3日，或 d 是当时已公开交易安排下本月预期最后一个交易日 }`。

这是可证伪的季节性输入，不直接规定加仓方向、SOXL权重或持仓期限。日历信息虽不需要先认证经济成因，仍必须在后续独立冻结的匹配实验中证明增量；论文日收益不能直接当成本项目09:47执行账户的净收益。

时间与缺失合同必须同时满足：

1. **前三日**：从月初至 d 的已确认开市记录计数；不需要读取未来开市日。月初历史覆盖不足或当日未确认开市则资格未知。半日市仍计一个交易日，不因为回报长度较短而删掉。
2. **预期末日**：使用决策当时已经公开的交易所日历或可审计的假日规则/变更公告；覆盖必须越过当月末并明确下一计划开市日。禁止把 `calendar<=as_of` 的最后一行、行情文件末日或 `groupby(month).tail(1)` 当作真实月底。
3. **临时停市**：只让已公告的更改影响公告之后的决策。以后意外停市不得追溯把先前某日改成“当时已知月底”；保留当时expected-last标记及其与最终实际日历的差异，不补出强制四个成交日。未知不填0；若前三日分支已经确定为真，OR结果可为1，其余未明情形保留未知。
4. **来源**：既有 Qlib 实际日历可支持截至决策的历史开市计数，但不能独自认证未来末日安排当时已知。交易所确会提前公布安排，例如 [NYSE/ICE于2025-12-23发布2026–2028日历](https://ir.theice.com/press/news-details/2025/NYSE-Group-Announces-2026-2027-and-2028-Holiday-and-Early-Closings-Calendar/default.aspx)；当前页面不能倒灌早年。还须保留假日规则生效边界，例如 [Nasdaq 2022 Juneteenth公告](https://www.nasdaqtrader.com/TraderNews.aspx?id=ETA2022-55)。不以今天的库输出自动代替历史知识时钟。

最小剩余工作仅是**拟研究年份的交易日历规则/公告知识账本与覆盖检查**，包含已知例外和版本/hash/公开时间；无需取得真实机构支付期限，也无需新期权、借券或成分股数据。本轮没有验收这个历史账本，所以只给予窗口与窄声明资格，不宣称已生成时点合格的全历史输入。随后若受授权，也只允许这个固定四日proxy、同可用性对照及原全部费用/风险/稳定性门；不扩展weekday、月底前若干日或其他calendar家族。
