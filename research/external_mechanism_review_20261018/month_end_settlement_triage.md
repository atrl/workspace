# 月末现金结算需求：外部机制资格草案

2026-10-18，只读文献与既有计划。状态：`mechanism_lead_absolute_deadline_unresolved`；未生成日历特征、市场标签或信号，0拟合、0策略/参考账户执行。ROD主线优先；本文件不授权回测、数据采集或增加候选族。

**判断：机制具有可区分的新意，暂不冻结交易窗口。** 月末为支付筹措已结算现金的卖压，与泛“月底工资买股票”不同。已核的本地 `funding_information_20261014` 检验SOFR减准备金付息率的H5现金优势，股票月末只作分组控制；`fomc_preannouncement_20261014` 是预告会议日固定09:47–13:59事件；`dtc_calendar_source_20261015` 仅做结算日历来源资格。三者未直接检验随股票结算制度迁移的现金截止日。但新输入不构成新的独立市场样本。

社区发现入口是[QuantConnect月历示例](https://www.quantconnect.com/research/15377/turn-of-the-month-in-equity-indexes/)，不采用其中收益或参数作为依据。原机制来源为Etula等的[《Dash for Cash》，RFS 2020](https://academic.oup.com/rfs/article/33/1/75/5494694)；作者[2017-12-28全文](https://wp.lancs.ac.uk/fofi2018/files/2018/03/FoFI-2018-0169-Kalle-Rinne.pdf)可读，Aalto正式版下载此次403。原文以T为当月最后交易日，T+3下示例是T−4卖出、T−1收款，以便月底支付；讨论卖压后的反转，并使用2014欧洲T+3→T+2变化检验时点。不能把当时的绝对窗口原样套用至今。

法定制度变化日期可以预先确定：美国T+2合规实施日为2017-09-05，T+1为2024-05-28；交易日、结算营业日和银行支付日不可等同，转换月份应逐笔按生效日处理。[SEC 2017最终规则](https://www.sec.gov/rules/final/2017/34-80295.pdf)、[SEC T+1风险公告](https://www.sec.gov/compliance/risk-alerts/shortening-securities-transaction-settlement-cycle)

有实质后续证据：[Nathan、Suominen、Tasa作者全文，2026-06-08版](https://danielrnathan.com/assets/pdf/The_Intramonth_Momentum_Cycle.pdf) §5明确提出边际卖压随T+3/T+2/T+1从τ−5/τ−4/τ−3迁移，τ也是月末最后交易日。它报告美国2024和欧洲2014改革支持相对的一天迁移；但其主要对象是弱势股票的相对卖压，不是SOXX单资产方向。文中也明确六日集中收益分解不等于月内交易策略。美国2017改革检验方向符合却不显著（p=.60），T+1后样本只有19个月；详细限制见[作者附录IA.17及IA.20](https://danielrnathan.com/assets/pdf/The_Intramonth_Momentum_Cycle_Appendix.pdf)。这是工作论文，不能描述成已同行评审的SOXX稳定超额证明。

**必须先解决的具体差异：** 原文T+3支付例子对应T−4；新稿T+3边界为τ−5。两文使用相同月末零点，不能直接当作记号转换。新稿将其边界联系到全样本平滑的PreTOM窗口，未在本次查阅范围内消除与原支付例子的一日差异。因此“缩短一天应后移一天”获得外部支持，不代表已经唯一确定绝对现金支付截止、传递缓冲和可交易预测窗口。禁止用SOXX结果在这两套偏移之间择优。

退休金解释的反证也要保留。[HKUST 2018条目](https://researchportal.hkust.edu.hk/en/publications/turn-of-the-month-effect-and-retirement-account-flows-evidence-fr-2/)及同组作者[2021年完整后续稿](https://www.aeaweb.org/conference/2022/preliminary/paper/iQikdBKZ)未发现401K月底流入能解释月底效应；后者还明确区分其检验与Etula的机构现金需求。它削弱“工资/退休金自动入场所以月底必涨”的说法，既不直接推翻为支付卖出筹现，也不能替后者证明SOXX有效。

仅保留一个待资格的输入想法：`is_last_sale_day_for_month_end_settled_cash`。在确定唯一支付截止与营业日规则后，按当时适用结算制度反推最后可卖出日；它是预先可知的0/1日历输入，无需期权、gamma或每日资金流估计。此定义是本项目待审推论，不是现成论文复制；目前不生成其值，也不附加前后若干日、方向或强弱势子窗口候选。

最小缺失只有三项：

1. 明确选择并证明一种现金支付截止/结算缓冲定义，解释原文与新稿的一日差异；可先通过作者现有附录或可核对机构支付规则完成，不能拿市场收益校准。
2. 对拟研究年份取得普通股票CNS与银行支付服务的正式营业日/例外规则。现有 `research/dtc_calendar_source_20261015/conclusion.md` 只有2026年度预告，Open1为有限服务、Open2不提供结算；2023–2025及服务级例外仍缺。已知XNYS交易日历不能单独补齐。
3. 若后续登记SOXL/SOXX/CASH实验，先说明从市场/弱势股票卖压到三资产配置的唯一可证伪预测与信息时钟；不自动引入成分股弱势排序或第二候选。然后独立审查成本、匹配日历对照及完整SOXX门，不以论文统计显著性替代本项目证据。

交付边界：仅此triage文件；未改ROD代码或既有卡片，未读取本项目市场收益。文献查阅支持后续来源澄清，不改变任何已关闭路线或当前执行授权。
