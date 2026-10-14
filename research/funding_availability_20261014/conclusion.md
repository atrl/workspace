# 融资信息保守时点：使用存档与真实政策版本，不回填早期

本轮0拟合、0回测、0收益相关检验。通过官方匿名ALFRED下载表单分批取得SOFR、IOER、IORB的Initial Release Only ZIP，逐份核对README、系列名、列结构、日期范围与SHA256；原ZIP及政策文件保存NAS。SOFR1966行中84行是没有官方SOFR观察的日期，保留原档，不变成0；1882个有效SOFR观察逐日与官方现行利率相符，期间无内部缺口。

ALFRED首批可识别SOFR观察为2019-03-29，存档日期2019-04-01。不能因为NYFed原始系列从2018年开始，就赋予这些旧值未经证明的历史首发资格。保守研究信息从2019-04-02 09:46纽约时间开始；2146个原生股票日历位置全部保留，1894日有已知信息，前252日保持未知。

时间合同：max(底层有效日期、SOFR存档日期、政策利率公告/存档日期)的纽约23:59:59之后，才进入原生股票交易日09:46。按日期级证据延迟，不宣称已认证最早市场发布时刻；不选择更早的滞后来提高回测。它规避同日14:30可能修订，也无需把债券/银行假日错误等同NYSE假日。观测年龄明确保存，旧值延续不伪造为新观察。

发现并修正一个重要版本陷阱：2020-03-16 IOER的初始存档在3月13日预填1.10%，但3月15日官方实施说明已公布3月16日起为0.10%。必须服从决策前已公开的后续政策，不能机械使用最初存档。修正后该日SOFR0.26%减IOER0.10%=16bp，按SOFR存档日3月17日结束后在3月18日可用。另用2021-07-28官方说明补足IORB起始2021-07-29的0.15%记录，保留制度标记及来源；没有依据收益选取修正。

例：2026-10-12与10月13日09:46仍使用10月8日观察，因为10月9日的SOFR到10月13日才存档，保守规则不允许提前放入开盘。测试另验证其直到10月14日才能进入该保守时钟。实际历史市场可能更早知道同日08:00值，本卡不冒充精确盘中首收。

决定：conservative_archive_date_funding_information_qualified_for_research_only。派生provider具备明确日期级保守研究时点，research_information_qualified=true；production_input_qualified=false、intraday_publication_receipt_certified=false。原始有效日provider仍不能直接拼入模型。SOFR分位数/成交量的季后修订统计未进入本派生信息字段。

下一步才可对固定融资压力机制做独立预登记的信息诊断，保持不拟合、不按收益调整时钟。此利差不是SOXL实际融资成本，没有任何新增策略超额证据，现役策略未改。

主要来源：[ALFRED SOFR](https://alfred.stlouisfed.org/series?seid=SOFR)、[2020年3月15日实施说明](https://www.federalreserve.gov/newsevents/pressreleases/monetary20200315a1.htm)、[2021年7月28日实施说明](https://www.federalreserve.gov/newsevents/pressreleases/monetary20210728a1.htm)、[NYFed发布和修订政策](https://www.newyorkfed.org/markets/reference-rates/additional-information-about-reference-rates)。[政策版本核对](policy_reconciliation.csv) · [时钟示例](clock_examples.csv)。

Recorder `a1c69bbbab724a33803c22aee98888d9`。
