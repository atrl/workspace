# 同期可比实际值：样本与版本不能拼接

固定核查两个前轮已识别的口径案例，未使用股价表现筛选事件或标签。本轮0拟合、0回测，不生成surprise，不改变已存预期或任何旧账户结果。

2019-02-14：已存Refinitiv DayAhead EPS预期0.75；Reuters署名、Business Standard转载的同期报道明确比较剔除项目后实际0.80与分析师预期0.75，与公司非GAAP数值一致。文章前段将营收预期归于IBES/Refinitiv，但EPS句没有单独标明供应商或样本范围，因此这是一项明确的新闻配对证据，仍不是完整I/B/E/S actual口径/历史版本认证。页面显示首次发布时间2019-02-15 03:15 IST、更新03:30 IST，保留原文，不将今天取得的网页自动当当年首次版本。

2025-05-28：原DayAhead普通预期及SmartEstimate均0.88；Reuters同期法文转载同时给出普通调整后实际0.81、剔H20费用及税项后的0.96，以及LSEG预期0.93。0.93段明确提及4月15日限制披露后提供预测的17位分析师；旧0.88快照没有相同样本范围证据。数值差可能涉及时间、样本或定义，不能只称“修订了5美分”，也不能把0.93覆盖原0.88。

该报道没有清晰消除“究竟哪个actual与哪组预期完全同口径”的问题。不能因为0.96与次年简报去年值相同就认证，也不能选择会产生正surprise的组合。0.81、0.96、0.88、0.93四个事实保留为不同字段，不拼成训练标签；下一年旧值禁止回填上年。

正常HTTPS下载LSEG 2025-05-29 Morning News Call成功，原PDF已冻结哈希及NAS归档，无TLS例外；第2页NVDA段落已视觉核验，仅讨论销售、指引和贸易限制，没有实际EPS/预期的数字配对。不能把“下载到同供应商晨报”当成口径缺口已解决。文件当前可得、页面/PDF日期均不独立证明历史首次公开时间。

来源：
- https://www.business-standard.com/amp/article/reuters/nvidia-beats-fourth-quarter-profit-estimates-shares-rise-119021500096_1.html
- https://www.boursorama.com/bourse/actualites/nvidia-prevoit-un-chiffre-d-affaires-inferieur-aux-estimations-les-restrictions-imposees-par-les-etats-unis-coutant-8-milliards-de-dollars-de-ventes-5685568571124616d4baa8eb40560a69
- https://share.lseg.com/assets/newsletters/Morning_News_Call/MNCGeneric_US_05292025.pdf

决定needs_contemporaneous_comparable_actual_and_consensus_population，合格训练案例仍为0。所需数据合同包含：当期可比actual的调整项目、对应预期的分析师样本/版本、生效与公开时点、提前预警。新闻出现同一数字只能补证，不能替代完整合同。该发现约束下一项预期差研究，尚不能解释旧策略亏损，也未实现稳定超额。生产保持不变。

原始工具文本、PDF和下载回执存Qlib Recorder/NAS；notebook核验哈希、关键数值文字与父Recorder两条原始预期一致，冷恢复逐对象哈希通过。没有新业务逻辑，不增加软件测试套件。保留两个案例的不同资格，不用一个报道配对成功外推全部35次事件。
