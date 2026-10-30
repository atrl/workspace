# Stage1 v5 现有 Qlib/NAS 候选准入清单（2026-10-30）

**范围与决定。**本次只核已登记的现有市场源、保存标签及关闭路线，不读取新的未来收益、拟合模型或运行账户；SOXX 0/2/5bp 历史 Goal 未通过。当前清单没有一项可直接进入“新的同日分钟决策、连续三资产配置”拟合，状态为 `needs_distinct_decision_time_mechanism`。这是**有限已登记候选的准入结果**，不是所有可能分钟因子都无效的证明。

| 候选源/机制 | 已验证的来源或局部经济证据 | 当前准入决定与具体原因 |
|---|---|---|
| SOXL/SOXX 原生 1 分钟 OHLCV | [完整分钟源](../research/stage1_event_minute_source_20261028/conclusion.md)、[固定原价股数五会话标签](../research/stage1_v5_fixed_share_lock_labels_20261030/conclusion.md)提供同状态整股 Tiered 局部价值 | [自身分钟回撤](../research/stage1_v5_fixed_share_drawdown_information_20261030/conclusion.md)虽变分数但2022Q2首分钟仍全部买SOXX；[保存模型审计](../research/stage1_v5_effective_sample_design_audit_20261030/conclusion.md)显示200403训练分钟行仅549日、五会话非重叠起点上界112。继续加同源窗口/正则不是新机制。 |
| 前一交易日尾盘有符号 SOXL−3×SOXX 残差 | [来源卡](../research/stage1_late_leveraged_tracking_residual_source_20261024/conclusion.md)有1060/1061旧训练日、精确f696评价944/944日来源；与原39及既有分钟字段的最高秩相关0.168；没有信息拟合 | 仅在次一交易日开盘前成为一个**日内不变**的值，不提供当前每分钟的行动时点信息，也不是已测得的基金再平衡流。仓库已有同类价差输入；仅换尾盘长度/父本/标签不满足新机制准入。原来源卡保持来源合格的原作用域，不宣称其预测力失败。 |
| QQQ/SMH/AMD/NVDA 原生 5 分钟同行价量 | 来源时钟、QQQ下跌时SMH相对抗跌变化曾通过[来源卡](../research/stage1_v5_peer_downshock_resilience_source_20261030/conclusion.md) | 匹配原39、状态、普通分钟价量、旧广度及无条件价差后，[固定信息门](../research/stage1_v5_peer_downshock_information_20261030/conclusion.md)首单2021/22仅改4/1日；SOXX MSE恶化、2022Q2仍61/62日买SOXX。改变条件、窗口或换同行称谓属已停路线救援。 |
| 逐日披露的半导体成分篮子 5 分钟价量 | [跨期来源卡](../research/constituent_intraday_history_20261014/conclusion.md)含37只历史成员并集、2024–2025共502完整日；[2026Q3来源](../research/constituent_intraday_q3_20261014/conclusion.md)另有64日。简单加权篮子领先SOXX的[固定信息门](../research/basket_etf_lead_information_20261020/conclusion.md)失败，双向轮动的[恢复价值门](../research/rotation_information_20261014/conclusion.md)也失败。 | **原清单遗漏此存储，现已纠正。**其5分钟历史并不覆盖当前固定股数标签的2018–2020发现训练期，2024–2025和2026Q3已被研究使用；不能移作2018–2020逐层点时新接收器的训练特征。披露篮子不等于每日真实指数权重，改变成员数/切点也不重开两条失败解释。 |
| Futu 原始 SOXL/SOXX 1分钟与 QQQ/SMH 5分钟成交额 | 原字节/NAS与Qlib时钟核对通过；[本体](../research/stage1_v5_raw_turnover_minute_source_20261030/conclusion.md)及[同行](../research/stage1_v5_peer_5m_turnover_source_20261030/conclusion.md)语义预检已完成 | 报告额/量在同bar OHLC±1bp内比例分别90.77%、93.2457%/99.7416%，未过预定99.9%；本体活动强度与旧close×volume秩相关>0.999999。不当真实VWAP、订单失衡或ETF流使用。 |
| 原39、VIX/指数及其他既有日线字段 | [无旧上限训练准入](stage1_cap_free_retraining_admission_20261029.md)与[旧路线审计](../research/stage1_minute_receiver_failure_audit_20261028/conclusion.md)列明连续模型、收益目标、路径/早盘价量及风险状态的完成作用域 | 日线字段不能直接为下一根分钟bar提供新信息；重新组合或提高模型容量不解决当前分数严重失准和订单不变。旧 VIX、开盘缺口、跨ETF残差等固定失败版本保持关闭。 |
| SOXX NAV/溢价、份额、基金流及历史期权链 | [SOXX官方历史下载](../research/stage1_soxx_official_premium_discount_source_20261023/conclusion.md)没有历史市场价/溢折价列；[数据边界](stage1_cap_free_retraining_admission_20261029.md)与来源卡保存原证据 | 当前没有可供同日分钟决策的合格、可按历史首次发布时刻验证的完整序列。不得将回溯日值、SOXL价差或不完整期权样本伪装为该信息。 |
| 发行人 SOXL/SOXS 份额的 Qlib `shares_million.1min` 缓存 | [原始快照卡](../research/leveraged_fund_snapshot_20261014/conclusion.md)仅取得2026-10-13一个TradeDate的原件；float32分钟字段是这一快照的观察缓存，原分钟`turnover_rate`无法反推历史份额。 | 文件名中的`1min`不是2018–2022逐分钟份额/申赎历史。历史净流、AUM、披露时钟均不足，不能称为真实杠杆ETF再平衡压力。 |

**来源清单纠正。**首次有限清单只列了四只同行5分钟源，漏列2024–2025的成分篮子提供者和单日份额快照缓存。重新核本机Qlib provider、日历边界和原卡后，两者都没有补出2018–2020可训练的独立分钟流/完整篮子特征；“目前无已准入候选”的有限结论不变，但原清单的覆盖描述已修正。以后作候选穷举前，先列出全部嵌套provider的标的、频率、日历首尾和原始来源身份，再判断可用训练期；不能仅按`1min/5min`文件名推断历史覆盖。

**接收任务仍需修正。**现有固定股数五会话标签只证明一笔预定增配的同状态局部价值，不是连续 SOXL/SOXX/现金目标的完整账户标签。旧 QQQ 条件四头 Ridge 在2022Q2预测 SOXX−HOLD +91.748bp、实际−20.859bp，62日都在首分钟行动；小额 Tiered 手续费不是主要亏损。新机制需事前指定与这个亏损链路不同的可观测原因，先以完整日历的低自由度、按日/五会话重叠隔离方式检验**输入变化→分数→整股行动→费后价值**，再在训练/推理同一风险域生成连续输出。没有达到该准入条件时不复训，也不把旧日线或来源合格等同SOXX超额。

**下一可执行边界。**继续只在既有 Qlib/NAS 数据里寻找尚未关闭、能在当前分钟首次可见并改变资金/风险选择的经济机制；每个提议先和本表及保存的 Recorder 对照。若没有，保持 `needs_distinct_decision_time_mechanism` 并按 Goal 的连续阻塞审计计数；不把它写成 `historical_research_passed` 或生产晋升。
