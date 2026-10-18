# 新机制有限筛选：Rule 201 状态与 SPO+

2026-10-18。结论：**本轮没有可直接冻结拟合的新候选**。保留一条 `needs_data_and_post_announcement_evidence` 的监管状态线索；SPO+ 暂不立项。共6组 web 检索、20条查询，后两组兼读原论文/作者代码；没有批量数据下载、特征/标签/收益计算、拟合或账户。Web 服务底层网络请求数不可见，不报告为0。本评议不改变当前日历来源主线或已关闭版本。

起点是已封存结果，不是新计算：分域头降低主 alpha 的 BCE/Brier，却没有提高约0.497的AUC；PG仅30/900步触发，2bp实际现金仍约46.44%。这说明“让提议更积极”或改进校准不等于识别了 SOXX 相对收益机会。见[分域头归因](../stage1_domain_heads_20261018/conclusion.md)、[PG归因](../stage1_aux_projection_allocation_20261018/conclusion.md)。持续目标仍是全部费用下稳定胜SOXX，含MaxDD≤30%、CVaR、完整年度/126滚动≥70%及独立前向资格；胜Stage1/B或降低现金本身不是成功。

## 1. 前一交易日触发、原决策时点已公开的 Rule 201 限制状态

**社区入口。** [Reddit 对交易所 SSR 表字段的讨论](https://www.reddit.com/r/WallStreetbetsELITE/comments/lb1f0f/)指向触发日、场所、状态、触发时间、预期解除日；[另一讨论](https://www.reddit.com/r/stocks/comments/lbg8kt/)提醒限制不等于禁止全部卖空。这些帖子含情绪化叙事，只用来发现产品，不能认证时点、做多方向或收益。

**原始依据与反证。** [SEC Rule 201 FAQ](https://www.sec.gov/rules-regulations/staff-guidance/trading-markets-frequently-asked-questions-7)说明：合规实施日是2011-02-28；以常规时段合资格成交相对上市市场前收盘触发10%跌幅，须由上市市场确认并由计划处理器传播后生效；一般延续至随后交易日，ETF没有一般豁免。因此本地OHLC低点穿过−10%不能冒充已公开的官方状态，也不能沿用社区“只要最后一笔是downtick”的解释。

[Ruchti、Barardehi、Bird、Karolyi，OFR 2023原研究](https://www.financialresearch.gov/working-papers/2023/10/11/are-short-selling-restrictions-effective/)在股票阈值附近、同一小时比较中发现限制减少积极卖空并提高价格；它没有认证 SOXL 在次日09:47以后、费用后相对SOXX的机会。[Gorbatikov/Sikorskaya作者稿](https://www.aeaweb.org/conference/2022/preliminary/paper/adANi4z2)，§6.1，提供ETF套利约束与误定价机制，但§3.4/PDF第13页明确排除杠杆、反向ETF，研究对象是AP网络与误定价而非本账户方向预测；不可移植到SOXL。[Jain/Jain/McInish原论文摘要](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6288.2011.00320.x)未发现清楚的估值、流动性与稳定性改善，方法/样本与OFR不同。保留这一不一致，不把任一论文当本项目验证。本轮未找到该状态到SOXL账户的作者实现。

**与旧路线差别。** 限定检索campaign、`src`及研究结论未发现已执行的Rule201状态实验；这不是全历史穷尽证明。它是有官方传播钟的交易约束状态，不是FINRA日成交short标记占比，也不是FTD余额/threshold证券表。外部固定10%规则有制度含义，不能据结果移动阈值。但状态高度依赖既有跌幅信息；新增非线性阈值不自动等于新增独立信息。

**数据与时点缺口。** [NYSE公告入口](https://www.nyse.com/market-notices/nyse)列有限制名单入口，[NYSE原始数据规范](https://www.nyse.com/publicdocs/nyse/data/NYSE_TAQ_Pillar_Products_Client_Specification_v2.3g.pdf)含SSR状态/触发时间字段。可公开阅读规范不等于免费取得完整历史数据；当前未认证SOXL/SOXX全期名单、证券身份、首次传播/更正/解除时刻及完整缺失语义，不购买TAQ、borrow或options。本项目已有价格只能作为控制量，不能替代官方确认。即使以后只用前日已触发的carryover状态，也必须在原09:29之前已公开，才能同时用于原训练输入与09:47账户；当日09:35新触发不能倒填09:29。

**最小可证伪检验，仅供以后设计。** 先限定一个 `SOXL_prior_session_SSR_still_active` 状态，不扩成SOXX/个股/盘中多个窗口。通过来源资格后，B/C保持原H5、原主辅训练/seed/损失/路由/风险与0/2/5bp账户；C仅加该状态，B/C同时拥有相同的连续跌幅及距10%阈值控制，A仍保留原冻结版本。先按原成熟训练域检查事件覆盖和独立事件数，不降低既有样本门。随后必须比较完整SOXX及强参考账户，不能只报触发日平均反弹、AUC或胜B。找不到免费历史传播证据、状态只是连续跌幅的重复编码、作用已在前日价格中体现、事件不足，或完整费用/风险/稳定门失败，都停止该版本，不换触发资产、期限或恢复阈值。

**裁决：needs_data_and_post_announcement_evidence，低优先级。** 有不同于FINRA的制度链条，却没有证明它能改善长期高现金和持续SOXX落后；论文的同日局部效应可能早于本项目可用决策钟。现在不采全历史、不生成代理特征、不训练。

## 2. SPO+：按下游决策遗憾训练预测器

**入口与primary。** [r/algotrading关于预测指标与策略收益脱节的讨论](https://www.reddit.com/r/algotrading/comments/1lflfn7/)是问题入口；[社交平台的SPO投资组合作业](https://www.linkedin.com/posts/samarth-srivastava-938a1b240_github-samarthftrneural-markowitz-porfolio-activity-7450561546235129856-2nom)仅作方法线索。[Elmachtoub/Grigas原论文](https://arxiv.org/abs/1710.08005)及[作者代码](https://github.com/paulgrigas/SmartPredictThenOptimize)给出以优化决策误差为目标的SPO及凸替代SPO+，代码含portfolio oracle；它不是“换一个更深网络”。理论针对规定可行集合与线性目标系数，并有分布/一致性条件，不是任意账户模拟器的收益保证。

**新意有限且尚不能接入。** 仓库未命中同名SPO+实现；但是[全路径直接扣费优化](../full_path_tiered_allocation_20261013/conclusion.md)已经27fits/66账户失败，原Stage1也已包含allocation utility，不能宣称此前只优化分类准确率。SPO+的优化oracle替代梯度与STE不同，但保留原整数持仓、Tiered费用、现金/峰值/风险内生路径的等价oracle尚未给出。改成静态Markowitz是更换策略与约束，不能记成仅换训练损失。现有成熟训练输入/收益标签原则上可供一个另行设计的监督检验；本轮未加载它们，没有新数据资格收益。

这里还有明确的数学边界：若可行权重满足 `1'w=1`，则 `(r-r_SOXX*1)'w=r'w-r_SOXX`；同一样本中减去基准常数不会改变最优权重，SPO决策遗憾中的该常数也抵消。这是本评议的代数推导，不是回测。因此把SPO改称“SOXX相对SPO”不能自行产生相对收益信息或解释现金偏高。原[超额选模路线](../benchmark_relative_loss_selection_20261012/conclusion.md)也已记录相应线性等价问题。

**反证与最小门槛。** [Elmachtoub等的ETO/IEO原研究](https://arxiv.org/abs/2304.06833)说明，在其模型正确设定且数据充分等条件下，两阶段估计再优化可在渐近遗憾分布上优于联合训练；SPO类训练没有普遍优势。若以后重议，先做不含收益搜索的形式化合同：同一可行集合/费用/状态下证明B/C仅估计损失不同，基准平移不变性以及oracle和真实整数账户的关系明确；做不到即停止。通过后才能另冻同容量、同源、同seed的损失配对和全部费用账户；若只靠增beta/减现金胜B却仍不能稳定胜SOXX，同样失败，不能据结果调风险半径、损失权重或窗口。

**裁决：not_ready_to_freeze，当前不立项。** 它是不同的监督替代损失，但目前既无可复用的原账户oracle合同，也无新增信息证据，不能列为高优先级救援。

本轮同时排除了：已登记的label-uniqueness（固定H5内部重叠近常数）、已有cost-sensitive action backup、条件分位数旧parked路线；通用GroupDRO检索未给出可唯一映射本账户的来源/分组/收益依据，不以新名称开启损失权重搜索。下一步仍由campaign按已授权来源主线推进；这两条线索都不构成新fit或账户执行许可。
