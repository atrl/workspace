# SEC 原始披露文本：唯一来源待资格候选

2026-10-18。状态：**`needs_frozen_pre_cutoff_lexicon_and_causal_sector_corpus`**。
本轮只保留一个候选：固定金融不确定性词表下，原始年度 10-K 措辞的同比变化，映射为事前公开 SOXX 披露篮子的文本代理。
这是来源可行性线索，不是已冻结特征、模型或 fit GO。未按当前未归因账户结果挑词、发行人、窗口或方向。

检索预算为 3 组、8 条查询尝试；前两组 6 条取得结果，第三组含 2 条查询的组合调用连接失败，仍计入额度且未重试搜索。
之后仅打开已知论文/作者与 SEC 官方文档。没有下载真实 filing 语料、字典文件、权重或市场数据集，
没有读取价格/标签/预测对象，没有计算特征、fit 或账户。

## 去重：与哪些旧路线不同

本轮检索 campaign、SOP、相关卡名及静态代码，未发现等价的固定词法不确定性增量实验；不宣称穷尽所有历史。
[NVDA SEC 原文序列卡](../nvda_sec_eps_series_20261015/plan.json)是连续 48 季的数值与版本核对，
不是训练文本 uncertainty；已闭 EPS、revenue、guidance、季节盈利及毛利率路线继续关闭。
[CPI_text 卡](../cpi_opening_context_information_text_20261015/plan.json)接收的是发布日期上下文与早盘响应交互，
不是对披露语料做情绪分析。已有 Companyfacts 数值和 NVDA 文本摘录也不能当成完整行业 10-K 原文语料。

新假说只来自**措辞信息**，不重新估算 guidance 数值，不把同一句收入预期再包装成另一种 surprise。
不选择 NVDA/AMD 等事后显眼公司，不建立新的个股预测主线；最终接收者仍是 SOXL/SOXX/CASH 配置。

## 社区线索、原论文与作者资源

[r/algotrading 的 SEC 文本讨论](https://www.reddit.com/r/algotrading/comments/gb2pj1)
主要暴露 10-K/8-K 清洗、章节与 exhibit 分离的困难；[QuantConnect SEC 数据文档](https://www.quantconnect.com/docs/v2/writing-algorithms/datasets/securities-and-exchange-commission/us-sec-filings)
提供了用词评分的发现入口。两者都不是本账户的盈利证据，也不自动提供合格历史接收时钟。

- [Loughran/McDonald，2011 原论文出版页](https://doi.org/10.1111/j.1540-6261.2010.01625.x)：首次出版日期 2011-01-06，
  研究 1994–2008 年 10-K。摘要说明通用负面词典会误判金融用词，并构建金融专用词表；列出与申报期收益、
  波动等的联系。它不是 SOXL/SOXX 的 H1/H5 预测或净账户验证，本轮未复现其全文回归。
- [作者的 Master Dictionary 页面](https://sraf.nd.edu/loughranmcdonald-master-dictionary/)
  明确有 uncertainty 类和词条添加/移除年份；当前版本是 **March 2026 / 1993–2025**，且说明会扩充词表。
  [作者代码入口](https://sraf.nd.edu/textual-analysis/code/)与官方加载器说明提供可审计实现方向，未下载运行。
  当前页面注明学术研究免费、商业许可另行联系；本轮不联系、不购买，也不声称该许可无条件适用于所有用途。
- [Feng Li，2006 工作论文原始登记](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=898181)
  的原作者摘要记录风险/不确定性词频变化与以后盈利及十二个月个股截面收益的研究。
  它说明词法路线在 2011 前已有研究，但不是近期开盘配置证据。正文直读本轮返回 403；
  未从课堂幻灯片或第三方复述复制精确词表/公式来冒充核验过的原文，不另提出第二个词典候选。

## 时间资格：训练组件与原始披露不能混为一谈

本项目是一次用 2011–2022 训练、2023 起评价的固定模型。**词表/变换不必在每个 2011 训练样本前已存在**；
能证明其在 2023 cutoff 前固定、或仅用训练域且不含 2023+ 文本/标签构建，即可作为训练组件讨论。
2011 年论文的发布时间晚于最早几行训练日期，本身不是阻断。
预 2011 规则只是更简单的证据路径，不是新增硬门。

但当前 2026 master 词表、全样本词频/IDF、2026 训练的 LLM 或 embedding 不能直接回看 2023–2026。
年度添加/移除字段不是已经冻结的旧版字节；尤其当前负号移除年份不提供完整版本历史，
简单保留正年份早于 cutoff 不能自动认证旧词表。最低需要明确版本的词表/许可/来源与 SHA，
仅使用其词成员关系，不使用含评价期语料的频率统计，也不事后删选“有效”词。

**每份原始披露内容及其 available_at 仍须按当次决策因果。**
年度报告的 fiscal period end、filed date、acceptance、网页首次可用、修订/抓取时点均不可互换。

## SEC 官方来源可行性与时钟边界

[SEC 数据访问说明](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data)
提供 1994Q3 起的索引、CIK/accession 原文路径及 SGML header；历史公开原文原则上覆盖本项目训练期，
但本轮未认证任何具体发行人的完整语料覆盖。该说明也指出 post-acceptance correction/deletion 会更新索引，
当前 quarterly/full index 不等于从未修订的历史快照。

[官方 API 文档](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
说明公开 API 不需认证或 API key；当前 submissions 只内含至少一年或最近 1000 份中的较大者，
更早记录要按提供的 additional files 继续定位。Companyfacts/frames 是结构化事实接口，不是原始文本替代品。
本轮没有调用公司 API 或下载索引/语料。

[SEC Webmaster FAQ](https://www.sec.gov/files/about/webmaster-faq.htm)明确：网页通常在 acceptance 后 1–3 分钟可见，
负载高时更久，时延不保证；**没有标示 sec.gov 首次可用时间的 timestamp**。
header 的 acceptance、`FILED AS OF DATE`、`DATE AS OF CHANGE` 含义不同。
[官方 filing 状态说明](https://www.sec.gov/submit-filings/filer-support-resources/how-do-i-guides/determine-status-my-filing)
还说明多数 17:30 ET 后提交的普通表单延至下一工作日 disseminate；这是当前规则说明，不能不核历史版本就覆盖所有年份。

因此不能把 09:28 acceptance 直接当作 09:29 可见，更不能因 09:47 才成交而允许模型偷看 09:29 之后的新披露。
未来来源资格须绑定 header 时间约定、时区/DST、实际或有依据的公开边界；只具日期时可提出事前固定的保守研究约定，
并保留 conditional/first-receipt 未认证限制，不能为了提高覆盖而缩短延迟。
官方当前 HTML/JSON 时间值也必须与原 header schema 对照，不能只看字符串尾缀就猜本地时钟。

只取原始 accession 的申报主体、form、primary document 与当次公开文本；后续 10-K/A 独立记录，不能覆盖旧版本后倒灌。
PAC 导致当前字节变化、原件无法恢复的记录须显式标未知，不把“今天还能下载”当原始 vintage。
10-K 的文本亦可能早已在 earnings release 披露；SEC acceptance 不是其经济信息首次出现的证明。

## 唯一候选及发行人映射

候选概念固定为：**原始年度 10-K 的 uncertainty 词频占比，相对同发行人上一原始年度报告的变化**。
使用一个已资格的 cutoff 前固定 LM uncertainty 词集合，不做现代 LLM 判断，不扩展 demand/supply 自选词组，
不混合 8-K、10-Q、20-F 或挑选句段来改善结果。该同比构造是本项目待检验推论，不声称逐项复现上述论文。

经济链条是：新增的风险/不确定性措辞可能反映管理层认知变化 → 形成价格输入没有显式测量的行业上下文
→ 原模型可能更好区分应持 SOXX、SOXL 或现金的情形。每个箭头均未证实。
词频不是“真实需求风险”，法务模板、否定句、文档增长和监管格式变化都可能产生相同读数。

发行人集合只能由事前已公开的披露篮子/历史身份决定。已有
[SOXX 披露篮子来源](../etf_basket_valuation_inputs_20261016/conclusion.md)使用当时已公开的最近 N-PORT 权重，
是滞后的 disclosed-basket proxy，不能称即时成分。其已核范围从 2019 后开始；不能用当前成分回填 2011–2018，
也不能将今天选出的 NVDA 等固定回整个训练期。CIK/合并更名/旧新主体须继承时点映射。
未提交 10-K 的外国发行人及缺失原件的权重应单列；本轮不为补覆盖混入 20-F，不重归一化已知部分冒充全篮子，
也不把未知文本当零不确定性。全 3014 原训练 cohort 不因文本缺失删行。

目前未取得固定词表字节、逐发行人原件/可用钟、同发行人前期报告及实际覆盖审计，故**尚无合格数值输入**。
不在这份 triage 中定缺失门槛、衰减窗口、issuer 子集或模型参数。

## 最小可证伪条件与下一步边界

后续若单独获准来源资格，先证明一个固定词表、一个年度 form/解析范围、原件配对与公开时钟，以及已披露篮子的覆盖；
保存原字节/段落位置，去除 HTML/XBRL 隐藏重复、表格和附件的规则事前固定。来源不足只结束该来源版本，不结束总 Goal。

若再获完整实验授权，最小 matched B/C 必须共享原输入、容量、availability/报告年龄/文档长度控制及训练预算，
C 只增加这一词法变化。不能把披露频率、篇幅增长、缺失年份或 issuer 选择造成的差异当文本信息。
在原 09:29 已知、09:47→次日 09:47 H1 与原 H5 共同成熟合同下执行完整费用账户，不能把申报日反应或一年截面 alpha
直接迁移为当下目标的成功。若信息已在入场前被价格吸收、只解释长期风险而不改善该接收层，或仅让现金更高，机制可能失败。

正文出处/词表版本无法固定、修订回填、时钟或发行人映射无法说明，则停止来源资格；合格之后仍须完整 SOXX、匹配 B、
原模型和强参考的净超额、MaxDD/CVaR、年度/126 滚动与配对区间门，历史通过也不替代充分独立前向。
本轮裁决仅为**未发现等价旧实验、值得有限来源资格审查的单一新信息线索**，没有收益预期或训练推荐。
