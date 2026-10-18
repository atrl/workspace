# SEC FTD 公布时点资格检查

2026-10-18；状态：**not_selected**；来源缺口：**needs_publication_timestamps**；机制缺口：**needs_etf_post_release_evidence**。本轮不立特征或训练计划，不切换 PG 主线。未下载 FTD ZIP/CSV、未批量采集、未读取本项目收益标签、未计算特征/梯度、未拟合或运行账户。

候选仅为 SEC 公开聚合未交割余额的新增信息，不是已关闭的 FINRA short-volume q。FINRA q 是场外成交中 short 标记占比；FTD 是 CNS 系统某结算日的未交割余额。两者不能互换，也不能把 FTD 当作空头仓位、净卖压或确定的未来回补量。

仓库去重：检查当前 `docs/`、campaign、研究结论、`src/data`/`src/workflow`，并用 `git ls-tree` 补查未展开的已跟踪路径；关键词为 FTD、fails/failure-to-deliver、RegSHO、threshold-list、未交割。未找到等价已执行 FTD/threshold-list 实验；命中主要是 FINRA 源的 RegSHO 路径及讨论链接。这个限定检索不是全历史穷尽证明，不重开 FINRA q。

社区入口与公开时钟：

- [QuantConnect Short data](https://www.quantconnect.com/forum/discussion/13119/short-data/)询问 FTD/平仓数据，回复未提供相应 LEAN 数据；没有可复用的发布时点交易检验。[Reddit FTD 工具讨论](https://www.reddit.com/r/Superstonk/comments/1dbes08/)则从结算日推算 T+x；这只是线索，未解决公开时钟或聚合余额账龄。
- [SEC 官方字段与说明](https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data)：自 2009-07 起每月两份，前半月通常月末公开，后半月约次月 15 日公开，**不保证特定日期**。字段为 settlement date、CUSIP、symbol、quantity、description、price，没有逐文件实际发布时刻；price 是前日收盘参考值。余额混合旧失败、新失败和已结清量，无法识别账龄或主体，长卖和短卖均可产生失败。完整源文件缺行与文件缺失必须分开处理，不能把下载缺口当作零余额。[SEC FAQ](https://www.sec.gov/about/webmaster-frequently-asked-questions)也只提供大致更新安排。
- [ChartExchange 发布历史](https://chartexchange.com/docs/ftd-release-history/)区分 Expected、File Time、Retrieved。其 2026-09 后半月示例列 File Time 为 10-01、Retrieved 为 10-15；这提示文件时间不可直接当公开时间。本轮没有验证其历史抓取回执、版本哈希或时区，不能据此认证完整发布钟。

原研究与反证：

| 来源 | 实际检验与适用边界 | 本候选裁决 |
|---|---|---|
| [Evans 等：Operational Shorting and ETF Liquidity Provision，作者稿](https://www.sec.gov/comments/s7-08-22/s70822-20149120-316297.pdf)，第 14–15、25–26 页 | 用 TAQ 逐笔买卖方向和 ETF 份额创建差异构造 operational shorting；次日 ETF 市价反转较集中于流动性错配较大的 ETF，NAV 没有相同预测力。FTD 是机制验证量之一。 | **不是 SEC 半月 FTD 文件真实公布后的策略**。不能把文中次日结果移到迟到的公开余额；原构造还需要当前未具备的逐笔与历史份额资格。 |
| [Devos 等：REIT 的 fails-to-deliver 公告研究，作者机构摘要](https://scholarworks.utep.edu/econ_papers/23/)，[期刊原文入口](https://link.springer.com/article/10.1007/s11146-013-9438-8) | 研究 REIT 的公告，发现公告影响收益的证据很弱；不是 SOXL/SOXX 或液态 ETF 的 SEC 半月公开文件检验。本轮核到摘要，未取得全文具体事件时钟。 | 保留反证；不能从个股/REIT 公告直接确定本账户的方向、期限或动作。 |
| [GetCoattail：FTD spike 分析](https://getcoattail.com/research/ftd-signal)，页面标 2026-10-15、revision history 标首次公开 10-17 | 方法明确使用 **settlement +14 calendar days** 后首个收盘作 publication proxy，非实际发布日志。单独 FTD spike/持续余额未见可靠差异；与 short interest 同时上升的结果来自 24 个比较格中的一个。混合股票/ETF、无交易成本。 | 即使标题和部分措辞使用 publication，也未通过真实公开时点门；混合信号不能证明单独 FTD 对液态半导体 ETF 的增量。本轮未下载其事件 CSV，亦未独立复现数值。 |

本轮未找到同时满足“真实公开日/时刻起点、液态 ETF 后续收益、可事前唯一映射 SOXL/SOXX/CASH”的证据。聚合余额不能给出某参加者的具体强制回补时刻；据结算日机械添加 T+35 等日数的路线在此直接拒绝。交易所 threshold list 是另一种发布产品，本轮不扩展为第二机制。

以后重议的最小输入是：①每份历史文件的可审计已公开/已接收时刻、当时字节与哈希、修订关系；只有日期时使用该日期结束后的下一合法决策，不能自行套固定 14/30 日滞后。②SOXL/SOXX 的历史证券身份、拆股与完整文件覆盖；任何归一化分母必须当时可得。③明确以公开后为起点、区别于原研究当日/次日机制量的液态 ETF 证据或单一可证伪解释。若机制必须识别失败账龄、主体或借券约束，公开聚合 FTD 自身不够；不因此购买 borrow/options 数据或恢复个股主线。

这些条件目前未满足，因此**不授权全历史采集、特征、训练或账户**，也不预设有益方向。资格失败不等于已经回测证明 FTD 无效，更不改变持续 Goal 的状态。
