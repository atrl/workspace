# FOMC 事前信息：去重、时钟与作者更新

2026-10-18。裁决：**本轮不推荐新的 Stage1 候选**。预公告固定窗口已失败封存；更长的政策周期虽非同一个窗口，
但作者更新给出反向周期，且尚未建立与原 H1 接收窗口和全期已知日程相匹配的输入合同。
不重开旧路线，不翻转信号，不选择会议子集或事件窗口。当前日夜监督实验的未审账户结果未用于本判断。

本轮使用 3 组、8 条搜索及有限原文/官方页面只读浏览；无市场数据集、作者数据/代码 ZIP 下载，无本地价格、标签、
预测对象读取或计算，无特征、fit/account。仅修改本笔记。

## 已测试的范围必须说准确

[fomc_preannouncement_20261014](../fomc_preannouncement_20261014/conclusion.md)实际固定窗口是
**前一原生交易日 09:47 → 会议日 13:59**，不是“会议日早盘到午后”。它对 61 场已预告会议比较相同股票总权重下的
SOXL 倾斜与 SOXX 对照，并设提前 5 session 的日历对照；全成本的净时点区间与年度门失败。
该卡明确禁止改 entry/exit、方向、权重和对照日期救援。把相同会议 flag 接入 NN 本身不能证明出现了新的机制。

[既有日程来源卡](../fomc_schedule_source_20261014/conclusion.md)已取得 2019 至 2026-10-09 的 61 场预告链，
依据上一份已公开纪要的下一会议日期及官方索引的 release date；另保留少量 2018 预告和 2020 年临时替代/取消记录。
这些来源可复用为研究基础，不能将来源通过说成 alpha 通过，也没有覆盖原训练 cohort 的全部 2011–2018 年已知状态。
本轮没有重读其价格或账户对象。

## 社区入口与原始证据

[r/LETFs 的 TQQQ pre-FOMC 讨论](https://www.reddit.com/r/LETFs/comments/1rz1r25/i_backtested_a_simple_tqqq_prefomc_overnight/)
采用“前日收盘买、次日开盘卖”，明确未扣费用。这里只作为发现入口：其标的、时钟、费用与本项目不同，社区结果不是成功证据。

| Primary 来源 | 本轮核实的关键事实及限制 |
|---|---|
| [Lucca/Moench 原稿，NY Fed Staff Report 512](https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr512.pdf)，§3、PDF 第 8 页 | 主要 pre-FOMC 窗口为前日 14:00 至公告日 14:00；原样本通常 14:15 公告，故排除了政策结果。不是开盘到开盘标签，也不是 SOXL 对 SOXX 净超额证据。 |
| [原作者 2018 更新](https://libertystreeteconomics.newyorkfed.org/2018/11/the-pre-fomc-announcement-drift-more-recent-evidence/) | 2011-04 至 2018-06 样本效应集中在有记者会的会议，收益时间向前移动。2011–2012 记者会会议的声明曾在 12:30 发布，非记者会会议为 14:15；2013 起声明统一为 14:00。不能给所有年代套一个发布钟。 |
| [Cieslak/Morse/Vissing-Jorgensen 作者稿](https://faculty.haas.berkeley.edu/morse/research/papers/cycle_paper_cieslak_morse_vissingjorgensen.pdf)，§2、PDF 第 5–6 页 | 研究会议间的周 0/2/4/6，而非只看公告前一天。日号排除周末、假日收益置零；它是 weekday 轴，不是删掉所有休市日后的 Qlib session 轴。原研究的平均正效应不能直接变成持续正收益假设。 |
| [Knox/Vissing-Jorgensen 更新，Fed FEDS 2026-023](https://www.federalreserve.gov/econres/feds/files/2026023pap.pdf)，稿内日期 2025-08-25，§5.2/6.2 | 将 pre-FOMC 图延至 2023，仍讨论隔夜分量，所以“所有效应永久消失”过强；但 **2017–2021 的 FOMC 周期显著反向**。作者认为固定正风险溢价不足以解释周期，意外政策消息可能长期偏好或偏坏。本项目不能据此反号或挑样本。 |

[原周期论文的出版方 Supporting Information](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12818?af=R)
列有 `jofi12818-sup-0002-ReplicationCode.zip`。本轮只确认入口，未下载或核验其数据授权、可运行性和时点实现；
不能把作者代码存在当成本项目历史 PIT 已完成。

## 原账户时钟不等价

令 D 为公告日。下面是时钟推导，不涉及读取或计算市场收益：

| 接收区间 | 与文献预公告窗口的关系 |
|---|---|
| 原 H1：D 前一交易日 09:47 → D 日 09:47 | 含前日更多早盘，却漏掉 D 日 09:47 至约 14:00 的会前时段。即使覆盖隔夜，也不是原 24 小时窗口的复制。 |
| 原 H1：D 日 09:47 → 下一交易日 09:47 | 跨入政策声明、记者会及其后隔夜，无法标为纯 pre-announcement。09:29 时不能知道下午政策内容或 surprise。 |
| 已封存事件卡：前日 09:47 → D 日 13:59 | 与文献及 2018 更新的较早累积时段高度重叠；失败后不得改到前日收盘、D 日开盘或其他终点找胜者。 |

因此，已知日期可以是事前上下文，但“知道有事件”和“知道当前 H1 条件收益更高”是两个命题。
原 allocation 目标仍是 H1，tail/alpha 为 H5，不能把事件论文的区间收益直接当作它们已获验证的监督。
对于周期，weekday 奇偶周也不能直接套作原生交易日计数；本轮没有选择一种移植方式。

## 公开来源可取得，完整 historical-known ledger 尚未取得

[Fed 官方日程](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm)说明每年通常八场预定会议、也可另行开会；
计划日期在前一场会议确认前仍是 tentative，常规纪要一般在政策决定后三周发布。
当代完整日历与最终实际会议日期均不是逐日历史 knowledge snapshot。

本轮公开、无认证读取了 [2011 历史索引](https://www.federalreserve.gov/monetarypolicy/fomchistorical2011.htm)、
[2010-12-14 纪要](https://www.federalreserve.gov/monetarypolicy/fomcminutes20101214.htm)和
[2011-08-09 纪要](https://www.federalreserve.gov/monetarypolicy/fomcminutes20110809.htm)。前者给出纪要 release date；
后两者分别写出下一次会议安排，后者还明确九月会议扩为两天。
这是早期来源可行性的例子，不是 2011–2018 全覆盖认证。会议发生日、纪要公开日、修订获知日和本次抓取日必须分别保留。

2019 起每场预定会议有记者会的制度变化，是 [2018-06-13 主席记者会原文第 2 页](https://www.federalreserve.gov/mediacenter/files/FOMCpresconf20180613.pdf)
已公开宣布的变化，不应按事后表现划分样本。既有卡的 2020 替代会议处理保留：临时会议不能回填成早就预定的事件，
取消消息也只有在决策钟之前已公开才可改变当次计划状态。当前页面存在 emergency meeting 条目，不证明它事前可预测。

若未来有独立的新机制，最低来源合同仍包括：完整拟研究日期范围的公告/确认/修订/取消链、每条发布证据与保守知识钟、
scheduled 与 emergency 明确区分、09:29 已知性、weekday 与实际 session 的独立定义，以及未知日期不补已知、不删训练行。
不要求把历史研究伪称原始 first vintage；只能支持条件性复原时须如实保留限制。

## 为什么不是 TOM 换窗口，以及为何本轮放弃

政策信息不确定性、注意力集中或会议间消息传递，在机制上不同于月末资金结算/纯交易日历。
预定 FOMC 日期也不是原 16/39 价格字段的确定性变换。但“官方日期是额外测量”仍不证明有增量可预测收益：
广泛股票风险溢价可能已由持有 SOXX 获得；要支持 SOXL/CASH 调整，还须识别相对机会并补偿费用与尾风险。
日历本身不观测未来政策消息的方向，也不能解决原长期现金偏高。

本轮不把会前事件换成新的正向杠杆规则，也不因周期反向改学一个负号。已失败的预公告版本保持关闭；
本轮检索的 campaign、卡名和静态代码中未找到长周期版本的等价已执行记录；但作者反证和接收时钟差异使其不足以成为当前下一项优先训练机制。
最小新证据应是事前可定义、公开可得的政策信息状态及其清楚的接收区间，或独立新增的同口径证据，
而不是在既有历史中选奇偶周、记者会类别或窗口。并不要求先证明经济因果才允许研究，但不能只靠泛称
“政策不确定性”把重复日历试验变成新假说。

若以后得到这样的实质变化，仍须另立唯一有限配对设计，完整继承费用后胜 SOXX、匹配控制和强参考、
MaxDD/CVaR、年度/126 滚动、配对区间与充分独立前向门。当前没有模型、来源采集、拟合、账户、交易或付费授权。
