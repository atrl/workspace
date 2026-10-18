# 隔夜／日内分解：监督任务资格

2026-10-18。**有一个与旧路线不同的监督假说，但尚不具备直接拟合资格**：保持原输入、交易时钟和主目标，只增加“同一H5内SOXL相对SOXX的隔夜−日内回报差”监督。它是归纳偏置，不是新外部信息，也没有解决长期高现金或稳定胜SOXX的实证。状态：`needs_train_only_label_contract_and_matched_supervision_design`。本轮4组web发现/原文读取、12条搜索查询；0真实行情/收益/特征/标签构造、0fit/account、0批量下载。主线TOM不变。

## 先去重与校准原目标

| 已有路线 | 不得重开或混同 |
|---|---|
| [night_session_signal](../night_session_signal_20261010/conclusion.md) | 原夜盘数值已因小时bar跨过09:30的前视撤回；不作为本假说的任何正证据。 |
| [NQ严格10:31后检验](../stage1_nq_overnight_after1031_20261010/conclusion.md) | 已关闭固定NQ18:00→10:00信号、10:31之后持有路线，不改期货窗口或反号。 |
| [opening_state](../opening_state_training_20261015/conclusion.md)、[opening_interaction](../opening_interaction_training_20261015/conclusion.md) | 已测试当日隔夜gap、09:30–09:45收益及二者交互，完整账户失败；不能重命名这些观测输入。 |
| [full_day_expansion](../intraday_full_day_expansion_20261012/conclusion.md)、[ROD](../stage1_close_rod_20261018/conclusion.md) | 已含晚段显式隔夜标签，及另行失败的尾盘短持有机制；不得把“预测隔夜”泛称首次尝试。 |
| [domain_heads](../stage1_domain_heads_20261018/conclusion.md) | 已失败的是主/辅助资产域分头。本候选若有新意，只能来自同一目标时间段的分量监督，不是再换分头名称。 |

限定检索campaign、上述结论和`src`未找到“保持原配置/分类目标，配对增加H5日夜**差分监督**”已经执行的记录；不是全仓历史穷尽证明。原目标的确切静态源码链如下，**没有读取实际label数组**：

- **allocation收益H1**：[`soxs_defensive_hurdle._load_feature_frame:134`](../../src/workflow/soxs_defensive_hurdle.py#L134)以`open.shift(-2)/open.shift(-1)-1`构造forward_open_return；[`_load_reintegrated_frame:135`](../../src/workflow/cash_floor_relative_feature_reintegration.py#L135)调用该loader。[`direct_continuous_soxx_allocation._build_dataset:270`](../../src/workflow/direct_continuous_soxx_allocation.py#L270)的target_timing明确写T+1开盘→T+2开盘。
- **tail/cash_target H5**：[`_attach_direct_targets:161`](../../src/workflow/direct_continuous_soxx_allocation.py#L161)把上述SOXX单日收益shift0…4后cumprod，取入场相对路径最差点；第164–166行构造tail_event及tail_cash_target。不是逐笔最大回撤或单日tail。
- **alpha H5**：[`_attach_relative_targets:315`](../../src/workflow/cash_floor_relative_feature_reintegration.py#L315)定义daily-reset27%SOXL/63%SOXX/10%cash参考；第324–339行取五个H1 log收益和减去五个SOXX log收益和。不是单纯SOXL五日收益减SOXX。
- **实际接收与共同成熟筛选**：[`MinutePitAllocator.fit:154`](../../src/model/minute_pit_allocator.py#L154)把前两列单日returns与tail/alpha/cash拆开，第168–174行分别用于allocation_objective与主BCE；[`fit:127`](../../src/model/minute_pit_allocator.py#L127)先统一要求label_available_at非空且严格小于cutoff、原index≤train_end、全部五个TARGETS有限，才得到同一个train。H1的存在不允许提前其共同成熟钟、放宽H5完整性或扩大原3014cohort。allocation本身另含年际/路径风险项，不等于单日收益MSE。

这只是新文档措辞校准：继承原数组、数学与共同成熟筛选，旧封存卡不改；不构成新标签或新机制执行授权。

## 社区入口与primary边界

[QuantConnect Overnight Anomaly](https://www.quantconnect.com/research/15296/overnight-anomaly/)给出SPY收盘买、开盘卖的公开示例，并明确提示高换手成本会抵消收益；[其MOO/MOC讨论](https://www.quantconnect.com/forum/discussion/14166)提醒拍卖成交与普通报价spread的模型区别。[r/quant反例讨论](https://www.reddit.com/r/quant/comments/1nx7bbk/can_you_really_trade_overnight_mean_reversion/)指出开盘端点混用可能把预测变量变成响应变量。这里只采纳问题线索，不认证社区回测、显著性或交易建议。

- [Lou/Polk/Skouras，JFE2019《A tug of war》作者原文](https://personal.lse.ac.uk/polk/research/TugOfWar.pdf)，§3–4：股票截面及策略的隔夜、日内分量可能持续且互相抵消，并讨论投资者群体差异。主开价是**首半小时09:30–10:00 VWAP**；脚注另报告CRSP开价、首笔成交和开盘midquote稳健性。研究主要是1993–2013股票/月度组合，不是SOXL/SOXX的H5监督或09:47净账户。其CRSP市场组合两分量并非一律一正一负，不能转述成“所有市场收益都在夜间”。
- [Hendershott/Livdan/Rösch，JFE2020《Asset pricing: A tale of night and day》作者原文](https://faculty.haas.berkeley.edu/hender/CAPMday-night.pdf)：beta与夜间、日内平均收益的关系不同，支持分量异质性；并不证明能用现有39字段预测这种变化，更不能直接将股票截面beta结果映射为杠杆ETF配置。
- ETF更直接的线索是Liu/Zhang/Zhang [《A hidden cost of ETF investing》](https://doi.org/10.1016/j.jbankfin.2025.107621)，[作者机构论文材料第一章](https://openresearch-repository.anu.edu.au/server/api/core/bitstreams/49a75f9a-b09c-4064-b54b-9e79bb4afe33/content)的可检索摘要使用**midquote**分解，讨论零售需求与套利供给。本轮全文接口未成功读取，未核实SOXL、杠杆ETF样本、具体端点及费用结果；只保留摘要级机制线索，不据此宣称本数据可复制。

没有找到上述作者提供且本轮可核验的监督训练复现代码；QC示例不是这些论文的作者实现。这些论文也没有提出下面这项Stage1新辅助损失，后者是本项目待检验的推论。

## 唯一可区分的候选：固定H5分量差监督

令d0为原09:29决策之后的第一个RTH开盘，d1…d5为随后五个共同交易日；O、C分别为**同一合法价格口径**的该资产开、收盘。对资产i定义：

`D_i = sum(j=0..4) log(C_i,dj / O_i,dj)`；
`N_i = sum(j=0..4) log(O_i,d(j+1) / C_i,dj)`。

于是 `D_i+N_i=log(O_i,d5/O_i,d0)`。只考虑一个新目标：

`Z = (N_SOXL-N_SOXX) - (D_SOXL-D_SOXX)`。

同端点总回报可以有不同Z，因此它暴露了原端点回报隐藏的实现路径差异。但这只是价格分量恒等式：**不是**原27/63/10参考组合alpha标签的精确分解，也不是含现金、Tiered费用、动态持仓和分红现金流账户的精确分解。不能用`3×SOXX隔夜回报`代替实际SOXL分量。

一个重要排除项是“分别跑相同X、相同正则/样本权重的线性Ridge，再把两预测相加”。同一线性估计算子A满足 `A*y_D+A*y_N=A*(y_D+y_N)`，与直接预测总回报等价。若换缩放/正则才有差异，不能把该差异描述为新增信息。两分量平方损失在sum/difference坐标中对应 `e_D²+e_N²=(e_sum²+e_diff²)/2`；所以可能非重复的干预是对共享表示提供**差分监督**，而不是相加后的另一条总回报预测。

若以后允许设计：B/C应同一个额外小监督头、相同输入/容量/初始化/训练预算；原H1配置损失、H5 tail/alpha及20741辅助资产任务均保留，B额外头监督固定H5总相对回报S，C额外头监督Z。只有标签分量不同；监督头不直接触发夜间买卖或新风险阈值。新头接入位置、损失系数和仅训练域缩放须事前完整冻结，不能靠PG、分域头、权重或窗口搜索救援。本轮未选这些实现参数，不授权训练。

## 数据和时钟门

现有原生日级SOXL/SOXX open/close原则上足以构造上述**标签**，无需期货、TAQ、options或新批量市场数据；但需要另行许可并封存训练域专用价格片段、公司行动口径与可用时点，不能反序列化含2023+收益的大对象后再称“未访问评价数据”。

1. d0开盘和此前一夜到d0的gap在09:29尚未知，不能作为当次输入。上式从d0开盘之后开始，**不把C前日→O_d0这一段纳入预测未来的H5分量**；所有未来O/C只能当成熟后的训练标签。
2. 新标签在O_d5及所有所需原值正式可用、且满足原H5成熟钟后才可训练。不得早于原成熟时间，或为了补缺失而缩短H5、跨缺口改成“下一可得开盘”。保持原主cohort；若无法同源提供所需标签则停止资格，不择已知年份训练。
3. O/C必须使用一致拆股/价格调整平面；不能将adjusted close除以raw open。需要保持价格回报与真实分红现金回报的区别；不能把除息跳空误当不可交易夜间损失。原有历史首次可用/PIT限制继续保留。
4. 日级open是原开盘代理，**不是09:47成交价**。日级分解可作为原09:29训练的辅助目标，但不能由此声称已经修复09:47账户标签；精确09:47分量另需分钟源且将重涉已失败执行对齐路线，本候选不这样扩展。论文10:00 VWAP也不能倒填09:29输入。
5. 不增加“日内尾损/隔夜尾损”阈值或状态路由：日级O/C不能认证夜间最大路径回撤，拆出两个风险state会同时改变监督定义与控制器。当前只允许讨论Z这个连续监督目标。

## 与Goal的联系、反证与停止条件

可能的链条是：相互抵消的分量携带不同条件信息→差分监督约束共享表示→原配置目标更好区分SOXL相对SOXX的机会与风险→少一些无谓现金。**每个箭头仍未验证**。现有输入未必能预测Z；日夜均值差也不等于时序可预测性；即便Z可预测，总收益可能仍相互抵消；原账户同时跨日内和隔夜持有，且内生风险状态可能阻断增量。不能通过预设“夜间高beta更赚钱”强制减现金。

最小下一步只应是被单独授权的训练域标签合同/代数机械资格：同源O/C、H5边界、成熟/缺失、分红口径、telescope恒等式与上述Ridge等价检查。不读评价收益、不先筛最优时段。若获后续完整设计批准，再固定同源同样本B/C、同seed/300epochs和0/2/5bp独立完整账户；所有预登记账户都执行，不能用中间AUC/MSE选择是否展示。最终仍须胜SOXX及匹配/强参考，满足全部回撤、CVaR、完整年度/126滚动、配对CI与独立前向条件。

标签无法在原cohort内严格构造、改动退化为已测试gap输入/同一线性总回报估计、需要换执行窗口或新阈值才能解释结果，均停止本候选。完整经济门失败也关闭固定版本；不能只因胜B、预测分量更准或现金下降就保留为成功。当前裁决是**有界待资格监督假说，非已证明alpha、非高优先级直接fit**。
