# 交易日历来源资格：首次缓存执行的部分结果

本次唯一source notebook执行64428已exit0，读取12份成功cache，保留累计17次显式请求receipt；0新增source HTTP、0价格/收益读取、0TOM特征、0拟合、0账户。允许的NAS证据归档已完成。

生成23796条元数据（3966个signal×NYSE/NYSE_ARCA/NASDAQ×09:29/09:46）、297条带来源断言。原生Dataset保存后直接prepare一致；946个outer signal与944个account日期映射均保留，2026-10-08/09两个非账户尾行没有补造可执行日期。原3014成熟训练mask不是本source阶段的验证对象。

**12cache / 10parsed / 2parsergap / 5rawgap**。7份待处理来源中，4份SEC403和1份BusinessWire超时为原文缺口；另两份ICE年历原文完整，但解析器拒绝`date (July 4 holiday observed)`这一合法语法，不能据此宣称2018–2023不存在公告。具体source span与拒绝原因见[parse_gap_diagnostic.json](parse_gap_diagnostic.json)。初次Recorder和NAS完整保留，拟议窄修不是经济参数搜索，真实第二次解析仍需另行审查和GO。

|venue|clock|metadata rows|conditional snapshot|unknown|PIT qualified|
|---|---|---:|---:|---:|---:|
|NASDAQ|original_0929|3966|0|3966|0|
|NASDAQ|research_0946|3966|0|3966|0|
|NYSE|original_0929|3966|1916|2050|0|
|NYSE|research_0946|3966|1916|2050|0|
|NYSE_ARCA|original_0929|3966|1916|2050|0|
|NYSE_ARCA|research_0946|3966|1916|2050|0|

NYSE和NYSE_ARCA在两个clock的条件覆盖计数相同，因为公告只有日精度；这不等于09:46当前开市确认已在09:29可知。两clock的首收均未认证，09:46仅保留显式条件假设，原训练收益起点兼容性仍未解决。NASDAQ例外公告不构成普通完整年历，不能拿NYSE/Arca年历填补。

年度表按**signal year**分组。NYSE/Arca每clock：2014–2016各252行conditional；2017为230/251；2018–2021暂为0；2023为250/250、2024为252/252、2025为233/250、2026为193/194。2013和2022各有1行conditional，是年末signal映射到下一决策年份，并非早年日历被回填。2025的17行缺口来自当时尚未公开的2026跨年日程；原文发布日不倒灌。

完成2条已知普通日程与最终观察日期的差异记录，完整记录留Recorder。它们仅为来源诊断，不证明各venue等价，也不允许回溯修正信号。全部年份/月份coverage、unknown和原始断言均保留，不择区间。

本结果仍是conditional_snapshot，不是PIT或前向证据；接收venue未选，临时闭市公告完整性、历史vintage、早年及修订原件缺口均未消失。不晋升，不启动任何feature或模型。下一步仅处理确定解析缺陷并独立审查修复后的来源资格。

首次Recorder：`638ecf0af9204ba38e2e36e7be0307f1`；NAS archive：`0f77e058238f7a28014f47b267a739df68881b1317a61110f97902054783035a`。38个artifact身份见verification_manifest.json。

独评已接受首次部分来源结果，确认两处为确定解析缺陷；仅进入有界解析修复，尚未批准特征或模型。评审：[source_result_review_initial.json](source_result_review_initial.json)。
