# 日历来源解析repair_1：固定缓存执行结果

一次获批repair source notebook执行65407已exit0。只修复已明确的leadingweekday/date+括号nominal holiday observed语法；原文、knowledge_at、venue与缺失规则均不变。12cache全部12parsed，0parsergap；原4份SEC403和1份BusinessWire超时仍为5rawgap。没有任何新增source HTTP、TOM特征、价格/收益读取、fit或account。

新建439条来源断言、23796条元数据（3966个signal×3venue×2clock）；946outer/944account/2tail日期映射及native Dataset直接prepare读回通过。原3014成熟训练cohort仍是后续label-aware门，本来源阶段未重建或改变。

|venue|clock|metadata rows|条件ordinary日程覆盖|conditional snapshot|unknown|PIT qualified|
|---|---|---:|---:|---:|---:|---:|
|NASDAQ|original_0929|3966|0|0|3966|0|
|NASDAQ|research_0946|3966|0|0|3966|0|
|NYSE|original_0929|3966|3176|2925|1041|0|
|NYSE|research_0946|3966|3176|2925|1041|0|
|NYSE_ARCA|original_0929|3966|3176|2925|1041|0|
|NYSE_ARCA|research_0946|3966|3176|2925|1041|0|

ordinary日程覆盖只是已载入公告的条件覆盖，不能当完整修订/临时闭市来源。NYSE/Arca每clock的3176行ordinary基线中，251个2022决策日因缺少Juneteenth修订原件继续保留unknown，因此conditional只有2925行。NASDAQ仅有特定例外公告，没有本corpus内完整普通年历，不能借用NYSE/Arca日程。

年度表使用signal_year：2014–2019对应NYSE/Arca每clock均为条件完整；2020为234/253；2021为251/252；2022为1/251；2023为250/250、2024为252/252、2025为233/250、2026为193/194。2013的1行及2022的1行来自年末signal映射到下一决策年，不是历史回填。2020/2025的跨年覆盖缺口保留原公告日期；2026未知尾行保留。所有年份与月份未筛选。

来源日历差异共6行：2018-12-05、2022-06-20、2025-01-09各有NYSE和NYSE_ARCA一行，已知普通表给出planned weekday open，但最终观察日历没有该日期。这明确暴露Bush、Juneteenth和Carter的缺失venue公告/修订来源；不能用观察到的闭市倒造当时公告，也不能将NASDAQ事件当NYSE/Arca证据。原记录见Recorder analysis/discrepancies；[小型原样摘录](discrepancy_summary.json)。

09:29/09:46的日程coverage数量相同，只因为现有公告采用日精度次午夜约定；current-open首收在两clock均未认证。09:46的条件开市假设不能被搬到09:29或原09:30起点收益。历史vintage、原始发布时刻及receiver venue仍未解决，historical_PIT和独立前向资格均为false。

本次使用独立repair_1 Recorder和NAS归档。父卡首次38artifact、attempt、plan、GO、summary和NAS身份逐项hash保持不变；没有覆盖或隐藏首次两篇解析拒绝。修复成功只解决解析缺陷，不构成TOM有效性、来源全覆盖或模型晋升。下一步为独立来源结果评审，任何feature/接收任务须另行冻结。

新Recorder：`82b3305714cc455db5bc2ccbcd42d2f1`；38artifacts；NAS：`cca1821148944e014df02352ac71989a2fdb350f851d403f444bb2f733d451f8`。首次Recorder：`638ecf0af9204ba38e2e36e7be0307f1`。

交付冷读由root唯一执行（21776 exit0）：38artifact SHA、23796×33 metadata Dataset直接prepare、12parsed/5gap/439assertions、946/944/tail及knowledge≤clock核验通过，PIT仍false。见[readback_verification.json](readback_verification.json)。这是交付/一致性检查，不替代独立来源资格审查。

独评已接受这份条件来源结果，仅允许另行设计单一预开市接收器；特征、模型、账户及晋升均未获资格。详见[source_result_review.json](source_result_review.json)。下一步固定09:29可知的NYSE计划与d之前共同观测session前缀，重新定义因果mask，不能直接沿用本表含当日实际开市的qualification字段。
