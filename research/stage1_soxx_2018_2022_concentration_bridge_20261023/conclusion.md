# SOXX 2018–2022 原始披露集中度 Qlib 来源衔接

新增 2018-09-30 原始 SOXX N-CSRS 的 30 条已闭合普通股值，复用前版 15 个原始旧表单及 NPORT-P 事件。按 SEC **申报日之后**的下一个实际 Qlib 交易日映射，2018 年这份报告于 **2018-12-10** 生效；它在 2019 年首个 Stage1 训练日已可用，先前未知的 41 日由该来源覆盖，而非事后回填。

2019–2022 的 **1,008 个**原 Stage1 Qlib 交易日均有一份此前已申报的**披露普通股集中度来源**，16 个原始报告事件的 accession、报告期、生效日各不重复，来源最大申报日龄 **171 天**。该覆盖只证明前五股占比与 HHI 这两个滞后披露代理的来源可得；它们不是 SOXX 官方逐日指数权重，也不证明 2011–2018 全训练覆盖、成分股价格、新低扩散、下跌集中度或增量预测能力。

原生 Qlib DatasetH 与 Recorder `b9f1a84df25846dc8bc45b2e95906b31` 读回通过；Yanxuan NAS Recorder 归档 SHA-256 `b73b8d47847fa71b8340a9759dfa4d958dfb3858ed9cd56f7c3e87f19201189b` 经远端哈希与限定恢复核验。新增 0 HTTP 请求、0 市场收益读取、0 拟合、0 原生账户。决定：`2019_2022_original_reported_concentration_late_train_source_coverage_complete_earlier_Stage1_pending`。
