# FT-900 半导体出口：三份原始月报来源预检

在冻结的三个官方 PDF 中，Exhibit 7（美国货物出口，按最终用途类别和商品）均有单列 **Semiconductors** 行。2011-12 月报于 2012-02-10 08:30 EST 发布，出口当月/上月为 3,438/3,503 百万美元；2017-05 月报于 2017-07-06 08:30 EDT 发布，为 3,862/3,865；2024-04 月报于 2024-06-06 08:30 EDT 发布，为 4,856/4,492。数值与页码在 `source_preflight_result.json`。2011 表中印出的变动 -64 与两个已四舍五入的表值之差 -65 相差 1，应保留原件，不自行改写。

三个 PDF 原始字节各读取一次，SHA-256、尺寸、HTTP 收据保存在 `capture_receipts.json`；NAS 打包存档及远端哈希核对见 `nas_binding.json`。本卡没有读取 SOXX 结果、拟合模型或运行账户。

后续只读取了一次官方 FT-900 历史索引并封存原字节。2011–2022 共 144 个目标月份，2019-02 的官方标题误拼为 `Ferbruary 2019`，其 PDF 链接可由年份分区和邻月核对。**更正前一版结论：144 个标题中只有 143 个索引 PDF 链接指向相应月份；“August 2015”误链到 `ft900_1509.pdf`。** `index_link_correction.json` 保留此前 144/144 声称及逐链接重新核对结果。不能把这个错误链接算作 2015-08 的内容证据；当前这张卡仅收到两个训练期 PDF，且未证明历史首次公开版本。索引原件已作 NAS 哈希读回。

这证明一个与 G.17 国内产量不同的**出口需求来源候选**，尚不证明它能预测收益。当前下载的历史 PDF 不能单独证明其字节就是当时首次公开的版本；2012 年最终用途分类调整及后续修订使今日 API 的长历史不能直接充当逐月 PIT 原件。下一步如继续，先限定核查 2011–2022 每份当期 Exhibit 7 的行覆盖、当期值/修订和首次公开时钟；再与 Stage1 原 39 输入做去重，并事前冻结信息门。只有信息门通过，才允许独立完整账户按 0bp、IBKR Tiered 费用先检验，SOXX 仍是最终硬门。

来源采集进展：已按冻结的 144 个月份 URL 映射取得 **144/144 份训练期 PDF 字节**（复用既有 4 份、新 GET 140 次）。每年 12 份构成一个 NAS 年包，12 包远端 SHA-256 全部一致；另从每个年包实际恢复 1 份 PDF 并核对尺寸与 SHA-256，144 份本地 PDF 均按请求收据复核。账本见 `full_training_request_ledger.jsonl`、`full_training_year_archives.jsonl`，限定恢复见 `full_training_readback.json`。本地源字节继续保留。**字节覆盖不等于**逐月 Exhibit 7 行、历史首次公开字节、2012 跨分类可比性或 Stage1 因子资格；尚未构造因子、拟合或运行账户。

随后只读原始 PDF 的固定行审核在 **144/144 个月份**找到唯一的 Exhibit 7 半导体出口数值行和首页发布标识，详细页码、当前及同版上月值、印出的月变化与原 PDF 哈希在 `full_training_row_audit.json`。此项已解决来源字段行覆盖；仍须核对每份报表的月份身份、历史首次公开字节与 2012 分类连续性。没有 Qlib 因子、模型或资金账户。

报表月份身份限定核查：143 份首页标题直接匹配目标月份；唯一例外为 2022-12 的年终合并标题“December and Annual 2022”。其精确 PDF SHA、2023-02-07 08:30 EST 发布标识、Exhibit 7 当月栏和官方索引 URL 经单例合同核对后，**144 份报表月份身份均有证据**。严格初次匹配失败与单例修复分别保存在 `full_training_report_identity.json`、`report_identity_exception_result.json`。这仍不证明当年首次公开的 PDF 字节未被后续替换。

留出期来源采集：已从同一官方索引确认 2023-01 至已发布的 2026-08 共 44 个正确月份 PDF URL；复用 2024-04 原件 1 份并新增只读 GET 43 次，**44/44 份 PDF 字节和本地 SHA-256**均核对。2023–2025 三个全年包及 2026 年 1–8 月包共 4 个 NAS 包远端哈希通过，且各恢复 1 份源 PDF 匹配 SHA 与尺寸。收据和限定恢复见 `held_request_ledger.jsonl`、`held_year_archives.jsonl`、`held_readback.json`。这些材料来自反复使用的历史区间，不是独立策略前向日；留出期逐月 Exhibit 7 行及报告身份仍待审核。

官方原件：[2011-12 FT-900](https://www.census.gov/foreign-trade/Press-Release/ft900/ft900_1112.pdf)、[2017-05 FT-900](https://www.census.gov/foreign-trade/Press-Release/ft900/ft900_1705.pdf)、[2024-04 FT-900](https://www.census.gov/foreign-trade/Press-Release/ft900/ft900_2404.pdf)；[历史发布索引](https://www.census.gov/foreign-trade/Press-Release/ft900_index.html)、[2012 分类变动说明](https://www.census.gov/foreign-trade/Press-Release/ft900/ft900_1202.pdf)。
