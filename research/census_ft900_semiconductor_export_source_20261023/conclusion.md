# FT-900 半导体出口：三份原始月报来源预检

在冻结的三个官方 PDF 中，Exhibit 7（美国货物出口，按最终用途类别和商品）均有单列 **Semiconductors** 行。2011-12 月报于 2012-02-10 08:30 EST 发布，出口当月/上月为 3,438/3,503 百万美元；2017-05 月报于 2017-07-06 08:30 EDT 发布，为 3,862/3,865；2024-04 月报于 2024-06-06 08:30 EDT 发布，为 4,856/4,492。数值与页码在 `source_preflight_result.json`。2011 表中印出的变动 -64 与两个已四舍五入的表值之差 -65 相差 1，应保留原件，不自行改写。

三个 PDF 原始字节各读取一次，SHA-256、尺寸、HTTP 收据保存在 `capture_receipts.json`；NAS 打包存档及远端哈希核对见 `nas_binding.json`。本卡没有读取 SOXX 结果、拟合模型或运行账户。

后续只读取了一次官方 FT-900 历史索引并封存原字节。2011–2022 共 144 个目标月份，143 个月份标题直接匹配；2019-02 的官方标题误拼为 `Ferbruary 2019`，其 2019 年分区、`ft900_1902.pdf` 链接及相邻月份独立核对后，得到 144/144 个**官方 PDF URL 绑定**。这不是 144 份 PDF 内容覆盖，更不是历史首次公开版本资格。索引原件已作 NAS 哈希读回。

这证明一个与 G.17 国内产量不同的**出口需求来源候选**，尚不证明它能预测收益。当前下载的历史 PDF 不能单独证明其字节就是当时首次公开的版本；2012 年最终用途分类调整及后续修订使今日 API 的长历史不能直接充当逐月 PIT 原件。下一步如继续，先限定核查 2011–2022 每份当期 Exhibit 7 的行覆盖、当期值/修订和首次公开时钟；再与 Stage1 原 39 输入做去重，并事前冻结信息门。只有信息门通过，才允许独立完整账户按 0bp、IBKR Tiered 费用先检验，SOXX 仍是最终硬门。

官方原件：[2011-12 FT-900](https://www.census.gov/foreign-trade/Press-Release/ft900/ft900_1112.pdf)、[2017-05 FT-900](https://www.census.gov/foreign-trade/Press-Release/ft900/ft900_1705.pdf)、[2024-04 FT-900](https://www.census.gov/foreign-trade/Press-Release/ft900/ft900_2404.pdf)；[历史发布索引](https://www.census.gov/foreign-trade/Press-Release/ft900_index.html)、[2012 分类变动说明](https://www.census.gov/foreign-trade/Press-Release/ft900/ft900_1202.pdf)。
