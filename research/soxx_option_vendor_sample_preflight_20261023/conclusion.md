# SOXX 期权偏斜候选：Cboe 免费样本来源预检

按冻结预算只下载 Cboe 官方 [Option Quotes](https://datashop.cboe.com/option-quote-intervals) 与 [Option EOD Summary](https://datashop.cboe.com/option-eod-summary) 两份公开样本，共 2 次 GET、0 购买。原始 ZIP 分别为 2,628,937 和 2,952,557 字节，响应头、ZIP 原字节及 SHA-256 收据见 `sample_receipts.json` 和 `sample_manifest.json`。首轮 NAS 上传通过，但程序误以为回执含 `nas_path` 而未做回读；按冻结机械修复从仓库 NAS 归档根与哈希推导路径，重新归档并限定恢复成功，归档 SHA-256 为 `ba88ddde28a5878107c6ef48d9a76f0149270b369ed80ba09e9ce4466424cf8f`。没有重复网页请求。

Option Quotes 样本含五个内层 CSV 变体；带 Calcs 的表头有报价时间、bid/ask、隐含波动率和 Delta，样本首行标的是 AAPL 或 ^SPX。EOD 样本有 15:45 报价、IV/Delta 和日终字段，首行是 SPY。样本仅证明字段格式，不证明 SOXX 个股期权链覆盖。两份产品页目前均声明历史起点为 **2012 年 1 月**，不能据此前 FAQ 的宽泛“2010 年”说法填补冻结 Stage1 的 2011 年训练缺口。Option Quotes 日内文件延迟 15 分钟，EOD 文件隔夜或次晨交付；冻结 Stage1 的 09:29 决策只能考虑**此前交易日**的合格报价，并须证明它在该时刻前已经送达。

**决定：`needs_SOXX_specific_historical_quote_sample_and_2011_gap_resolution`。** 本卡没有读取 SOXX 收益、构造 25 Delta 恒定期限偏斜、拟合模型或运行账户。下一步先取得供应商对 SOXX 在 2012、2017、2022 年原始期权链的有权限样本和报价/交付时刻，并确定 2011 年只能保持未知还是有另一个合格来源；购买前还要核对报价完整度、拆分后的合约身份、IV/Delta 正负及缺失率。不得从 AAPL/SPX/SPY 通用样本推断 SOXX 可用，更不能声称超过 SOXX。
