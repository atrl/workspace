# SOXX 期权偏斜候选：Cboe 免费样本来源预检

按冻结预算只下载 Cboe 官方 [Option Quotes](https://datashop.cboe.com/option-quote-intervals) 与 [Option EOD Summary](https://datashop.cboe.com/option-eod-summary) 两份公开样本，共 2 次 GET、0 购买。原始 ZIP 分别为 2,628,937 和 2,952,557 字节，响应头、ZIP 原字节及 SHA-256 收据见 `sample_receipts.json` 和 `sample_manifest.json`。首轮 NAS 上传通过，但程序误以为回执含 `nas_path` 而未做回读；按冻结机械修复从仓库 NAS 归档根与哈希推导路径，重新归档并限定恢复成功，归档 SHA-256 为 `ba88ddde28a5878107c6ef48d9a76f0149270b369ed80ba09e9ce4466424cf8f`。没有重复网页请求。

Option Quotes 样本含五个内层 CSV 变体；带 Calcs 的表头有报价时间、bid/ask、隐含波动率和 Delta，样本首行标的是 AAPL 或 ^SPX。EOD 样本有 15:45 报价、IV/Delta 和日终字段，首行是 SPY。样本仅证明字段格式，不证明 SOXX 个股期权链覆盖。两份产品页目前均声明历史起点为 **2012 年 1 月**，不能据此前 FAQ 的宽泛“2010 年”说法填补冻结 Stage1 的 2011 年训练缺口。Option Quotes 日内文件延迟 15 分钟，EOD 文件隔夜或次晨交付；冻结 Stage1 的 09:29 决策只能考虑**此前交易日**的合格报价，并须证明它在该时刻前已经送达。

**决定：`needs_SOXX_specific_historical_quote_sample_and_2011_gap_resolution`。** 本卡没有读取 SOXX 收益、构造 25 Delta 恒定期限偏斜、拟合模型或运行账户。下一步先取得供应商对 SOXX 在 2012、2017、2022 年原始期权链的有权限样本和报价/交付时刻，并确定 2011 年只能保持未知还是有另一个合格来源；购买前还要核对报价完整度、拆分后的合约身份、IV/Delta 正负及缺失率。不得从 AAPL/SPX/SPY 通用样本推断 SOXX 可用，更不能声称超过 SOXX。

公开采购页的只读报价观察见 `live_vendor_quote_observation.json`：2012-01-04 单日 SOXX、含 IV/Greeks 的 Cboe EOD 与 390 分钟报价分别显示 **10 美元**小计，但页面价格不保证该日实际交付完整 SOXX 合约链。另一条 [ORATS Delayed Data API](https://orats.com/data-api) 当前显示 **199 美元/月、20,000 请求**和 2007 年起的一般历史；[ORATS 近收盘历史档案](https://orats.com/near-eod-data)为一次性 **599 美元**、约 **500 GB**、14 天下载窗口，99 美元/月的 FTP 服务仅含购买后新数据。已有公开单日 SOXX 样本，仍未证明 2011–2022 全期 SOXX 覆盖、原始送达时间及 NAS 归档许可。未加入购物车、联络供应商或购买。

已准备 [ORATS 询样草稿](orats_inquiry_draft.md)，六个抽样日期均在冻结 Stage1 Qlib 交易日历内。发送给供应商需要用户明确授权；截至本卡更新，询样仍未发送。

**2026-10-23 来源更正：** [Cboe 官方 FAQ](https://datashop.cboe.com/faqs)把 Option Quotes 和 Option EOD Summary 的总体起点列为2010年1月，但两个现售[Option Quotes](https://datashop.cboe.com/option-quote-intervals)、[EOD Summary](https://datashop.cboe.com/option-eod-summary)产品页写2012年1月。不能据产品页断言Cboe无2011数据，也不能据FAQ断言2011 SOXX及IV/Delta可交付；具体冲突与询样问题见 `cboe_start_conflict_addendum.json`。旧卡对“2011缺口未证明可填”的决定不变。

**现售界面核查：** Option Quotes 与 Option EOD Summary 的历史日期选择器均只列2012–2026年，购物车为0、未提交表单。这与两个产品页一致，但不能排除FAQ所称2010起的旧版/人工交付渠道。2011 SOXX 加Calcs的实际可交付性仍须供应商确认；见 `cboe_checkout_ui_preflight.json`。

**配置时钟更正（2026-10-23）：** 两份未发送询样草稿及旧 `source_path_decision_20261023.json` 曾把配置决策误写为09:46 ET；冻结 f696 实际在次交易日 **09:29 ET** 决策。草稿已修正，旧来源路径记录按 [`allocation_clock_correction_20261023.json`](allocation_clock_correction_20261023.json) 的时钟覆盖。只有此前已首发并可取得的前一交易日期权版本才能进入不改时钟的分配器；迟交、修订或无源日期保持未知。0消息、0购买。
