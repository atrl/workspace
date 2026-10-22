# CFTC Nasdaq Mini TFF 官方源身份预检：通过，发布时间尚未合格

冻结源预检预算内下载 CFTC 2011 与 2026 年度 TFF Futures Only 原件，两次 GET；随后单独冻结的年度采集预算取得 2012–2025 年全部 14 个原件，累计 16 次 GET。2011 的 52 条及 2026 截至 10 月 13 日的 41 条周度报告均能以精确合约代码 `209742` 找到 Nasdaq Mini，并包含杠杆资金多空和未平仓量字段。原始 ZIP、请求收据及哈希已备份 NAS，远端哈希与指定成员恢复验证通过。最终结果见 `schema_alias_result.json`。最初的代码列歧义、日期值格式及日期列名变化均作为机械解析失败和修复过程保留，没有增加请求预算或读取市场结果。

**科学决定：2011–2026 年源身份覆盖已通过，尚不可进入特征或 0bp 账户。** CFTC 历史列表的日期是报告日，不是发布时间。通常在周五 15:30 美东发布上周二持仓，但 2025 年停摆补发有单独的逐期时间表。2011-06-24 官方公告记载该合约 `209742` 的 2011-06-21 未平仓量调整；2019-04-03 公告又记载 2019-03-26 持仓的事后补录，2018–2019 停摆与 2023 ION 事故也导致补发；当前年度 ZIP 不能直接证明所有数值为首发版本。CFTC FAQ 也说没有完整的历史发布日期列表。必须先固定实际可用时钟及首发版本处理，再构造 Stage1 决策前特征。

- [CFTC 年度压缩原件索引](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm)
- [CFTC TFF 字段字典](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalViewable/cotvariablestfm.html)
- [CFTC 发布时间表](https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm)
- [CFTC 历史特殊公告，含 2011 与 2025 例外](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements/index.htm)
- [CFTC COT 常见问题，历史发布日期限制](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm)

14 个中间年度各有 52–53 条精确合约报告，全部通过源身份核验；年度原件及响应头的 NAS 归档哈希为 `8fb79e0754542869d4edf6be530bfbeab8ce1951e213abc789966fe7c88cd298`，远端哈希及 28 个指定成员恢复均通过。年度执行见 `annual_intake.ipynb` 和 `annual_intake_result.json`。

下一步是冻结发布时间与首发版本的有界审查，对例外报告使用原始逐周发布件比对或明确排除；通过后才按 Qlib 日历映射冻结 Stage1 决策。零模型拟合、零 Qlib 账户、零市场收益读取；SOXX 超额目标仍未实现。
