# CFTC 2025 积压报告：计划发布与当前周报更新时点

固定三份 CFTC 官方页面（12 月 9 日特殊公告、2025-12-02 和 2025-12-23 两期 TFF Futures Only 周报），各请求一次，原始正文/响应头、UTC 时刻和哈希已在本机与 Yanxuan NAS 留存，远端哈希及限定逐文件恢复通过。响应头含临时 Cookie，原字节不提交 Git。0 市场收益读取、0 拟合、0 账户。

12 月 9 日公告表把 12-02 报告安排在 12-17 发布，其当前周报页标记 `Updated December 17, 2025`。12-23 报告安排 12-29 发布，但当前页标记 `Updated December 31, 2025`。`Updated` 不是首次公开证明；它说明当前页面版本晚于计划 12-30 的 Stage1 可用时点，无法认定当前年度文件中的该行在 12-30 已同值公开。

对已封存的 Qlib 14 日映射做一次只读定向核验：2025-12-30 信号确实选用了 12-23 报告，`source_known=true`，可用时点写为 12-30 00:00 美东。因此现有映射**在进入任何信息拟合前必须修复时点**；保存原计划与 Recorder，另冻结修复版，将该报告和依赖它的 delta4 行设未知，除非取得精确首发版本。不能凭本卡断言全历史发布日、因子有效或 SOXX 超额。

证据：[CFTC 特殊公告](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements/index.htm)、官方 [12-23 TFF 周报](https://www.cftc.gov/sites/default/files/files/dea/cotarchives/2025/futures/financial_lf122325.htm)。
