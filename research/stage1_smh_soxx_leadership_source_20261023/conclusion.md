# SMH 相对 SOXX 领导性：发行人原始历史源预检

VanEck 官方 [SMH 页面](https://www.vaneck.com/us/en/investments/semiconductor-etf-smh/overview)的“[NAV & Premium/Discount History](https://www.vaneck.com/us/en/investments/semiconductor-etf-smh/downloads/fundhistoprices/)”下载，首次 GET 因无 Cookie 进入 302 循环；冻结仅启用 Cookie 保存的机械修复后，第二次 GET 返回 XLSX。失败头哈希、修复方案和无 Cookie 值的传输收据均保留。原始文件 SHA-256 `c8b0c443…`，Yanxuan NAS 归档 SHA-256 `11c83f7f…` 已远端哈希及限定恢复。

文件一张表、4,090 个唯一日期，2011-12-20 至 2026-10-21。2012–2022 有 **2,768** 行正值 NAV 与 Last Trade，2023–2025 有 **1,020** 行（含周末重复值）；全表 NAV、Last Trade、溢折价均无缺值，Volume 缺 361 行。SMH 2011 年 12 月才开始，原 Stage1 2011 年大部分训练日没有该来源。VanEck 明确 NAV 于营业日收盘后计算；因此在 Stage1 09:46 决策时，只可能使用已经公布的前一交易日数据。文件是当前回溯下载，历史首次送达时刻、修订与拆分调整仍需验证。[VanEck 2023 年拆分公告](https://www.vaneck.com/us/en/press-releases/smh-vaneck-semiconductor-etf-share-split-press-release.pdf)记载 2023-05-05 开盘起 2 比 1 拆分，当前文件跨拆分日价格连续，提示回溯调整；不能将原始价位当作当时发行值。

**决定：来源候选，尚不构成 PIT 特征。** 这个 SMH 大盘龙头相对 SOXX 的价格领导性机制不同于已失败的 XSD 等权广度与 SOXX 自身趋势，但“不同”不证明有增量。当前零 SOXX/SMH 未来结果读取、零拟合、零账户。下一步先以原生 Qlib 日历核对完整交易日、拆分与前日可用时点，再冻结与原 Stage1 39 项的匹配信息门；任何通过都须在完整自有状态 0bp Tiered 账户、2/5bp 和独立前向证据上重新验证。
