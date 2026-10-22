# SOXX 2023–2026 原始 N-PORT 普通股集中度

15 份已归档且身份合格的原始 SEC XML（14 份首版、1 份迟到修订）各有 30 条 `EC`／Long／NS／USD／CORP 普通股，共 **450 行**。逐股原始 `pctVal` 与 `100 × valUSD / netAssets` 差均不超过预定的 `0.000001` 个百分点；38 行原件 CUSIP 占位符保留，不倒推跨报告身份。只用当份 30 股原始市值求股内权重，前五股占比范围 **0.329802–0.383932**，HHI 范围 **0.046813–0.051563**；衍生品与短期投资未混入普通股分母。

针对 **2025-09-30** 报告的 2026-07-13 `NPORT-P/A`，与 2025-11-26 首版按原证券名、CUSIP、股数及市值逐行比较，**30 条普通股完全相同**，前五股占比与 HHI 不变。修订不能回填至其申报前，也不能覆盖当时较新的 2026-03-31 报告；每日有效状态仍需单独映射。

原生 Qlib 事件 DatasetH 与 Recorder `1cb553668fed4767a50a90c2b76fe3d0` 读回通过，Yanxuan NAS Recorder 归档 SHA-256 `c59f0c8b1cf45cdc7021765e493f7d543dbffc8cf4c2d63d5d96962b1b1d6574` 经远端哈希和限定恢复核验。原始 XML 来源另有独立 NAS 归档。**0 新 HTTP 请求、0 市场收益读取、0 拟合、0 原生账户。** 这些是基金滞后披露持仓代理，非官方逐日指数权重或 SOXX 超额结果。

决定：`held_original_NPORT_equity_concentration_source_qualified_daily_active_map_pending`。
