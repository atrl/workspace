# 费城联储 ADS 实时版本：原始 ZIP 已封存

为寻找不同于价格/VIX和已失败月度半导体产量、出口信号的广泛经济活动来源，按冻结 URL 一次读取费城联储 [ADS 所有实时版本](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/ads)。官方网页标示约 5.1 MB，但本次服务器响应 `Content-Length=168,969,969`；首次 10 MB/90 秒预算只得到不完整的临时传输，已保留失败收据并未解析。另冻一次 190 MB/360 秒预算后，取得完整 **168,969,969 字节**的官方 ZIP，SHA-256 `6b6001da2127ad94066828b3a3151368adeb8f83a7dfe65f1901678703cfef44`。内部只有 `ADS_All_Vintages-zip.xlsx`（解压后 203,849,796 字节）。原始 ZIP 已在 NAS 按哈希归档并实际恢复同一字节，存档 SHA-256 `641bffb7c565097790d976a7c80c39e4ab17f6a233e44135e3a6f4d9f00fc84a`。

决定：`official_ADS_all_vintages_original_ZIP_archived_schema_pending`。目前仅证明来源身份、完整字节及备份；**尚未审查 XLSX 的版本日期、历史首发时钟、2011–2026 覆盖、修订或与原 Stage1 39 输入的增量**。本卡 0 市场结果、0 Qlib 因子、0 拟合、0 账户，不能据此声称 SOXX 超额。下一步只读此已封存原件的工作表/版本结构，先冻结解析与时序规则。
