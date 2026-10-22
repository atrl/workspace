# 费城联储 ADS 实时版本：原始 ZIP 已封存

为寻找不同于价格/VIX和已失败月度半导体产量、出口信号的广泛经济活动来源，按冻结 URL 一次读取费城联储 [ADS 所有实时版本](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/ads)。官方网页标示约 5.1 MB，但本次服务器响应 `Content-Length=168,969,969`；首次 10 MB/90 秒预算只得到不完整的临时传输，已保留失败收据并未解析。另冻一次 190 MB/360 秒预算后，取得完整 **168,969,969 字节**的官方 ZIP，SHA-256 `6b6001da2127ad94066828b3a3151368adeb8f83a7dfe65f1901678703cfef44`。内部只有 `ADS_All_Vintages-zip.xlsx`（解压后 203,849,796 字节）。原始 ZIP 已在 NAS 按哈希归档并实际恢复同一字节，存档 SHA-256 `641bffb7c565097790d976a7c80c39e4ab17f6a233e44135e3a6f4d9f00fc84a`。

只读该已封存 XLSX 的表头和日期轴，`Sheet1` 维度为 `A1:BHQ24338`，首列为 `Date`；其余 **1,576 个唯一版本表头**均符合 `ADS_INDEX_MMDDYY`，版本日从 **2008-12-05 至 2026-10-22**，包含 2011 年 91 个、2022 年 85 个、2023 年 86 个及截至本次下载的 2026 年 64 个。观测轴首尾为 1960-03-01 与 2026-10-17。详见 `schema_header_audit.json`。此步骤没有读取 ADS 数值或市场结果。

决定：`official_ADS_all_vintages_original_ZIP_archived_header_dates_qualified_availability_pending`。来源身份、完整字节、NAS 备份和版本**表头日期**已经核对；版本日还不能代替盘中首次发布时间，尚未生成 Stage1 决策日可得值、缺失图或增量信息检验。本卡 0 市场结果、0 Qlib 因子、0 拟合、0 账户，不能据此声称 SOXX 超额。下一步冻结保守的首次可用交易日映射，再从同一原件检查值与缺失，并按原 Stage1 Qlib 日历对齐。
