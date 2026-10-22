# SOXX 2011–2018 历史申报索引：候选目录已封存

按冻结清单一次读取 SEC iShares Trust CIK `0001100663` 的四份官方历史分片（`-011` 至 `-008`），均 HTTP 200、合法列式 JSON，共 8,037 条信托申报索引行。其中固定六类 N-Q/N-CSR/N-CSRS 及修订格式共有 **215 条候选**：N-Q 99、N-CSR 51、N-CSR/A 7、N-CSRS 51、N-CSRS/A 7，申报日范围 2011-03-01 至 2019-07-30。四份上级目录的 `filingTo` 均与分片实际末日相差 1–3 天；两种原值均保留在 `capture_receipts.json`，没有改写来源或据此推断缺档。

四份原始 JSON、HTTP 响应头、完整候选清单和各自哈希已归档 NAS，归档 SHA-256 `e1194e5874e4fd46b70354e91ee47cd742421425dc81c71f617c4549e2505c4c`，远端归档哈希及每份原件限定恢复均通过。原始字节仍在 `data/snapshots/`；NAS 收据不代表所有 workspace 文件已备份。

决定：`official_four_historic_sidecars_archived_trust_candidates_only`。这 **215 条都是信托级索引候选，不是 215 份已核实的 SOXX 明细**。本卡 4 个 SEC GET、0 申报正文、0 行情/收益读取、0 Qlib 因子、0 拟合、0 账户。下一步在同一报告期的候选正文中有限检索 SOXX 系列与原始股票明细；逐期接受时钟、披露滞后和历史每日证券价格仍需单独审查，不能用目录或单一 2010 篮子回填全训练期。

与并行主卡 `research/stage1_soxx_legacy_basket_history_20261023` 的独立清单逐 accession 核对：2011–2018 年的 **188 条未修订 N-Q/N-CSR/N-CSRS 完全一致**。本卡另外保留 **13 条同期修订申报**（N-CSR/A 7、N-CSRS/A 6）及跨入 2019 年的 14 条记录；详见 `reconciliation.json`。修订表格只表示候选，尚未证实修订了 SOXX 持仓。后续正文筛选以主卡为入口，审查 SOXX 系列时必须同时检查同期间修订申报的接受时间和语义，避免把后修订版本提前用于历史决策。
