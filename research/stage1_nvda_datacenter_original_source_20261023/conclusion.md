# NVIDIA 原始 SEC Data Center 实际收入：来源覆盖预检

只读已封存的 **63 份** NVIDIA 季度盈利公告 SEC 原件，逐件核对 accession 和原字节 SHA-256；父来源及本卡小审计均在 Yanxuan NAS 归档、限定恢复。2011、2012 年的八份公告没有 Data Center/datacenter 字样；全部 63 份中 **50 份**出现该术语，简单的“术语附近收入金额”文本探针命中 **31 份**。这仅是定位线索，不是 31 个已认证的收入数值或因子。

来源结构解释了探针低估：2016–2019 年的实际收入常放在同一原始申报的 CFO commentary 附件 `EX-99.2` 的 “Revenue by Market Platform” 表，而新闻稿 `EX-99.1` 多为业务叙述；2019 年一份新闻稿原件标为 `EX-95.1`。2020 年后的公告逐步在新闻稿或表中直接给出 Data Center 收入及同比。若研究“AI 数据中心需求”，应固定**原始表中的当前季与上年同期金额**，保留早期未知和名称从 `Datacenter` 到 `Data Center` 的变化，不能把整段“提到 Data Center”当成增长值。

首次执行把整份 SEC submission（含图像/ZIP）送入 HTML 解析器，在第一份原件遇到无效字符引用而中止；没有持久化结果。一次机械修复仅选择新闻稿 `DOCUMENT/TEXT`，62 份 `EX-99.1` 加一份有原始新闻稿说明的 `EX-95.1`，原 accession 与原字节不变。失败尝试读了 1 份，修复执行又读 63 份，**累计 64 次原件读取，超过冻结的 63 次预算 1 次**；错误收据、修复计划和预算收据均保留，不再重跑。

本卡 **0 数值因子、0 SOXX 收益读取、0 拟合、0 账户**。决定：`original_SEC_NVDA_Data_Center_text_coverage_only_numeric_semantics_unqualified`。下一步另行冻结 CFO 原始表的逐季数值/单位/同比及 SEC 接收时钟审计；即使来源过门，还要先检验其相对原 Stage1 39 项的增量，再进入完整 0bp 账户。
