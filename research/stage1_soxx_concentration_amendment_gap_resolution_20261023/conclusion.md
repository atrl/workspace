# Stage1 SOXX 披露集中度：11 日修订缺口已关闭

复用冻结原生 Qlib 来源 Recorder 的 48 个首次报告事件和全部 3,020 个训练日。四份晚到修订的 SOXX 普通股名称（仅内部空白规范化）、股数及披露市值均与首次版相同，因此只将 2017-08-15 至 08-29 的 **11 个 `amendment_uncertain` 标记清除**，相应 `qualified_source` 改为真。原 accession、报告期、SEC 申报日、生效交易日、前五名占比、HHI、来源年龄及其余 3,009 天均逐项保持不变；无修订导致的数值事件或来源年龄重置。

新原生 Qlib DatasetH/Recorder `479e9d93a77a47198c23a08e429d8598` 的 3,020 行读回通过，Yanxuan NAS 归档 SHA-256 `d28046b6f86009838e232fa4cec8ca5657226bdd89abd409b785a8cd11c827e3` 的远端哈希与限定恢复通过。原 Recorder `d128dfd4beca4cf384e428e45a14707f` 保留作修复前证据，详细门和逐字段不变量见 `source_audit.json`。

决定：`conservative_Stage1_train_source_calendar_complete_amendment_gap_closed_information_unrun`。这只证明 2011–2022 训练日按 SEC 申报日后的实际 Qlib 会话有完整且修订语义一致的披露篮子来源；逐份历史首次公开的精确盘中时钟、2023–2026 复用诊断来源、独立前向证据及 SOXX 费用后效果仍未证明。本卡 0 行情结果读取、0 拟合、0 账户，Goal 继续。
