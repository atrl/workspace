# FT-900 规范来源信息计划 V3：拟合前身份阻断

V3 修正了 V2 的执行授权冲突，并继续引用 23 个过渡未知日的规范 Qlib 来源。独立审查确认原 Stage1 39 输入、成熟标签、共同控制、训练域填补、固定预算和来源哈希在设计层匹配，但发现 notebook 的失败结果状态仍硬编码为 `...v2_failed`。如果运行并失败，`result_review.json`、摘要和结论会把 V3 错标为 V2，无法正确关闭固定版本。

决定 `BLOCK_DESIGN_BEFORE_FIT`，见 `design_review.json`。本版没有代码单元执行、SOXX 结果读取、模型拟合、bootstrap、Recorder 或资金账户。原冻结计划和独立审查保留；下一版只修复版本身份并重新审查精确哈希，不回填本版结果。

后续仓库状态更正：另一张固定卡 `research/stage1_ft900_export_unknown_information_20261023` 已对**同一规范来源、原 Stage1 39+3 控制、同一 H5 标签及同一复用留出期**完成两次拟合和独立结果复核，信息门失败、0 账户。本 V3 继续拟合会重复已关闭版本，故不再执行，也不建立 V4 重试；精确重复关系见 `duplicate_closeout.json`。本卡的 0 拟合与另一张卡实际 2 拟合分别计数。
