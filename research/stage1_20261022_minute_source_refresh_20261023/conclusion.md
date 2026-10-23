# 10 月 22 日 SOXL/SOXX 原始分钟来源与 NAS 恢复

按冻结的一次来源刷新计划，Futu OpenD 对 SOXL/SOXX 各请求 2026-10-21—22 与 10-22 单日原始 NONE 分钟，共 **4 次**；每只基金分别返回 780 与 390 行。原始 NONE 缓存与公布调整面经现有构建流程进入 Qlib 1/5/15 分钟存储，并发布为 Yanxuan NAS 已验证快照 `bf6d5121d6cf13fb7f694e3b439d8c54ab33f61590786967b4d3f31d129f3349`，`as_of=2026-10-22`。新原始来源清单的 `parent_source_id` 为 `c7ddae9b31f9a1714d886b20625c2eec4068b9bd41801c1bac4aa7ffb5c93b6e`；NAS 全对象发布哈希核验及来源清单、两份公司行动原件限定冷读通过。

原生 Qlib `D.calendar` 在 10 月 22 日有 **26** 条15分钟日历；`D.features` 对 SOXL/SOXX 各取到26条，当日09:45—10:30前四根均 `$is_complete=1`、`$bar_count=26`。原生 Recorder `12dd136d74564851b6c17f83c9e756b1` 的来源结果、日历及特征对象已归档 NAS，SHA-256 `1e4aba62fed3c925ac53652b346be32737554ddb24b39e53604e0704453f0ae7`，三件对象限定下载与本机字节一致。2020-03-16 SOXX 仍有1条既存无效原始分钟，保持隔离；它不改变本次10月22日完整性结论。

首个 notebook 运行在 NAS 来源注册完成后，因从 `<stdin>` 运行导致 Qlib 子进程无法重载脚本而停止；后续只读已发布快照，设单进程并从脚本文件运行，**没有重复请求 Futu**，也没有新模型或资金账户。归档 Recorder 的 `before_source_id` 字段因续跑读取当前快照而误写成新ID；[身份修正](identity_correction.json)以冻结计划和新来源清单的父ID交叉证明旧ID，原 Recorder 保留不篡改。

决定 `Oct22_SOXL_SOXX_NAS_source_and_Qlib_minute_freshness_restored_no_forward_strategy_claim`。此次只修复数据新鲜度；现役纸面 release 仍有已记录的代码身份/首小时语义审核缺口，没有新候选前向日、订单或 SOXX 经济成绩。完整 Stage1 0bp 稳定超额及合同v2后续门槛仍未通过。
