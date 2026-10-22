# Stage1 HYG−LQD 信用风险偏好 20 日来源

在已归档、NAS 定点恢复的两只 ETF 原始历史图表中，以各日**原始 close 加除息日现金分红**除以前一日原始 close 计算单期总收益；无拆股事件。每只 ETF 都有 2010-01-04 至 2026-10-22 的 4,227 个有效收盘价与 200 条分红。取截至 Stage1 信号日已结束的最近 **20 个真实交易日**对数总收益之和，HYG 减 LQD；下一个交易日 09:46 决策才使用该数值。不使用当前回溯 `adjclose` 作为模型输入。

原生 Qlib 日历上，冻结 Stage1 **3,020/3,020** 个 2011–2022 训练日和 **944/944** 个复用 2023–2026 评价日均有该固定坐标；没有幸存者样本压缩。当前回溯 `adjclose` 仅作来源诊断：单期对数收益与 raw-close+dividend 口径的绝对差中位数，HYG 为 `2.22e-7`、LQD 为 `1.96e-7`；HYG 有 2 日差超过 1bp，最大约 1.41bp，LQD 无超过 1bp 的日期。该审查不能证明当前回溯分红金额在过去各日的**首次发布版本**，`historical_adjustment_vintage_qualified=false` 明确保留。

Qlib DatasetH、Recorder `018a98da05724acda3c50b11a7b9a69f` 读回通过；Yanxuan NAS Recorder 归档 SHA-256 `0aa49fe9ef45ed6d24bf1df9e1e38de8bdfb155231050591370a47cf8b12f243` 经远端哈希和限定恢复核验，原始 Yahoo 价格/事件另有两个 NAS 原件归档。**0 新外部请求、0 SOXX 结果读取、0 拟合、0 原生账户。**

决定：`HYG_minus_LQD_20_session_source_calendar_complete_historical_vintage_unqualified_no_fit`。这是不同于已失败 SOXX 持仓集中度与原 Stage1 价格/VIX/广度输入的信用市场来源。下一步只可在明确回溯来源限制下，冻结原 Stage1 39 输入对照与一个 H5 信息检验；即使通过，也须原生 0bp 账户、来源时点复核、2/5bp 和独立前向证据分别过门。
