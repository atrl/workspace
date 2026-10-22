# TSMC 原始 SEC 6-K：64 期接受时钟与下一交易日

复用 64 个季度月份已识别的原始 6-K（正文原字节和 NAS 限定恢复由上游卡保存），只读原 Stage1 训练价格来源的 Qlib 日历。以 SGML `<ACCEPTANCE-DATETIME>` 日历日期和 SEC submissions API `acceptanceDateTime` 文本的 UTC 日历日期中较晚者为界，取**下一实际 Qlib 交易日 09:46 美东**。同时把 SGML 头分别按纽约当地与固定 EST 解释，并把 API `Z` 字面值按 UTC 解释，要求决策较三个解释中最晚时刻至少 12 小时。

结果：**64 期中 63 期**通过保守下一交易日时钟，最短缓冲 **20.506 小时**；2026-10 事件超出已封存 Qlib 快照，保留 `out_of_snapshot`，不填入最后一天。全部 64 期 API 文本与 SGML 头按纽约时间解释相差恰好 **4 小时（48 期）或 5 小时（16 期）**，但 UTC 与 SGML 的日历日期没有一次不同。不能仅凭尾随 `Z` 把 API 时刻当作无歧义的真实公开时刻；本卡用下一交易日避免日内误用。

首次执行因把上游平铺的 NAS 收据误作嵌套字段，在 Qlib 调用前失败；机械修复只改两个字段名，消费一次 `D.calendar`，0 外部请求、0 市场收益读取、0 拟合、0 账户。SEC [PDS 规范](https://www.sec.gov/files/info/edgar/specifications/pds_dissemination_spec.pdf)区分接受与公开分发，故这些是**条件性日期资格**，不是逐期首次公开页面收据。前瞻营收的币种/会计语义还未统一，也没有 Stage1 信息或 0bp 账户结果；SOXX v2 目标未完成。
