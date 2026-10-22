# 原 Stage1 SOXL 选择损失集中度 v1：解释门失效

原生 Qlib 的 944 日物理选择损益与季度闭合仍有效，但 v1 附带的 `frame.soxl_return−frame.soxx_return` 是决策前已知的过去收益，不能解释同一持有期损失。2026Q3 只有 8 个负选择日，固定 top10 占比必然为 100%，不能作为异常日门。保留 v1 Recorder/NAS 和失败证据；v2 只修正来源时间语义、使用 top2/top5 与有效损失日数描述，不拟合或重跑账户。
