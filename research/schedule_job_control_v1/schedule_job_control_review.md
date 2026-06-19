# Schedule Job Control Review

## 中文结论

- 做了什么：验证 enabled、disabled、locked 三类通用 schedule job 控制路径。
- 核心指标：enabled_status=success；disabled_status=skipped；locked_status=locked；disabled_task_calls=0。
- regime 差异：本卡片不评估市场 regime；它保证不同调度状态下的任务执行差异可控。
- 是否晋升：否；这是 src/schedule 稳定契约验证，不是策略晋升。
- 风险：若 disabled job 仍执行，会导致数据更新、dashboard refresh 或 paper execution 在暂停状态下误运行。
- 下一步：继续把定时触发的 calendar 判断、失败重试和通知摘要留在 schedule pipeline notebook 中审阅。
