# Schedule Market Calendar Gate Review

## 中文结论

- 做了什么：验证 `src/schedule` 可在 job 执行前识别 normal、half_day 和 closed 市场日历状态。
- 核心指标：closed job 返回 skipped，task 调用次数为 0，lock_created=False，且不发送成功/失败通知。
- regime 差异：本卡片不评估市场 regime；它处理的是 schedule 日历状态差异，供后续数据更新、模型刷新、dashboard export 和 paper/sim 复用。
- 是否晋升：否；这是 schedule 框架 gate，不是模型或策略晋升。
- 风险：若休市或半日市没有在任务入口明确处理，定时任务可能在错误日期更新数据或触发 paper/sim。
- 下一步：把该 gate 接入更高层 schedule pipeline 的运行前检查，并在生产 notebook 中固定真实交易日历来源。
