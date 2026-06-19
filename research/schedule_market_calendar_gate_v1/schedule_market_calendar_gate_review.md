# Schedule Market Calendar Gate Review

## 中文结论

- 做了什么：验证 `src/schedule` 可在 job 与 pipeline 执行前识别 normal、half_day 和 closed 市场日历状态，并验证可选 skipped 通知摘要。
- 核心指标：closed job 返回 skipped，task 调用次数为 0，lock_created=False，默认不发送成功/失败通知；notify_skipped=True 时通知包含 market_status、skip_reason 和 next_trading_date。data_update、model_dashboard_refresh 和 paper_execution pipeline 同样在 provider/exporter/输入读取前 skipped。
- regime 差异：本卡片不评估市场 regime；它处理的是 schedule 日历状态差异，供数据更新、模型刷新、dashboard export 和 paper/sim 复用。
- 是否晋升：否；这是 schedule 框架 gate，不是模型或策略晋升。
- 风险：若真实休市或半日市来源缺失，pipeline 可能在错误日期跳过或执行；production notebook 需要固定真实交易日历来源。
- 下一步：在生产定时配置中统一注入真实日历决策，并固定真实交易日历、半日市和休市原因来源。
