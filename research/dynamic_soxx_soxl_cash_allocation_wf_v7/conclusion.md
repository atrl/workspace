# 中文结论

- 做了什么：新增 v7 rolling threshold 泛化审计；同一组 same-day 15m close-known 特征用于 next-open open-to-open 配置，比较 XGB、LGBM 和 XGB/LGBM 平均。
- 容量结果：online threshold capacity pass count=38；best model=xgb；best pass return=441.4457%，maxDD=-7.5787%。
- 泛化结果：validation-only nested return=45.8483%，maxDD=-6.0966%，excess=-367.8082%。
- 结论：XGB 在 2023-2026 测试窗口有 capacity pass，但参数选择不能被 2019-2022/滚动验证期支持；这属于 regime 泛化失败，不晋升。
- 是否晋升：不晋升；decision=not_promoted_online_threshold_capacity_pass_failed_generalization。
- 下一步：如果要继续推进，必须引入可解释的 regime selector 或跨资产/跨周期训练数据，而不是把测试窗口 capacity 参数直接生产化。
- qlib recorder：4e88b9dc813742049b82ab24e08f5c0a
- 不构成买卖建议。
