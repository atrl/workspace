# 中文结论

- 原因：旧 10:31 恢复阈值 0.55 过宽，同时没有要求 09:31 上游配置模型仍有足够 alpha 置信度，因此既会频繁触发，也会在上下游信号冲突时重新加风险。
- 消融：只提高恢复门后 return=420.1836%、actions=24、worst 5d=-11.8563%；再加 source-alpha veto 后 return=424.9560%、actions=19、worst 5d=-10.7010%。前者主要降频，后者消除冲突动作并改善尾部。
- 修复：保留冻结 Stage1 的 09:31 持仓与风控，只允许 `source alpha >= 0.25` 且恢复概率 `>= 0.825` 时执行 10:31 恢复；动作 beta=0.60。
- 开发期 10bp：baseline return=415.4288%，MaxDD=-27.5285%；candidate return=424.9560%，MaxDD=-26.7281%。
- 交易频率：真正对应“每日第二次变化”的 10:31 recovery actions 从 51 降到 19，减少 62.75%；总 turnover 减少 13.67%。09:31 日频主调仓没有伪装成被删除。
- 固定决策 25bp：候选仍超过 baseline；压力测试只重算成本，不重训、不改变动作路径。
- 稳定性：相同 C/beta/alpha-floor 分支的 0.825、0.850、0.875 三个相邻阈值均通过收益、风险和频率门，不是单一阈值尖峰；但增益主要来自 2020，2021/2022 轻微落后，已保留为风险标签。
- outer 复用诊断：baseline return=378.0558%，candidate=380.2061%，候选 10:31 动作=0；2023-2026 不参与选参，且已反复查看，不能冒充干净 holdout。
- 决策：validated_research_candidate_for_forward_shadow_not_production。已得到超过冻结 Stage1 的研究候选，但生产替换仍 blocked，下一步是冻结参数做新的 forward shadow，而不是继续看 outer 调参。
- qlib DatasetH/Recorder：训练段与完整 replay 已进入 recorder b03eab80b52f47039489ac2119e385b4，卡根只保留小型审计 sidecar。
- 不构成买卖建议。
