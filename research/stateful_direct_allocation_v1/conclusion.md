# 结论

- what_ran: 保留冻结 qlib walk-forward 特征编码作为初始化，在同一个最终仓位头中加入因果 GRU 账户状态；模型直接输出 SOXL/SOXX/cash，已删除推理时 cap/wait/ramp 和底仓加 overlay。
- key_metrics: development 25bp return=217.2719%，MaxDD=-43.0651%，worst 5d=-26.9498%；outer 10bp return=425.1761%，MaxDD=-39.2422%，worst 5d=-16.1274%。
- regime_window: 候选只由 2019-2022 年度与两年 development 窗口选择；2023-至今 outer 只在选择冻结后诊断。
- turnover: outer 开盘换手由 61.8739 降至 47.8614，变化 22.65%；活跃日 810 对 404。
- promotion_decision: `retain_current_stage1_stateful_direct_model_not_promoted`；development gate=False，完整替换门=False。
- risks: outer 已被此前多轮研究重复观察，不是全新未见测试；教师来自当前 Stage1，只用于 development 行为压缩，outer 标签不参与训练或选择；尚未验证真实 bid/ask、延迟和排队。
- next_action: 若 development 风险门失败，停止扩大网络或使用 outer 调参；只保留该结构证据并继续使用当前 Stage1。
- recorder: `7943709a719f4ca9b82513476e7c92df`。
- 不构成买卖建议。
