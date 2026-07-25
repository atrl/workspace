# 中文结论

- 本轮没有找到满足稳健替换门槛的新模型，`replacement_promoted=false`；Stage1 继续作为受保护基线。
- 当前 Stage1 frozen outer 10bp：收益 541.79%，MaxDD -28.11%，最差五日 -11.60%。
- 已删除三条未晋升的独立可运行链：成本感知目标跟踪、GRU 状态直接分配、Stage1 扩展特征恢复；对应源代码、专属测试和旧研究卡均进入删除清单。
- 删除不等于遗忘：核心指标、失败机制、重启条件和本轮结构诊断已保存在 `cleanup_decisions.csv` 与 `retained_ideas.md`，并写入 qlib Recorder。
- outer 没有参与选择或反向调参；本卡只记录冻结诊断和清理决定。
- 受保护范围包括 Stage1 来源模型、恢复动作、on-policy 验证、泛化验证、富途模拟盘生产卡与 `soxl_backtest` legacy 对照。
- qlib recorder：7ea48e01114c493c948f9a660dceea09
- 下一步：只有新增 PIT 训练数据、成交成本标签或预登记 forward holdout 时，才重新开启已删除方向。
- 不构成买卖建议。
