# 中文结论

- 做了什么：以冻结的 Stage1 confidence-veto 为直接基线，固定测试论文启发的四个结构消融：恢复动作只限趋势、35% SOXX 波动率预算、趋势/震荡/危机 beta cap，以及状态 cap + 2%/5% no-trade band + 3日最短持有。没有用 2023-2026 选结构或参数。
- 基线：开发期 10bp return=424.9560%、MaxDD=-26.7281%、open trade days=1762、10:31 actions=19。
- 恢复状态反证：趋势限定把 10:31 actions 从 19 降到 0，但 return 降到 395.5684%。原19次动作实际位于危机/震荡后的恢复阶段，不能按普通趋势追随信号删除。
- 波动率预算：return=119.0238%、MaxDD=-26.8276%；固定35%预算与现有 controller 重复去风险，收益损失过大。
- 状态限仓：return=83.3565%、MaxDD=-30.5291%；简单把震荡 beta 限到0.8、危机限到0.25并不适合当前模型。
- 稀疏执行：open trade days 从 1762 降到 455，turnover 从 129.0629 降到 102.0020，但 return 仅 112.4183%、MaxDD=-27.1721%。这证明每日目标变化包含有价值的信息，粗暴3日冻结并不是免费降频。
- outer 复用诊断：基线 return=380.2061%，稀疏方案=184.0858%；该窗口已反复查看，只用于确认失败没有反转，不能作为选择证据。
- 决策：research_reject_no_structural_profile_passed_all_gates。没有结构候选同时超过基线收益、守住风险、通过25bp并减少20%开盘交易日；保留 confidence-veto baseline，不修改生产模拟盘。
- 下一步：若继续降频，应把目标改成“未来3-5日累计调仓价值/交易成本”的监督式 action-value objective，而不是继续调固定波动率、MA或no-trade阈值。
- qlib：SOXX状态特征通过 D.features 读取，DatasetH 对齐开发/outer 段，完整 replay 和模型比较保存在 recorder 6b7ad80a749640d79daa0654234592e2；卡根只保留小型审计 sidecar。
- 不构成买卖建议。
