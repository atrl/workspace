# 中文结论

- 做了什么：保持原 Stage1 合法 SOXL/SOXX/cash 输出、回撤状态机、第一小时执行时点和教师标签不变，只对 recovery head 做四组特征消融；训练通过 qlib DatasetH，证据写入 qlib Recorder。
- 开发期选择：`absolute_plus_breadth`，Logistic C=0.10，阈值=0.71，动作 beta=0.6。
- 开发期效果：新候选 return=432.4489%，原 Stage1=417.2721%，财富提升=2.9340%；总换手变化=-10.3580%。
- 特征结论：绝对风险单独和“绝对风险+标准化状态”均未形成可选增益；只有绝对 VIX/波动/回撤/相对动量与滞后一日 SOXX 宽度共同进入 recovery head 后，在开发期形成通过五点阈值邻域的候选。这仍是自适应开发期搜索证据，不等同于新的样本外证明。
- outer 10bp 冻结诊断：新候选 return=468.4878%，原 Stage1=541.7889%，MaxDD=-27.4452%，worst 5d=-11.4361%。
- 25bp 固定决策压力：return=420.2666%，MaxDD=-28.7433%，worst 5d=-11.6673%。
- 稳健性：选中阈值邻域数量=5，邻域最差财富比例=0.9882；outer 不参与任何候选、阈值或 beta 选择。
- 是否晋升：`rejected_feature_recovery_outer_generalization_failed`。开发期候选被冻结 outer 泛化门否决，不进入 paper challenger，production Stage1 保持不变。
- 下一步：停止继续调概率阈值；下一轮改成带交易成本和五日尾部惩罚的连续 advantage 目标，并用独立 tail-veto head 约束恢复动作。
- 不构成买卖建议。
