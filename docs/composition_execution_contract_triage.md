# 组成训练与执行合同：有限机制分诊

本次只比对当前源码、固定计划和已完成结论，不拟合、不重放、不改阈值。遵循仓库soxl-qlib-experiment技能；Qlib模型/策略接口分别以[官方Model文档](https://qlib.readthedocs.io/en/latest/component/model.html)及[Strategy文档](https://qlib.readthedocs.io/en/latest/component/strategy.html)为准。

结论：泛称“加入账户状态/完整路径训练”会重开已失败路线。**保留当前同一个EMA、离散风险控制、三费用共同最小cap、全体恢复同意以及T的精确前向合同**确实不同于旧试验，但这种结构差异尚不证明预测或执行价值，不能直接启动新训练。

|合同|composition_objective_training|full_path_tiered_allocation_20261013|
|---|---|---|
|网络输入|原39市场字段/共同编码器，没有自身持仓、峰值或风险状态|市场字段加自身三资产权重和drawdown；见[src/model/full_path_tiered.py:74–95](../src/model/full_path_tiered.py#L74)|
|训练持仓|只把T后的权重送入原allocation_loss；收益、换手代理，未执行EMA或控制器|逐日自身q/cash/peak/月度费用；整数前向、STE梯度；见[roll_full_path](../src/model/full_path_tiered.py#L98)|
|训练损失|年度相对收益均值/离散、换手与10bp代理；实际cash/DD惩罚系数为0|63预热+126评分块；2bp Tiered代理；短缺系数1、DD软罚5；120轮。块起点重置现金/峰值/月度计数；见[fit](../src/model/full_path_tiered.py#L139)|
|部署|原raw权重→EMA/漂移接受逻辑→自身风险/恢复→T→实际费用；见[训练工作流](../src/workflow/composition_objective_training.py#L100)|自身状态直接输入学出的PathPolicy；计划明确不接旧等待/ramp控制器，也没有当前三费用共同cap/恢复共识|
|结果边界|3拟合/3账户全部退步且DD>30%；T(smoothed)现金2.37%与真实2bp开盘现金45.17%只是管线描述，不是隔离控制器的因果效应|27拟合/66账户；两开发年落后，外层daily虽有累计超额但监控DD约41%；已关闭，不调罚系数/epoch/seed挽救|

关键代码：T只包裹[allocation_objective](../src/model/composition_objective_allocator.py#L12)；原训练循环仍在[minute_pit_allocator.py:160](../src/model/minute_pit_allocator.py#L160)。EMA/接受与持仓漂移见[_smooth_weights](../src/workflow/pit_constituent_auxiliary_training_allocation.py#L1303)；真实风险与恢复发生在[replay_full_path](../src/backtest/full_path_tiered.py#L40)，共同控制器实现见[causal_cost_scenario_controller](../src/backtest/causal_cost_scenario_controller.py#L10)。这些算子不交换，加入T到旧代理不等于执行对齐。

另一个明确差异是融资约束：[原Tensor结算](../src/model/full_path_tiered.py#L51)没有[实际结算的买入缩量循环](../src/backtest/ibkr_tiered.py#L94)。原15%现金缓冲下这不等于已有结果错误，但T可产生零目标现金，不能不修改就声称旧Tensor前向适配当前账本。

既有闭路仍有效：stop_rearm修改循环，cushion_target直接按净值缓冲配风险，risk_layer旁路删除风险包，annual_recovery/full_history_recovery重训恢复分类器，tangent_router_refit拟合局部execute/HOLD优势，multisession_label检查更长动作优势。它们不等价于当前完整共同控制器的梯度合同，但也不提供“加入状态就会改善”的证据；不能改名重做或继续调其阈值。

唯一下一动作：提交[固定机械可行性计划](../research/coupled_controller_gradient_feasibility_20261015/plan.json)给独立评估，不直接训练。冻结最终2022三个tangent模型及**当前**控制器身份，不替换为现有年度恢复模型；选最后189个成熟pre2023价格区间，明确只是样本内机械检查。固定两个logit方向和两组中心差分尺度，对精确Python共同控制器与单一exact-forward/STE-backward近似做前向与独立有限差分核验。预算27参考影子路径+3Tensor基线路径，0优化、0模型拟合、0原生账户；失败即关闭该近似，不换epsilon、梯度规则或窗口救结果。

这不是“前向一样所以可以训练”：精确资金约束的detach前向修正只能保证值正确，不能保证其反向有用。每个方向、每个固定尺度必须通过误差/符号及跨尺度一致性门，且不能全零。硬阈值、资金裁剪或状态切换若破坏方向一致性，就是拒绝训练的具体证据。

任何后续经济试验仍需另行批准：minute价格最早2018，不能把2011训练样本假装有同样执行历史；控制与候选必须共用可训练域。最终2022模型在早期价格上的机械资格不是2020/2021 OOF，届时需各折自己的时间合格控制器。没有新的可用信息或经资格验证的优化方向时，继续修改状态/损失只是在扩大搜索。

## 固定机械资格的完成状态

批准计划c73103cf...已完成30条影子等价路径、0优化/模型拟合/原生账户。前向EMA/目标/cap一致，但4个预登记梯度对照全部失败，两个方向的有限差分随固定尺度翻号。独立读回复算一致；该固定近似关闭，不能把前向值校正当作可训练性。详见[执行结论](../research/coupled_controller_gradient_feasibility_20261015/conclusion.md)。本笔记前述机械步骤现已完成，不是待追加运行任务。
