## 固定Recorder-only失败归因

唯一归因执行为exec73107（exit 0）。本节只计算已封存900条training_trace、固定B/C权重、原预测/目标和native账户的描述统计；0新fit、forward、autograd、replay、CI重抽、来源请求或NAS冷读。原Recorder和NAS archive不改。

学习确有改变，但触发稀疏且偏晚：seed71在第271–299轮触发13次，seed273在235–299轮触发17次；seed172触发0次，其最终全部10个参数tensor与B逐元素精确相同。其余两seed的最大权重绝对差分别0.002043/0.003289。900步full/shared parity误差均为0，所有实际global clip系数为1，未发现本轮裁剪激活。

真实AdamW参数delta相对复合主任务P及全参数allocation的线性损失变化估计均900/900为负；但仅共享参数allocation分量690/900为正。后者不能证明总配置损失上升，前者也不能保证有限步损失或收益改善。未重建原B optimizer，C路径上的original-gradient诊断不是B优化器反事实。

原始等权ensemble的SOXL/SOXX/cash平均绝对差分别0.000335/0.000409/0.000692；保存的平滑目标对应0.000455/0.000505/0.000848。alpha AUC仅由0.496897变为0.497082；Brier由0.247403变为0.247320，属于很小的分类变化，未形成有力的时序区分证据。全部5输出与所有seed分类分数均保留。

差异传入了持仓，但2bp传递非常有限：C/B实际平均beta为0.849059/0.849143，现金类权重为46.4358%/46.4308%；风险state和clock在944天完全相同，open execute仅5天不同，首个差异2023-06-09。0bp C/B state有12天差异，5bp为0天；open execute分别9/12天不同。A/B/C全部费用账户第一小时recovery_action均为0，不能把本轮结果归因于实际恢复交易。

C的0–2、0–5、2–5bp风险state各有24/70/46天差异，费用改变了自身路径；不能把费用情景收益差全部解释成单纯fee subtraction。所有首分歧及总差异见account_divergence小表，A只读取原authority已有字段，没有补造trace。

|滑点bp|完整年度胜B|完整年度胜A|完整年度胜SOXX|全部季度胜B|全部季度胜A|全部季度胜SOXX|
|---:|---:|---:|---:|---:|---:|---:|
|0|1/2|2/2|1/2|10/16|12/16|8/16|
|2|0/2|1/2|1/2|9/16|9/16|7/16|
|5|0/2|2/2|1/2|7/16|11/16|7/16|

全部年度含2023/2026 partial、季度含首尾partial，均在period_comparison和period_wins小表明示；不择窗。2bp对B的整体微增并未体现为完整年度胜出（0/2），对SOXX也仅1/2完整年度、7/16全部季度胜出。

固定结论：实际改变了少量后期学习与预测/整数持仓路径，但经济幅度与稳定性不足；关闭这一固定非对称投影版本，不据本次归因调整anchor、阈值、seed、权重或控制器。整体目标仍未达成，是否进入新机制由campaign另行冻结。

来源与计算：[failure_attribution.json](failure_attribution.json)、[scope](failure_attribution_scope.json)；训练与参数：[gradient_steps](attribution_gradient_steps.csv)、[actual_step_directions](attribution_actual_step_directions.csv)、[tensor_difference](attribution_model_tensor_difference.csv)；预测与账户：[output_difference](attribution_output_difference.csv)、[account_exposure](attribution_account_exposure.csv)、[account_divergence](attribution_account_divergence.csv)；完整区间：[period_comparison](attribution_period_comparison.csv)、[period_wins](attribution_period_wins.csv)。
