# 原辅助梯度的非对称投影：固定实验

仅完成C的3个seed、各300轮拟合；B直接继承已精确复现的共享头模型/预测/账户，没有重新拟合或补跑。原3014主样本、20741辅助样本、39+16输入、标签、损失权重和初始化均固定。只有辅助梯度与复合主任务梯度严格冲突时，投影共享154参数上的辅助分量；其余梯度按冻结合同保留。

三个C模型全部保存并固定SHA后才读取评价集。完成C的0/2/5bp三个独立费用原生账户，并复用A/B/SOXX/9097/27baa十五个原始3778点账户；不包含FINRA或尾盘ROD。A是冻结现役模型在同源研究时钟下的重放，非实际券商执行复现。

实际投影次数为seed71：13/300、172：0/300、273：17/300，总计30/900。2bp下C净收益270.535%，B为270.223%，A为280.650%，SOXX为380.568%，9097为434.269%。C的0/2/5bp净收益分别299.587%/270.535%/267.973%；收盘及采样DD、CVaR门通过，但财富、年度、滚动稳定性和配对CI整体门失败。12项CI仅5bp对A的下界为正，不能替代所有必需比较。

2bp完整年度2024/2025相对SOXX财富分别1.067990/0.940887，仅1/2完整年度胜出；2023与2026为partial，始终单列。126日窗口正超额比例0/2/5bp分别39.19%/35.04%/34.55%，均低于70%门。

|方案|滑点bp|累计净收益%|收盘最大回撤%|CVaR5%|
|---|---:|---:|---:|---:|
|A_active_research_replay|0|229.686|-29.487|-4.389|
|A_active_research_replay|2|280.650|-27.060|-4.211|
|A_active_research_replay|5|41.521|-30.517|-4.026|
|B_shared|0|280.383|-25.709|-4.215|
|B_shared|2|270.223|-25.766|-4.215|
|B_shared|5|265.858|-25.819|-4.204|
|SOXX|0|380.588|-40.863|-4.905|
|SOXX|2|380.568|-40.867|-4.905|
|SOXX|5|380.538|-40.872|-4.906|
|strong9097|0|441.681|-26.812|-4.399|
|strong9097|2|434.269|-26.881|-4.400|
|strong9097|5|423.730|-26.956|-4.401|
|C_aux_projection|0|299.587|-25.704|-4.209|
|C_aux_projection|2|270.535|-25.777|-4.216|
|C_aux_projection|5|267.973|-25.830|-4.197|
|additional27baa|0|485.498|-27.171|-4.454|
|additional27baa|2|477.826|-27.223|-4.455|
|additional27baa|5|466.073|-27.317|-4.457|

决定：**not_promoted_fixed_aux_projection**，独立评估已接受固定负结果并关闭本版本，整体稳定超额目标尚未达成。
失败或未取得的门：net_relative_wealth_all_costs, complete_annual_70pct, rolling126_70pct, paired_block_lower_positive, fresh_fullminute_risk_qualified, qualified_independent_forward。

完整/不完整年度、全部季度和126交易日滚动窗口见相应小表；全部3seed分类及训练梯度保留，不择窗、不按分类分数取消账户。原历史被多次研究复用，重叠窗口和多seed不构成独立市场验证。

全局裁剪系数与AdamW实际更新仍可能变化；投影的局部方向性质不证明配置损失保护、有限步收益改善或控制器因果。采样风险不等于全分钟资格，历史来源和纸面公司行动也不等于历史PIT/真实结算。生产未改。

关闭本固定投影版本，不调整角度、权重、裁剪、seed、epoch、窗口或控制器救援；下一机制由campaign另行冻结。

Recorder：`fce9a058996142c6bef71196aba31db9`；完整证据与NAS身份见summary.json、verification_manifest.json、nas_binding.json。

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

交付读回：一次NAS下载核对92件归档；主/辅助/评价Dataset均可直接prepare。评价预测946行与冻结B一致，超出944个账户信号的2026-10-08/09均未交易。负责人最初误写945行校验预期，已在同一缓存上更正并留痕，没有修改实验或重跑研究。见readback_verification.json。

独评及后续：见independent_evaluation.json；本固定梯度投影版本关闭，不改anchor、权重或阈值救援。下一步仅资格化外部公开交易日历代理的独立机制，不将未解决的机构结算截止日假说改称已证明。
