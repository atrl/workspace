# 尾盘ROD新增受限恢复许可：固定结果

原f696+b472晨间链保留；B/C共享新增尾盘许可，C仅多严格float64价格方向门。0拟合、6新原生账户；12旧native参考只被动增加估值点，不能称作12次新native验证。

|方案|滑点bp|净收益%|收盘DD%|全采样点DD%|
|---|---:|---:|---:|---:|
|A_active_research_replay|0|229.686|-29.487|-29.717|
|A_active_research_replay|2|280.650|-27.060|-27.266|
|A_active_research_replay|5|41.521|-30.517|-31.539|
|B_close_permission|0|267.165|-29.175|-29.411|
|B_close_permission|2|137.206|-27.666|-27.870|
|B_close_permission|5|177.288|-27.900|-28.103|
|SOXX|0|380.588|-40.863|-42.926|
|SOXX|2|380.568|-40.867|-42.929|
|SOXX|5|380.538|-40.872|-42.935|
|strong9097|0|441.681|-26.812|-26.812|
|strong9097|2|434.269|-26.881|-26.881|
|strong9097|5|423.730|-26.956|-26.956|
|C_close_ROD_positive|0|183.479|-29.386|-29.617|
|C_close_ROD_positive|2|138.856|-27.269|-27.474|
|C_close_ROD_positive|5|29.057|-27.447|-27.652|
|additional27baa|0|485.498|-27.171|-27.171|
|additional27baa|2|477.826|-27.223|-27.223|
|additional27baa|5|466.073|-27.317|-27.317|

决定：not_promoted_fixed_close_ROD_v1。失败或未获得的门：net_relative_wealth_all_costs, complete_annual_70pct, rolling126_70pct, paired_block_lower_positive, fresh_fullminute_risk_qualified, qualified_independent_forward。

全部年/季度（含partial）、126日滚动及12项置信下界均保留，不择窗；费用情景各自NAV/风险/费用状态独立。新增整数lot按15:59精确退出，不重置财富。来源为条件历史snapshot，假设延迟并非实盘首收或auction成交证明，重复历史不是前向独立证据。

本版各成本均发生实际增量lot，机会集非空；固定经济门失败，独立评估已确认关闭。本结论仅针对本固定版本，不改符号、窗口、上限、时钟或退出参数挽救。

Recorder：`f1f70f58b59946feabe131ff8b3d9645`。来源结论独立保存在source_preflight/summary.json；稳定超额目标未完成。

固定结果与区间差异：C在0/2/5bp的累计净收益为183.479%/138.856%/29.057%；2bp B为137.206%、A为280.650%、SOXX为380.568%。C只在2bp略胜B，不能通过全部成本对照。12项预定置信下界全部失败；30%采样回撤和CVaR门通过，但全窗财富、年度和rolling门失败，版本关闭。

完整2024/2025年度，C的2bp净收益分别−8.050%/18.324%，相对SOXX财富0.783028/0.823463。计入两个partial年度后，2bp C胜B为3/4、胜A及SOXX均0/4；全部16季度分别6/16、3/16、6/16。完整14季度及partial2季度单列，全部成本的180项配对完整保留于failure_period_comparison.csv。

一次固定归因并非机会空集：B在0/2/5bp实际尾盘交易日142/207/168，C为92/113/141。所有执行整数lot均精确平掉，累计股数恒等式误差0，现金恒等式最大误差约5.27e-11美元。2bp C已执行lot毛价差−158.71美元，entry+exit费用1939.72美元，现金净变动−2098.43美元；其对A终值差却为−141794.37美元。这是不同层面的已记录数值，不能把后者简单归于这笔费用或单一controller。

2bp C与B的risk_state/risk_clock在944日完全相同，但open执行决定18日不同，首次为2023-08-16。C与A的risk_state在241日不同，首日2024-02-21；clock不同274日，open执行不同58日。C三成本0对2、0对5、2对5之间，risk_state分别有170、304、194日不同。跨成本差异包括原Stage1自身费用/状态敏感性和新增尾盘反馈，不是简单费用相减。

2bp期末端点平均SOXL/SOXX/cash权重：C为12.388%/35.777%/51.835%，B为12.264%/35.646%/52.089%，A为15.082%/36.597%/48.321%。这是next-open端点持仓的描述，不是日内时间平均暴露或因果cash-drag分解。机会拒绝原因按原固定顺序记录第一个失败条件，且来自各自状态；详见failure_late_reasons.csv，不将其解释成共同反事实机会集。

一次交付冷读已由研究owner完成，见[readback_verification.json](readback_verification.json)：82个artifact、sign Dataset及6native/12passive角色核验。执行者没有重复NAS冷读或替自己签署晋升；固定负结果仍交独立结果评估。一次归因来源和公式见[failure_attribution.json](failure_attribution.json)。整个稳定超额Goal仍未达成。

独立评估：`ACCEPT_NEGATIVE_RESULT_CLOSE_FIXED_VERSION`。本固定版本正式关闭，不改符号、窗口、上限、退出或控制器挽救；持续Goal转入独立的辅助梯度投影训练设计，尚未达到稳定SOXX超额。见independent_evaluation.json。
