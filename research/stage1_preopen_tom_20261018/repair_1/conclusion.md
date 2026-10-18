# 固定09:29条件NYSE日历代理：配对实验

完成B/C各3seed固定300轮拟合，全部6模型保存固定SHA后才加载评价标签/收益。使用原train-only3014/20741样本与39/16输入，4新增allocation槽中B/C共享3known masks，仅C读取已知centered±0.5，unknown在canonical保留NaN。原shared encoder/aux/loss/optimizer和Stage1政策不改。

6个新ensemble费用账户、15封存参考、21组比较及15CI均保留；不按分类分数/seed选择。来源为09:29有条件计划，不是当前日已开市确认、机构资金或PIT证据。

|方案|bp|累计净收益%|收盘DD%|CVaR5%|
|---|---:|---:|---:|---:|
|A_active_research_replay|0|229.686|-29.487|-4.389|
|A_active_research_replay|2|280.650|-27.060|-4.211|
|A_active_research_replay|5|41.521|-30.517|-4.026|
|B_shared|0|280.383|-25.709|-4.215|
|B_shared|2|270.223|-25.766|-4.215|
|B_shared|5|265.858|-25.819|-4.204|
|B_preopen_tom_context|0|245.907|-28.234|-4.250|
|B_preopen_tom_context|2|241.008|-28.545|-4.252|
|B_preopen_tom_context|5|244.948|-27.716|-4.225|
|C_preopen_tom|0|112.279|-29.631|-4.038|
|C_preopen_tom|2|109.644|-29.820|-4.040|
|C_preopen_tom|5|102.344|-30.114|-3.982|
|SOXX|0|380.588|-40.863|-4.905|
|SOXX|2|380.568|-40.867|-4.905|
|SOXX|5|380.538|-40.872|-4.906|
|strong9097|0|441.681|-26.812|-4.399|
|strong9097|2|434.269|-26.881|-4.400|
|strong9097|5|423.730|-26.956|-4.401|
|additional27baa|0|485.498|-27.171|-4.454|
|additional27baa|2|477.826|-27.223|-4.455|
|additional27baa|5|466.073|-27.317|-4.457|

决定：not_promoted_fixed_preopen_TOM，执行者不自签晋升。失败或未取得的门：net_relative_wealth_all_costs, sampled_drawdown_30pct, complete_annual_70pct, rolling126_70pct, paired_block_lower_positive, fresh_fullminute_risk_qualified, qualified_independent_forward。

全部年度/季度和126日滚动保留，2024/2025完整、2023/2026partial分列。B的masks本身含部分日历信息，C/B仅检验其外数值增量；还必须对A、旧B、SOXX和强参考满足全门。重复历史和3seed不构成独立市场证据。

日历vintage/临时闭市/修订缺口与条件前缀假设保留；费用会改变自身状态，不能把路径差异解释成单控制器或简单费用因果。生产未改，源数据请求0。

下一步为独立结果评审；固定失败版本不得调整venue、窗口、mask、seed、编码或控制器救援。历史通过也仅是待风险/新前向资格的候选，整体稳定SOXX超额目标未完成。

Recorder：`891acb11f06746baa0d17e98f756bb8f`；完整证据见summary/verification_manifest/nas_binding。

实际结果：C的0/2/5bp净收益112.279%/109.644%/102.344%；2bp matched B为241.008%、旧B270.223%、A280.650%、SOXX380.568%、strong9097为434.269%。全部15CI下界不为正。所有成本采样DD均超过30%（30.522%/30.711%/30.994%），5bp收盘DD也达30.114%；CVaR门通过，其余财富、年度、rolling稳定性门失败。

2bp完整2024/2025相对SOXX财富为1.068489/0.634172，仅1/2完整年度胜出；2023/2026为partial单列。C的126日正超额窗口比例0/2/5bp为26.62%/25.64%/24.91%，均低于70%。全部种子/费用/年度/季度保留，没有选赢家。

本固定版本已由独评确认关闭，见[independent_evaluation.json](independent_evaluation.json)。root唯一最终NAS冷读55774 exit0，98artifact、6模型、六模型屏障、6newnative+15refs及21metrics/15CI核验通过，见[readback_verification.json](readback_verification.json)。这不是经济通过或晋升；只允许一次已预登记的Recorder-only失败归因，不调窗口/offset/mask/seed救援。

## 一次固定Recorder-only失败归因

42903已完成并保存四张归因表，但最终JSON遇到NumPy int64序列化错误；随后获准output_finalize57415 exit0，仅规范化标量和补齐同口径小摘要。四张原表SHA逐一保持，无账户/日历分组收益/CI重算。失败cell、原代码snapshot和两个receipt均保留；正式98artifact和NAS归档不变。

**数值输入已改变提议，并传入了持仓，但完整路径变差。** C/B原始现金目标平均绝对差2.2146个百分点，平滑后为1.9274个百分点。C原始平均名义β1.109828略高于B1.095926，原始现金35.5667%低于36.1995%；2bp实际平均名义β却为0.690251对0.842774，现金类仓位53.0050%对45.6483%。名义β按3×SOXL权重+SOXX权重计算，并非回归估计的市场β；现金类包括应收款，不能全部解释为可立即交易现金。

2bp实际SOXL/SOXX平均权重C为11.0151%/35.9799%，matched B为14.9629%/39.3888%。C非正常risk_state637天、B445天；state差异319天，首2024-09-06，clock差异376天；open execute差118天，首2023-01-11。全部比较账户的first-hour recovery_action均为0。C三费用state与clock彼此相同，持仓/费用仍可不同；上述是自身NAV、峰值、费用和动作共同形成的路径描述，不是单控制器、简单fee subtraction或单日因果分解。

最后一次更新前allocation proxy loss在3seed中有2个C优于matched B、1个略差，未形成一致改善。H5 alpha AUC B0.496320/C0.496153、tail AUC B0.601711/C0.602076几乎不变；日历只直接进入allocation head，因此这些H5分类分数不是对calendar H1信息的独立检验。本研究支持的结论是固定数值干预未兑现交易增量，不能由此证明所有日历信息普遍无效。

|滑点bp|完整年度胜matched B|完整年度胜A|完整年度胜SOXX|全部季度胜matched B|全部季度胜A|全部季度胜SOXX|
|---:|---:|---:|---:|---:|---:|---:|
|0|0/2|1/2|1/2|7/16|6/16|4/16|
|2|0/2|1/2|1/2|8/16|5/16|4/16|
|5|0/2|1/2|1/2|7/16|8/16|4/16|

全部4年度含首尾partial，16季度中14个complete、2个partial，180个配对逐项保留。2bp完整年度C胜matched B为0/2、胜SOXX1/2；全部季度分别8/16和4/16。没有选取较好年份或费用场景改写失败结论。

固定calendar/mask分组全部24格×4方案×3成本共288行，包括零支持格。944账户区间中：三mask全known且proxy0有748日、全known且proxy1有179日；first3已知而last未知的3日proxy仍为1，另14日proxy未知。matched B与C共享这些mask，三个partial-true日中数值本身不构成mask之外的新信息。

|2bp固定分组|日数|C实际平均名义β|matched B实际平均名义β|C组内已发生区间复利%|matched B组内已发生区间复利%|
|---|---:|---:|---:|---:|---:|
|全known，proxy0|748|0.697946|0.855351|140.401|254.158|
|全known，proxy1|179|0.666764|0.774685|-10.760|-3.263|
|first3 true、last unknown|3|0.269268|0.464959|1.010|1.900|
|proxy unknown|14|0.669629|1.122355|-3.256|-2.322|

上述复利是同一完整持仓路径中、可能不连续的已发生净收益区间乘积，仅作描述性贡献分组，不是另一个可交易账户或分组策略回测。C在四个有支持组的2bp记录都低于matched B，不能据此重选mask、反号或移动窗口；unknown组的差异尤其体现先前训练与状态路径，并非当前数值信号作用。

固定版本保持not_promoted并关闭，不修改venue、offset、mask、seed、权重或控制器救援。整体稳定SOXX超额目标未达成，下一机制需另行冻结。

来源与计算：[failure_attribution.json](failure_attribution.json)、[原scope](failure_attribution_scope.json)、[失败receipt](failure_attribution_attempt_1.json)、[output finalize](output_finalize_scope.json)。四小表：[全部年/季](attribution_periods.csv)、[提议及实际暴露](attribution_path_summary.csv)、[状态与首分歧](attribution_state_differences.csv)、[完整calendar/mask格](attribution_calendar_groups.csv)。
