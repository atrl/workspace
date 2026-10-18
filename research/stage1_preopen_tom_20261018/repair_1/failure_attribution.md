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
