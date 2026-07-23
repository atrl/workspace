# Taleb肥尾与脆弱度特征结论

- 跑了什么：固定同一tabular多任务直接连续仓位模型，分别加入经验尾部、期权隐含尾部、结构脆弱度和全部Taleb特征；没有底仓、先验或仓位网格。
- 基准特征：开发期10bp财富/SOXX=0.8745，MaxDD=-43.29%，tail1 AUC=0.5076。
- 开发期诊断优胜：tabular_base_multitask；10bp财富/SOXX=0.8745，25bp=0.6082，MaxDD=-43.29%，tail1 AUC=0.5076。
- 连续尾部诊断最佳：tabular_base_implied_tail / soxl_future_adverse_5d，平均pinball skill=-0.0454，正skill年份=3/7。
- Outer仅诊断：10bp财富/SOXX=0.7809，MaxDD=-54.92%。
- 晋升决定：not_promoted_taleb_features_or_risk_gates_failed。
- 风险：Hill/ES等滚动尾部估计仍有参数不确定性；成分股回填和2024年才开始的外部15分钟数据均未参与选择；outer没有参与特征或模型选择。
- 下一步：只有开发期标签可学习性和风险收益同时改善，才保留对应特征族；否则停止日线尾部工程并转向真正PIT盘前或更长历史的期权截面。
