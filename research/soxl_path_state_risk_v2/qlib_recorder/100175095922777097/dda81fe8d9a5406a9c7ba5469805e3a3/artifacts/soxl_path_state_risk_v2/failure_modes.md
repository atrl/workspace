# Failure Modes

- overlapping_labels: 逐日标签重叠，决策同时检查episode。
- daily_resolution: 日线无法区分同日先涨后跌与先跌后涨。
- fixed_thresholds: 门槛在运行前固定；稳定性网格仅诊断，不据此事后选参。
- factor_only: 风险辨识不等于含成本策略收益。
