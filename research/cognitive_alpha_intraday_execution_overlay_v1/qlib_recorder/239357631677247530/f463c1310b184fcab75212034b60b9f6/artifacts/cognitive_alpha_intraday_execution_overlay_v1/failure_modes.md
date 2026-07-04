# Failure Modes

- Overlay 使用的是已有日级 target_position，不重新训练仓位模型；它只能改善成交和盘中退出，不能创造隔夜趋势收益。
- 15m 信号来自 bar close，overlay A 只允许下一根 15m open 入场；没有使用同一根 bar 的未来价格。
- 成本压力用固定 bps；尚未验证 bid/ask、排队、延迟、停牌、跳空和盘前盘后成交质量。
- walk-forward 只在已有日级 selected positions 上重新选择执行 variant；不是完整重训日级 LGB/XGB 仓位模型。
- conditional_risk_sleeve_walk_forward.csv 按年度 fold 重新选择条件 sleeve；主切分 selected_conditional_risk_sleeve.json 不能单独作为年度稳定证据。
- 若 daily_position_source_audit.csv 缺少 2022-2026 的 selected positions，年度日内 overlay 只能作为局部诊断，不能作为超过日级模型的晋升证据。
- 若 validation 选择 baseline，说明日内执行层没有足够证据替代 next-open 一次成交。
