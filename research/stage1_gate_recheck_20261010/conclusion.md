# Stage1 风险门可复现复核（中文结论）
- 目的：用**已提交**的 recorder 重算四个风险门，验证生产卡的 `worst_rolling_5d_pct=-12.11%` 是否可复现。
- 来源卡：intraday_on_policy_staged_validation_v1；门槛：{'max_drawdown_pct': -30.0, 'worst_day_pct': -12.0, 'cvar5_pct': -8.0, 'worst_rolling_5d_pct': -12.0}。
- 结果：
  - recorder 454e94c9ed5f（2023-01-03..2026-06-26）：收益 629.7%，MaxDD -28.11%，worstDay -9.38%，CVaR5 -5.25%，worst5d -11.30%，gate=PASS
  - recorder 76a6fe6a54cf（2023-01-03..2026-06-26）：收益 639.0%，MaxDD -28.11%，worstDay -9.38%，CVaR5 -5.26%，worst5d -11.30%，gate=PASS
- 已提交证据整体：全部 PASS。
- 决定 `committed_evidence_passes_all_gates`。
- 限制：只复核已提交 recorder；生产卡未提交的 recorder 无法在此验证；通过本身不等于晋升，仍需连续前向证据。
