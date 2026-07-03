# Failure Modes

- promotion_decision: not_promoted_under_base_constrained_ref
- strict_position_risk_gate_pass: False
- baseline_risk_fallback_pass: True
- walk_forward_gate_pass: False
- intraday_execution_status: ok
- soxx_pit_status: single_snapshot_survivorship_biased
- lgb_xgb_consensus_selection: True
- survivorship_bias: SOXX 成分股仍是 current holdings backfill，未达到 point-in-time holdings 生产标准。
- execution_gap: 1m stress 只覆盖已有 Futu cache，仍不是真实订单簿/夜盘成交证明。
- model_dependency: 当前只允许 LGB/XGB min-consensus 候选晋升；若共识仍依赖 base/risk-gate priors，base_constrained_ref regime 失效时仍会同步退化。
