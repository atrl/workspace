# Failure Modes

- promotion_decision: not_promoted_failed_walk_forward_risk_gates
- walk_forward_gate_pass: False
- baseline_risk_fallback_pass: True
- previous_model_return_floor_pass: False
- intraday_execution_status: ok
- selected_vs_base_constrained_ref_return_gap_pct: -276.51021061533805
- primary_architecture_risk: base_constrained_ref 仍决定 opportunity 集合；如果 base 完全错过新 regime，meta-label 只能做 take/skip，不能凭空发现方向。
- intraday_limit: 15m 特征只用 lag1，适合开盘前风控；盘中做 T 仍需要独立的实时执行模型和更细订单/成交验证。
- constituent_limit: SOXX holdings 若不是完整 point-in-time，会残留 survivorship bias。
- execution_gap: 1m stress 是开盘/VWAP 压力测试，不是真实订单簿成交保证。
