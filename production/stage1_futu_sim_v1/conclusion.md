# 中文结论

- 已把晋级的 Stage1 固定为生产模拟源；Stage2 仍关闭，失败的 stateful direct 模型没有进入任务。
- 同口径区间：2023-01-04 至 2026-07-31；Stage1 10bp return=485.7896%，MaxDD=-28.1051%。
- 当前生产晋升决策=paper_sim_blocked_failed_position_risk_gates；risk_gate_pass=false。晋升门仍未通过；本次仅按用户明确要求运行独立的 Futu SIMULATE 观察周期，不代表生产晋升。
- legacy soxlbacktest 成本归一到 10bp 后 return=346.4954%，MaxDD=-55.1610%。
- card 内当前最新信号日=2026-07-31；目标计算状态=pending_next_open_mark；controller_state_source=historical_model_replay。该完整历史目标只用于 qlib 回放一致性诊断，不可直接执行。
- 实际模拟目标必须读取 active paper cycle：旧周期先平仓归档，新周期按平仓后实际现金净值重新基准化；控制器只使用本周期确认成交与 qlib 结算标记。
- 数据过期、控制器来源不属于 active cycle、风险状态无法重建时一律 blocked，不再允许错误目标以 close-only 名义卖出。
- 后续目标股数按已对账的实时策略净值计算，不重复使用固定 100,000 美元。现金是 SOXL/SOXX 目标后的剩余权重，不下现金订单。
- 富途接口被限制为 SIMULATE；本任务没有真实账户分支。
- 模拟盘的 open/recovery/postclose 执行与同步检查继续运行但不再逐次推送；每日上海时间 10:05 仅发一次 Bark，合并最近任务、模拟委托与持仓同步摘要。Bark 凭据仅从运行环境读取，不写入仓库。
- Stage1 专用 dashboard 可通过 `production/run_stage1_dashboard.sh` 启动，支持自选区间、冻结模型推理状态、交易/换仓点、因子变化诊断及 SOXL/SOXX 同窗对比。
- 因子变化只用于交易点附近的标准化诊断，属于非因果归因，不是精确 SHAP。
- qlib recorder：7b5d8201e7824d688db4f9b988554cc5
- 不构成买卖建议。
