# 中文结论

- 已把晋级的 Stage1 固定为生产模拟源；Stage2 仍关闭，失败的 stateful direct 模型没有进入任务。
- 同口径区间：2023-01-04 至 2026-07-28；Stage1 10bp return=468.3528%，MaxDD=-28.1051%。
- 当前生产决策=paper_sim_blocked_failed_position_risk_gates；risk_gate_pass=false。风险门槛失败时，09:31/10:31 ET 调度会在连接券商前 fail-closed，不提交模拟订单。
- legacy soxlbacktest 成本归一到 10bp 后 return=334.6595%，MaxDD=-55.1610%。
- 当前最新信号日=2026-07-27；目标计算状态=ready。该目标仅作诊断，只有生产风险门槛通过后才允许在 09:31 ET 进入模拟执行。
- 模拟账户只使用 10,000 美元策略名义本金；现金是 SOXL/SOXX 目标后的剩余权重，不下现金订单。
- 富途接口被限制为 SIMULATE；本任务没有真实账户分支。
- Stage1 专用 dashboard 可通过 `production/run_stage1_dashboard.sh` 启动，支持自选区间、冻结模型推理状态、交易/换仓点、因子变化诊断及 SOXL/SOXX 同窗对比。
- 因子变化只用于交易点附近的标准化诊断，属于非因果归因，不是精确 SHAP。
- qlib recorder：723edc94be6b4ed49eb5eb28d69b8b85
- 不构成买卖建议。
