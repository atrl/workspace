# 中文结论

- 已把晋级的 Stage1 固定为生产模拟源；Stage2 仍关闭，失败的 stateful direct 模型没有进入任务。
- 同口径区间：2023-01-04 至 2026-07-30；Stage1 10bp return=467.5186%，MaxDD=-28.1051%。
- 当前生产决策=paper_sim_blocked_failed_position_risk_gates；risk_gate_pass=false。风险门槛或数据新鲜度失败时进入 close-only：禁止增加风险，但保留减仓。
- legacy soxlbacktest 成本归一到 10bp 后 return=346.4954%，MaxDD=-55.1610%。
- 当前最新信号日=2026-07-30；目标计算状态=pending_next_open_mark。只有生产风险门槛和数据新鲜度都通过才允许增加风险；否则仅允许 close-only 减仓。
- 模拟账户初始资金为 100,000 美元；后续目标股数按已对账的实时策略净值计算，不重复使用固定初始本金。现金是 SOXL/SOXX 目标后的剩余权重，不下现金订单。
- 富途接口被限制为 SIMULATE；本任务没有真实账户分支。
- 模拟盘的 open/recovery/postclose 结果与异常会即时推送 Bark；每日上海时间 10:05 另发一次心跳，即使无交易也报告最近任务状态。Bark 凭据仅从运行环境读取，不写入仓库。
- Stage1 专用 dashboard 可通过 `production/run_stage1_dashboard.sh` 启动，支持自选区间、冻结模型推理状态、交易/换仓点、因子变化诊断及 SOXL/SOXX 同窗对比。
- 因子变化只用于交易点附近的标准化诊断，属于非因果归因，不是精确 SHAP。
- qlib recorder：680aa02e4de143bab3f6e6cb1ce6c1b2
- 不构成买卖建议。
