# 中文结论

- 已把晋级的 Stage1 固定为生产模拟源；Stage2 仍关闭，失败的 stateful direct 模型没有进入任务。
- 同口径区间：2023-01-04 至 2026-07-24；Stage1 10bp return=541.7889%，MaxDD=-28.1051%。
- legacy soxlbacktest 成本归一到 10bp 后 return=341.2037%，MaxDD=-55.1610%。
- 当前最新信号日=2026-07-24；状态=pending_next_open_mark。周末无法提前知道下个开盘价，因此最终开盘仓位会在 09:31 ET 用模拟盘报价结算上一段路径后生成。
- 模拟账户只使用 10,000 美元策略名义本金；现金是 SOXL/SOXX 目标后的剩余权重，不下现金订单。
- 富途接口被限制为 SIMULATE；本任务没有真实账户分支。
- Stage1 专用 dashboard 可通过 `production/run_stage1_dashboard.sh` 启动，支持自选区间、冻结模型推理状态、交易/换仓点、因子变化诊断及 SOXL/SOXX 同窗对比。
- 因子变化只用于交易点附近的标准化诊断，属于非因果归因，不是精确 SHAP。
- qlib recorder：04236fdaa94748b6a580b15759bb7382
- 不构成买卖建议。
