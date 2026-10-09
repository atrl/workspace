# 中文结论

- 已把晋级的 Stage1 固定为生产模拟源；Stage2 仍关闭，失败的 stateful direct 模型没有进入任务。
- 同口径区间：2023-01-04 至 2026-08-07；Stage1 10bp return=501.7297%，MaxDD=-28.1127%。
- 当前生产晋升决策=paper_sim_blocked_failed_position_risk_gates；risk_gate_pass=false。晋升门仍未通过；本次仅按用户明确要求运行独立的 Futu SIMULATE 观察周期，不代表生产晋升。
- legacy soxlbacktest 成本归一到 10bp 后 return=364.5196%，MaxDD=-55.1610%。
- card 内当前最新信号日=2026-08-07；目标计算状态=pending_next_open_mark；controller_state_source=historical_model_replay。该完整历史目标只用于 qlib 回放一致性诊断，不可直接执行。
- 实际模拟目标必须读取 active paper cycle：旧周期先平仓归档，新周期按平仓后实际现金净值重新基准化；控制器只使用本周期确认成交与 qlib 结算标记。
- 数据过期、控制器来源不属于 active cycle、风险状态无法重建时一律 blocked，不再允许错误目标以 close-only 名义卖出。
- 后续目标股数按已对账的实时策略净值计算，不重复使用固定 100,000 美元。现金是 SOXL/SOXX 目标后的剩余权重，不下现金订单。
- 10:31 仅在 recovery_action=true 时换仓；若恢复信号未触发则明确 no_action，不因价格、净值或整股取整变化重复对齐同一目标。
- 支持通过 `production/run_stage1_futu_sim.sh --slot start-now` 在任意时刻请求启动：美股常规交易时段内按实时价格和 active paper ledger 幂等接管；10:30 ET 后纳入已观察的首小时恢复状态；闭市时仅 armed，不提交订单。
- 回测换手按每次决策前经价格变动漂移后的实际持仓计算，不再用上一次目标权重代替实际权重；因此交易成本与模拟执行口径一致。
- 富途接口被限制为 SIMULATE；本任务没有真实账户分支。
- 模拟盘的 open/recovery/postclose 执行与同步检查继续运行但不再逐次推送；每日上海时间 10:05 仅发一次 Bark，合并最近任务、模拟委托与持仓同步摘要。Bark 凭据仅从运行环境读取，不写入仓库。
- Stage1 专用 dashboard 可通过 `production/run_stage1_dashboard.sh` 启动；只选择一个模拟开始日，账户从该交易日的 100% 现金状态独立回放到最新可用日。
- 初始建仓必须记录为首笔交易并计入换手与成本，后续交易、因子变化诊断及 SOXL/SOXX 同窗对比均来自该独立模拟，不继承开始日前仓位。
- dashboard 不突出展示 active paper cycle、账本或执行同步；本地源仅监听 127.0.0.1。`https://soxl.atrl.me/` 是预留入口，在 Zero Trust Free 方案与 Access 应用启用、验证前保持 tunnel 停止。
- 因子变化只用于交易点附近的标准化诊断，属于非因果归因，不是精确 SHAP。
- qlib recorder：e9c97a327bdf46f5a6a57f2d58d75062
- 不构成买卖建议。

## 服务中断披露（2026-10-09 补记）

- 该模拟周期在 **2026-08-13 至 2026-09-25** 之间**没有运行**：无成交、无调仓、无风控。
- 证据：`execution_state_v2.json` 在 2026-08-06 与 2026-09-25 之间无 completed_slots；`launchd.stdout.log` 在 2026-08-07..2026-09-22 零条目；`paper_ledger_v2.json` 无 2026-08-13..2026-09-27 成交。
- 根因：`com.soxl.stage1.futu.sim` 调度停摆，且**心跳 agent `com.soxl.stage1.futu.sim.heartbeat` 从未安装**，因此中断是静默的。
- 影响：中断期内账户只是被动持有 2026-08-12 的目标仓位，其净值变动**不是策略证据**。
- 修正后前向（仅活跃/受管理交易日，17 个）：策略 +13.42%，SOXX +11.02%，SOXL +22.19%；超额 vs SOXX +2.40pp。样本极小，**不构成晋升证据**。
- 整改：心跳 agent 已于 2026-10-09 安装并加载；`detect_execution_outage` 在超过 3 个交易日无确认成交时告警。详见 `runtime/soxl/stage1_futu_sim/execution_outages.json` 与 `forward_evidence_corrected.json`。
