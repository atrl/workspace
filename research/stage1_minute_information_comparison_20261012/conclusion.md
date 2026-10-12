# 完整 Stage1 对照结论

本次修正比较层级：先前分钟卡评估分配器单层，不能替代旧模拟盘完整策略。复用冻结分配器预测及现役动作/恢复模型，没有训练、调参或切换模拟盘。
固定历史诊断：2023-01-04 至 2026-10-09，944 个完整账户区间。每档费用独立持仓、净值峰值、回撤、风险时钟与 HOLD/恢复动作；Qlib 原生订单、现金、数量逐日核对。

| 费用 | 分配器接完整 Stage1 | 净收益 | 最大回撤 | 相对 SOXX 终值 |
|---|---|---:|---:|---:|
| 10bp | frozen_current | 219.63% | -26.92% | 0.6399 |
| 10bp | base | 295.42% | -24.43% | 0.7917 |
| 10bp | availability_control | 209.08% | -28.50% | 0.6188 |
| 10bp | core_path | 91.39% | -30.05% | 0.3832 |
| 10bp | soxx | 399.46% | -42.79% | 1.0000 |
| 25bp | frozen_current | 75.08% | -32.38% | 0.3511 |
| 25bp | base | 56.32% | -29.15% | 0.3135 |
| 25bp | availability_control | 156.03% | -29.34% | 0.5134 |
| 25bp | core_path | 98.05% | -30.41% | 0.3971 |
| 25bp | soxx | 398.71% | -42.79% | 1.0000 |

旧卡开发期 10bp 最大回撤 −30.08%、25bp −31.88%；旧外层约 −26.92%/−27.32%。这些是已观察的历史结果，不是硬性保证。旧窗口、旧价格来源的数值保留在 legacy_reference.csv，不能直接与扩展窗口相减。

旧卡 25bp 是固定 10bp 开盘/HOLD/恢复事件后重新扣费的压力测试（cost_stress_contract=fixed_target_hold_recovery_events_repriced_at_each_stage_net_nav），没有按 25bp 净值重新决定风控路径。本卡两档费用各自重算状态。两种费用口径必须区分。

年度差异见 regime_metrics.csv。旧单模型与三 seed 重训基线的差异不能全部归因于分钟数据；core_path 必须同时与 base 和 availability_control 对照。
分钟单层的历史改进需要经过完整控制器重新验证。两档费用下候选与对照的排序不同，不能据此重新选模。控制器成本敏感性见 cost_path_differences.csv，净值回撤与风险状态核对见 controller_state_diagnostics.csv。

历史执行价格统一为 NAS 真实分钟及已公布复权事件构建的开盘/首小时结束标记。开盘及 10:30 close 是历史记账标记，不声明真实 09:31/10:31 成交、首次数据接收时点或现金分红资格。模型分钟输入仍为此前冻结的 raw NONE，未用本次复权系数重算特征。
诊断门：{"net_return_above_soxx_both_costs": false, "observed_drawdown_within_30pct_both_costs": false, "return_above_refit_and_availability_controls_both_costs": false, "original_minute_development_gate_pass": false, "independent_forward_gate_pass": false}。
晋升：not_promoted_full_stage1_comparison_diagnostic；原分钟开发门已失败，独立合格前向为 0，objective_achieved=false。比较修正不能重新挑选候选或把复用历史变成新的样本外证据。
下一步：结束此固定对照；保留失败路线。模型优化需要新的有效信息或预先冻结的独立前向对照，不能靠增加同一历史上的参数搜索。
Recorder：9d671fcfb1aa467bb8a9fd83971fb015。行情、大账户及模型引用留 NAS；Git 保留 notebook、瘦卡和代码。
