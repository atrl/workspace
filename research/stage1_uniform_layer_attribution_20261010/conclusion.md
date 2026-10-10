# 冻结配置、风险投影与动作层诊断

使用同一价格版本、五层独立账户、两窗口与10/25bp费用；零拟合、零新候选。

完整目标读取已哈希parent17冻结推理快照；原f696预测2686行逐行对照，末尾四日来自已保存的冻结模型推理延展。未重新推理、未静默HOLD。

Qlib原生账户对账：20/20；完整动作层与父e4逐日收益、三权重和动作一致。

| 窗口/费用 | 层 | 收益% | 最大回撤% | CVaR5% | 相对SOXX财富 |
|---|---|---:|---:|---:|---:|
| development_walk_forward / 10bp | raw_allocator_only | 404.26 | -51.71 | -5.33 | 1.2006 |
| development_walk_forward / 25bp | raw_allocator_only | 350.72 | -52.28 | -5.33 | 1.0747 |
| outer_reused_diagnostic / 10bp | raw_allocator_only | 305.64 | -45.61 | -6.17 | 0.8144 |
| outer_reused_diagnostic / 25bp | raw_allocator_only | 285.84 | -46.35 | -6.17 | 0.7758 |
| development_walk_forward / 10bp | open_risk_projection_nohold_no_recovery | 171.67 | -29.93 | -3.51 | 0.6468 |
| development_walk_forward / 25bp | open_risk_projection_nohold_no_recovery | 99.14 | -30.21 | -3.58 | 0.4748 |
| outer_reused_diagnostic / 10bp | open_risk_projection_nohold_no_recovery | 220.43 | -26.92 | -4.81 | 0.6433 |
| outer_reused_diagnostic / 25bp | open_risk_projection_nohold_no_recovery | 66.12 | -36.44 | -4.40 | 0.3340 |
| development_walk_forward / 10bp | open_risk_projection_withhold_no_recovery | 115.46 | -30.08 | -3.46 | 0.5130 |
| development_walk_forward / 25bp | open_risk_projection_withhold_no_recovery | 96.75 | -30.18 | -3.58 | 0.4691 |
| outer_reused_diagnostic / 10bp | open_risk_projection_withhold_no_recovery | 219.49 | -26.92 | -4.80 | 0.6414 |
| outer_reused_diagnostic / 25bp | open_risk_projection_withhold_no_recovery | 75.14 | -32.38 | -4.42 | 0.3522 |
| development_walk_forward / 10bp | full_withhold_and_recovery | 114.59 | -30.08 | -3.49 | 0.5109 |
| development_walk_forward / 25bp | full_withhold_and_recovery | 96.72 | -30.09 | -3.59 | 0.4691 |
| outer_reused_diagnostic / 10bp | full_withhold_and_recovery | 219.49 | -26.92 | -4.80 | 0.6414 |
| outer_reused_diagnostic / 25bp | full_withhold_and_recovery | 75.14 | -32.38 | -4.42 | 0.3522 |
| development_walk_forward / 10bp | nohold_with_recovery_reference | 118.19 | -29.93 | -3.47 | 0.5195 |
| development_walk_forward / 25bp | nohold_with_recovery_reference | 102.23 | -30.13 | -3.58 | 0.4822 |
| outer_reused_diagnostic / 10bp | nohold_with_recovery_reference | 220.43 | -26.92 | -4.81 | 0.6433 |
| outer_reused_diagnostic / 25bp | nohold_with_recovery_reference | 66.12 | -36.44 | -4.40 | 0.3340 |

HOLD与恢复动作的交互（收益百分点，只是有序路径对照）：

- development_walk_forward / 10bp：无HOLD时加入恢复 -53.48pp；有HOLD时加入恢复 -0.87pp。
- development_walk_forward / 25bp：无HOLD时加入恢复 +3.09pp；有HOLD时加入恢复 -0.03pp。
- outer_reused_diagnostic / 10bp：无HOLD时加入恢复 +0.00pp；有HOLD时加入恢复 +0.00pp。
- outer_reused_diagnostic / 25bp：无HOLD时加入恢复 +0.00pp；有HOLD时加入恢复 +0.00pp。

风险循环的实际账户统计：

- development_walk_forward / 10bp：stop事件 41 次；emergency最长连续 96 日。详情、状态占比和分组配对log财富差见risk_mechanism.csv及Recorder。
- development_walk_forward / 25bp：stop事件 44 次；emergency最长连续 134 日。详情、状态占比和分组配对log财富差见risk_mechanism.csv及Recorder。
- outer_reused_diagnostic / 10bp：stop事件 20 次；emergency最长连续 42 日。详情、状态占比和分组配对log财富差见risk_mechanism.csv及Recorder。
- outer_reused_diagnostic / 25bp：stop事件 30 次；emergency最长连续 358 日。详情、状态占比和分组配对log财富差见risk_mechanism.csv及Recorder。

年度、beta及Qlib product模式风险统计见annual_metrics.csv、risk_report.csv。费用归因从10bp实际目标路径重新结算0/25bp，10bp自复现检查真执行；不是从净收益再次减费。

风险投影减少回撤也牺牲上涨暴露；该比较不能证明约束过度，不能靠删除风控达成上线。raw使用post_fee_desired，旧动作层使用pre_fee_account，差异及账户反馈均包含在路径比较中。这些层不是唯一可加的因果贡献，条件分组不是可交易子策略。

外窗已被反复查看，只是复用历史诊断，不是新鲜样本外。收益、尾部风险、年度稳定性和独立前向证据仍需分别达标。

晋升决定：不晋升；本卡只定位机制。下一步依据冻结状态统计制定一个明确机制增量，再单独冻结验证；不得继续搜索失败的信息路线、改符号或放宽门槛。
