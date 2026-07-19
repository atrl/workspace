# 结论

- 执行内容：16只长期股票类3x ETF的归一化路径状态模型，按2023至2026年度样本外拼接，并用SOXL/SOXX/现金控制器执行。
- 关键指标：年化收益 12.07%，年化波动 11.59%，最大回撤 -9.59%，5% CVaR -1.76%。
- 区间差异：高位震荡与下行确认采用独立状态；跨资产验证通过率为 0.0%。
- 晋升决定：not_promoted_risk_control_gates_failed；失败门槛为 upside_capture, average_risky_weight, return_to_drawdown_benchmarks, transfer_fold_pass_rate。
- 风险：模型概率仍可能在SOXL独有半导体周期中失效，且固定成本未覆盖开盘跳空冲击。
- 下一步：仅保留样本外风险改善稳定的状态，继续缩减无迁移增益特征并做paper验证，不进入真实交易。
