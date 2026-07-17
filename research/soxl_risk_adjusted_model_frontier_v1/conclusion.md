# SOXL 风险调整模型前沿结论

## 做了什么

通过 qlib Recorder 对象读取三条冻结路线的预测、标签、组合报告、仓位、风险分析和选择证据；前沿 recorder 为 `ae2deb9590dd4fd3b3bfcf4cc3c93aa3`，没有在 test 上重新选择或重训。

## 核心指标

三条路线均保留在 `model_selection.csv`；最终决定为 `not_promoted_no_risk_adjusted_candidate`。

## Regime / window 差异

Regime MoE、tail budget、20/60 日序列结果按同一窗口和执行成本比较；blocked 路线不会被省略。

## 晋升决定

`not_promoted_no_risk_adjusted_candidate`。

## 风险

canonical 数据截至 2026-07-02；三折仍是冻结策略诊断，nested walk-forward 未完成。本结果不构成买卖建议。

## 下一步

完成 expanding-window nested 重训，并补齐 PIT 成分股历史后再尝试 validated promotion。
