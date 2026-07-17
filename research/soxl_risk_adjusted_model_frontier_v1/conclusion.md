# SOXL 风险调整模型前沿结论

## 做了什么

读取三张已冻结 route summary 和 qlib recorder 证据，未在 test 上重新选择或重训。

## 核心指标

三条路线均保留在 `model_selection.csv`；最终决定为 `not_promoted_no_risk_adjusted_candidate`。

## Regime / window 差异

Regime MoE、tail budget、20/60 日序列结果均按各自 card 原样进入比较；blocked 路线没有被省略。

## 晋升决定

`not_promoted_no_risk_adjusted_candidate`。

## 风险

canonical 数据截至 2026-07-02；三折仍是冻结策略诊断，nested walk-forward 未完成。本结果不构成买卖建议。

## 下一步

完成 expanding-window nested 重训，并补齐 PIT 成分股历史后再尝试 validated promotion。
