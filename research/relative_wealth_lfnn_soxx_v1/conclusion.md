# 中文结论

## 跑了什么

用 qlib DatasetH/DataHandlerLP 读取 SOXL/SOXX 因果状态与 next-open returns，在 2011–2019 上训练 paper-inspired global-in-time feasible network。模型直接输出 SOXL/SOXX/cash long-only 权重，以 100% SOXX 为基准，按 63 交易日调仓，并在 stationary block bootstrap 三年路径上优化相对终值目标。2020–2022 只用于 validation；2023–2026-07 只做已经看过的诊断 outer。

## Bootstrap validation

选中 `market_state_augmented`、年化目标 6.0%。10bp：终值财富比中位数 0.8616，跑赢概率 24.41%，Active IR -0.7200。25bp：中位数 0.8607，跑赢概率 24.37%，Active IR -0.7244。validation gate=False。

## Outer诊断

10bp 策略收益 523.14%，SOXX 355.91%，终值财富比 1.3668，Active IR 0.9247，MaxDD -48.36%。

## 决定

`research_reject_relative_wealth_validation_distribution_gates_failed`。无论 outer 是否好看，这个架构是在 outer 已被观察后提出，不能据此晋升；需要 2026-07-21 之后 sealed forward。当前 cash 收益按 0，且 SOXL/SOXX 是集中半导体映射，不等同论文宽基 LETF+债券。

不构成买卖建议。
