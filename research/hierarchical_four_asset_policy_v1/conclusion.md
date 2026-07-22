# 中文结论

## 跑了什么

用独立 qlib SOXL/SOXX/SOXS provider 执行分层策略：现金控制总风险，SOXX 为核心，SOXL 为 20%/40% 条件进攻仓位，SOXS 仅测试 5%/10% 与 1/3 日尾部覆盖。SOXL 与 SOXS 互斥，SOXS 退出后冷却两日。候选阈值只在 2017–2019 tune 选择；2020–2022 validation 选择变体；2023–2026 outer 只诊断。

## Validation

选中 `soxs_10pct_3d`。10bp：收益 7.74%，SOXX 40.10%，财富比 0.7690，Active IR -0.5625，MaxDD -47.89%，最差五日 -25.13%，SOXS 平均仓位 0.12%，现金平均仓位 5.78%。validation gate=False。

## Outer 诊断

10bp：收益 350.13%，SOXX 379.94%，财富比 0.9379，MaxDD -49.93%；四个年份中 2 个通过年度尾部风险门。outer 不参与选择。

## 决定、风险和下一步

`research_reject_hierarchical_four_asset_failed_validation_gates`。若 validation 失败，说明把四种资产做成分层状态机仍未修复进攻标签漂移；SOXS 只能保留为特征或极小尾部压力测试，不能晋升。下一步应更换 benchmark-relative 进攻目标，而不是用 outer 返调当前阈值。

不构成买卖建议。
