# 中文结论

## 跑了什么

从独立 qlib SOXL/SOXX/SOXS provider 读取复权后的 open/close，比较基础特征、仅增加 SOXS 特征、仅允许受约束 SOXS 防守、SOXS 特征加防守四组消融。候选参数只在 2017–2019 tune 选择，2011–2019-11 refit，2020–2022 validation，2023–2026-07 outer 只诊断。SOXL 与 SOXS 互斥，SOXS 上限 20%，连续最多五日，之后转现金直到风险信号复位。

## Validation

选中 `soxs_defense_only`。10bp：相对 SOXX 财富比 0.9417，Active IR -0.1783，收益 31.92%，MaxDD -23.11%，SOXS 平均仓位 2.31%。validation gate=False。

## Outer诊断

10bp：收益 85.17%，SOXX 379.94%，财富比 0.3858，MaxDD -30.37%。outer 不参与选择。

## 决定

`research_reject_soxs_ablation_failed_validation_gates`。SOXS 是否提供增量必须以 feature-only 与 tradable-sleeve 的差异判断；已观察 outer 不能用于返调阈值。下一步只能进入 sealed-forward paper shadow，或在 validation 失败时停止本架构。

不构成买卖建议。
