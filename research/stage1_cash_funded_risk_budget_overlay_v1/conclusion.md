# 中文结论

- 做了什么：冻结现有 Stage1，仅在 SOXL 已触及 30%、SOXX 20 日趋势为正且相对动量未处于 prior cap-event 最高三分之一时，测试从现金增加 5%/10% SOXL 或 SOXX；现金不得低于 15%，SOXL 临时上限 40%。
- 封顶后现金：开发期 69 次封顶，平均现金=19.76%；outer 176 次封顶，平均现金=21.08%。
- 开发期选中诊断候选：`soxl_010_pullback`，10bp return=224.4029%，相对 Stage1 增量=2.9904 个百分点，动作=12，MaxDD=-28.2589%。
- 收益门：要求至少 20.00 个百分点，实际=2.9904，gate=False。
- 动量诊断：开发期封顶样本的相对动量与下一日 SOXL-SOXX spread Spearman=-0.1803；低三分位平均 spread=0.6994%，高三分位=-0.3492%，确认高动量存在过热/反转风险。
- outer 复用诊断：相对 Stage1 增量=-22.6576 个百分点；该窗口不参与选择，也不是 clean holdout。
- 决策：`not_promoted_development_uplift_or_coverage_gates_failed`；promotion=blocked，生产 Stage1 不变。
- 失败门：['return_gate_pass']。开发期的小幅改善不足以补偿样本稀疏和outer 方向反转。
- 下一步：把本卡保留为现金出资失败边界；只有新增 PIT 广度信息或预登记 clean-forward 后才重开，不用 outer 调阈值。
- qlib recorder：`84eff9da51e943958ebe1193cc07b5dc`。
- 不构成买卖建议。
