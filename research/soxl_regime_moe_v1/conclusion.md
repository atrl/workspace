# soxl_regime_moe_v1 结论

## 做了什么

使用 canonical qlib store、DatasetH/DataHandlerLP 和 qlib Recorder 执行 `regime_moe` 路线；模型、预测、标签、仓位和组合证据均登记在 recorder `07e951ab7960462ea70f23897f403273`。

## 核心指标

- 候选测试收益：0.9449；最大回撤：-0.3372；平均绝对仓位：0.2134。
- 匹配 legacy 收益：2.3828；最大回撤：-0.5018。
- as-of 为 2026-07-02，没有外推到更新日期。

## Regime / window 差异

路线状态为 `ok`；选择候选为 `validation_selected_regime_moe`。Regime 或窗口明细见 sidecars；序列横截面模式状态为 `not_applicable`。

## 晋升决定

`not_promoted_failed_risk_adjusted_gates`。失败门包括：test_max_drawdown, test_positive_return_capture, test_return_to_drawdown, test_total_turnover, walk_forward_max_drawdown, walk_forward_positive_return_capture, walk_forward_return_to_drawdown, walk_forward_total_turnover, nested_walk_forward。

## 风险

三折报告为冻结策略的时间切片诊断，尚未完成每折重新拟合的 nested walk-forward，所以 `nested_walk_forward` 明确失败；canonical 数据截至 2026-07-02。本结果不构成买卖建议。

## 下一步

先补齐 nested expanding-window 重训；序列横截面路线还需经过验证的 PIT 成分股历史，再决定是否进入 validated card。
