# 中文结论

- 做了什么：新增 trainable regime policy nested WF，不再用人工模板；regime classifier、alpha model、risk model 都在 inner folds 内训练/选择。
- 核心指标：annual target pass years=[]，overfit audit=mitigated，common_data_end=2026-04-13。
- regime/window 差异：post_bear/rebound/range/source 这类人工模板被替换成 supervised regime label；outer SOXL 年份只报告，不参与选择。
- 是否晋升：not_promoted_trainable_regime_policy_goal_or_overfit_audit_failed。
- 风险：虽然人工模板已移除，但多 3x ETF 仍共享权益 beta；5m/15m 风险状态只作为 risk cap，不证明独立盘中 alpha。
- 下一步：若 annual gate 未过，继续扩充候选 policy 和独立资产/分钟历史，而不是按年份手工修模板。
- qlib recorder：e7813db27089425d866fbf7d45786f56
- 不构成买卖建议。
