# 中文结论

- 做了什么：trainable regime policy nested WF 已移除人工模板和 prior-year regime selector；regime classifier、alpha model、risk model 都在 inner folds 内训练，最终只用 SOXL inner 校准、drawdown/activity gate 和跨 ETF stress audit 选固定参数网格候选。
- 核心指标：annual target pass years=[2024]，overfit audit=mitigated，common_data_end=2026-04-13。
- regime/window 差异：post_bear/range/source 这类人工形态模板不再参与 selector；rebound/trend/defensive/selective 仅作为监督学习标签，outer SOXL 年份只报告，不参与选择。
- 是否晋升：not_promoted_trainable_regime_policy_goal_or_overfit_audit_failed。
- 风险：固定参数网格仍可能有数据窥探风险，尤其如果看完 2023-2026 后继续改网格；多 3x ETF 仍共享权益 beta；5m/15m 风险状态只作为 risk cap，不证明独立盘中 alpha。
- 下一步：若 annual gate 未过，继续转向可训练 position model、SOXX 成分股特征和更长分钟历史，而不是按年份手工修模板。
- qlib recorder：3c65ccf84286479d91c3f0f04b48bc46
- 不构成买卖建议。
