# 中文结论

- 做了什么：trainable regime policy nested WF 已移除人工模板、prior-year regime selector 和手写 policy grid；regime classifier、alpha model、risk model、learned position model 都在 inner folds 内训练，最终只用 SOXL inner 校准、drawdown/activity gate 和跨 ETF stress audit 选择模型族与特征集候选。
- 核心指标：annual target pass years=[]，overfit audit=mitigated，common_data_end=2026-04-13。
- 特征：SOXX 成分股 normalized lagged breadth/dispersion 特征可用 12 个，lag_days=1；它们是 nested feature-set 候选，不是强制 alpha 输入、模板或 gate。实际选中特征集=['base_soxx_diffusion']。
- regime/window 差异：post_bear/range/source 这类人工形态模板不再参与 selector；rebound/trend/defensive/selective 仅作为监督学习标签和诊断输出，仓位由 learned_position_target 回归头给出，20D alpha blend 只是 nested 候选架构之一，outer SOXL 年份只报告，不参与选择。
- 是否晋升：not_promoted_trainable_regime_policy_goal_or_overfit_audit_failed。
- 风险：learned_position_target 的标签设计本身仍可能过拟合训练年代，尤其如果看完 2023-2026 后继续改标签；SOXX 当前成分股回填历史有 survivorship bias；多 3x ETF 仍共享权益 beta；5m/15m replay 只是盘中减仓执行近似，本轮即使被选择也不证明独立盘中 alpha。
- 下一步：若 annual gate 未过，继续优化标签和可验证的新因子，不再按年份手工修模板或阈值 grid。
- qlib recorder：28888ec1315e4f40995758881ff013bb
- 不构成买卖建议。
