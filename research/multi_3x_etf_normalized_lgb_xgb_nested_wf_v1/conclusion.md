# 中文结论

- 做了什么：不再选择人工模板，构建 SOXL/TQQQ/UPRO/SPXL/TECL 多 3x ETF 归一化 pooled LGB/XGB nested walk-forward 研究卡。
- 核心指标：annual target pass years=[]，overfit audit=mitigated，common_data_end=2026-04-13。
- regime/window 差异：regime 不再手写选择模板，候选参数只由 inner leave-one-asset-out folds 选择，outer SOXL 年份只报告。
- 是否晋升：not_promoted_multi_3x_nested_wf_goal_or_overfit_audit_failed。
- 风险：多 ETF 数据只能把可比样本扩到 2026-04-13；仍需要补齐更多 3x ETF、分钟级成交成本、以及完全独立未来窗口。
- 下一步：若 annual gate 未过，优先扩大预注册 ETF universe 与固定候选空间，而不是人工添加年份模板。
- qlib recorder：3f73ff5d9e2447fabec5c9bbe3968eb3
- 不构成买卖建议。
