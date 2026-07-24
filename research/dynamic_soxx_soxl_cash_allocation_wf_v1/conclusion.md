# 中文结论

- 做了什么：新建 SOXX/SOXL/cash 动态配置 walk-forward card，benchmark 从 SOXL B&H 改为 SOXX，目标为累计收益超额 SOXX 20 个百分点且最大回撤不低于 -10%。
- 核心指标：annual pass years=[]，generalization gate=False，common_data_end=2026-07-23。
- 模型原理：LGB/XGB/consensus 分别训练 SOXL 相对 SOXX 5D 超额概率、SOXX 5D 正收益概率、5D tail risk 概率；仓位只输出 SOXX、SOXL、cash 三个权重，且权重和为 1。
- 泛化验证：selector 使用 SOXL calibration + ['TQQQ', 'UPRO', 'SPXL', 'TECL'] proxy generalization gate；outer proxy stress 不参与选择，只报告结构泛化是否仍为正。
- 特征：基础 3x/underlying rolling zscore/percentile、SOXX 成分股 diffusion 特征 12 个，以及 5m/15m intraday risk state；分钟数据只做风险状态，不声明日内 alpha。
- 是否晋升：not_promoted_dynamic_allocation_goal_or_generalization_gate_failed。
- 风险：SOXX 当前成分股历史仍有 survivorship bias；proxy 3x ETF 与半导体并不等价；如果 annual gate 未过，不能靠事后改阈值宣布实盘能力。
- qlib recorder：b9bc3e02854c4e808d7ef2fcae357308
- 不构成买卖建议。
