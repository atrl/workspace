# Stage1 现金拖累诊断与回收叠加（中文结论）

- 诊断样本：真实生产目标历史 12 个确认 open 槽（2026-07-31 至 2026-10-07）。
- 平均 raw_beta 0.5377675253166092，平均 beta_cap 0.5408710782578476，上限 binding 比例 0.0。
- 平均权重：SOXL 0.165 / SOXX 0.329 / cash 0.506。
- 诊断结论：cap_binding_fraction 接近 0 说明现金来自策略自身的原始目标，而不是硬上限的截断；高现金是策略选择，不是 bug。
- 真实同期对比（25bp）：策略 +1.67% / Sharpe 0.81，SOXX buy&hold +0.49% / Sharpe 0.44；现金回收叠加与放宽约束均未优于真实生产目标。
- 叠加候选（把部分现金转入 SOXX，不改变 SOXL 权重）：recycle 比例 [0.0, 0.25, 0.5, 0.75, 1.0]。
- 最佳叠加 `recycle_0`；是否优于基线：False。
- 结构约束变体（不改信号，只重新投影）：最优 `action_beta_1.0`；重投影口径下是否改善 Sharpe：True。
- 关键澄清：真实生产目标本身好于任何重投影（真实模型输出含每日信息，重投影会丢失信息），所以本卡不能声称放宽约束改善了真实策略。
- 长历史（冻结上游原始目标，2016-2026）约束扫描：冻结 a_b=0.6 的 Sharpe 最高；放宽 action_beta 提高收益但降低 Sharpe 且回撤翻倍；max_soxl 上限从不 binding。
- 所有变体均跑输 SOXX buy&hold 收益，但冻结配置风险调整后最优。
- 决定 `frozen_constraints_are_best_risk_adjusted_choice`。
- 风险：样本短、非晋升证据；叠加只降低现金、增加 SOXX 暴露，不改变冻结策略参数；更宽价差会侵蚀收益。
- qlib Recorder：4d94018befc34b6396a1447887875561；provider_uri=/Users/rziiiii/dev/workspace/data/qlib/soxl。
