# Stage1 现金拖累诊断与回收叠加（中文结论）

- 诊断样本：真实生产目标历史 12 个确认 open 槽（2026-07-31 至 2026-10-07）。
- 平均 raw_beta 0.5377675253166092，平均 beta_cap 0.5408710782578476，上限 binding 比例 0.0。
- 平均权重：SOXL 0.165 / SOXX 0.329 / cash 0.506。
- 结论：If cap_binding_fraction is near zero, the cash is the policy's own raw target rather than a hard-cap artifact; the fix is an overlay, not a cap change.
- 叠加候选（把部分现金转入 SOXX，不改变 SOXL 权重）：recycle 比例 [0.0, 0.25, 0.5, 0.75, 1.0]。
- 最佳叠加 `recycle_0`；是否优于基线：False；决定 `cash_overlay_no_improvement`。
- 风险：样本短、非晋升证据；叠加只降低现金、增加 SOXX 暴露，不改变冻结策略参数；更宽价差会侵蚀收益。
- qlib Recorder：96a92d05788d49148a26e4dd943e9d16；provider_uri=/Users/rziiiii/dev/workspace/data/qlib/soxl。
