# 结论

- 做了什么：继续 R-G-E-V-D，把 TSMOM/stop、canary sprint、signed overnight、macro credit rate gate 四类外部知识分支组合成固定 ensemble，并通过 qlib.init + D.features 读取 SOXL/SOXX/VIX 与 macro-credit qlib store。
- 核心指标：selected=agentic_multi_branch_ensemble_v1，return_pct=8775.1236%，max_drawdown_pct=-19.7008%，baseline_return_pct=6570.0233%。
- regime/window differences：低波 SOXX momentum 提供主收益，canary/autocorr 与 macro risk gate 降低连续暴露，signed overnight 提供低相关 session leg；macro leg coverage 到 2026-06-08，其余日线分支覆盖到 2026-06-09。
- promotion_decision：promote_to_production_review，当前研究卡已同时满足收益与回撤门禁，但仍需生产 paper/sim 固定参数复核后才能成为执行规则。
- risks：OHLC stop 在真实成交中会受跳空和流动性影响；权重来自已完成分支搜索后的组合选择，后续需要 walk-forward/纸交易再验证。
- next_action：把 selected_model.json 固定参数迁入 production paper/sim 候选；recorder_id=c00e240b920a4a54a9aaf4050c3570a7。
