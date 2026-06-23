# 结论
- what_ran: 执行 agentic_alpha_macro_credit_search_v1，将 GitHub/论文/arXiv/Twitter 的 TAA、杠杆 ETF 和信用风险知识映射到收益率曲线、credit spread、BAA yield、DXY、oil 与 SOXX trend，并用 `build_qlib_store_from_frame` 构建 qlib provider 后通过 `D.features` 审计。
- key_metrics: best macro_credit_rate_gate return=83085.3169%, MaxDD=-80.3447%；best drawdown-gated return=762.1911%, MaxDD=-19.0348%。
- regime_window_differences: 信用/利率 gate 可筛掉部分风险期，但高收益候选仍继承 SOXL 崩盘回撤；vol-target throttle 能把 MaxDD 压到 20% 内，但收益远低于生产 baseline。
- promotion_decision: not_promoted；未找到同时满足收益超过现有生产策略且 MaxDD 小于 20% 的候选。
- risks: 宏观变量多为月度或外部日频序列，虽然用 t 日信号交易 t+1 收益，仍需 validated card 复核实时可得性和发布延迟。
- next_action: 继续 R-G-E-V-D，优先寻找能够在高收益候选上降低 crash gap 的非线性/组合保险分支。
- qlib_recorder_id: 6768729892374924a5da1ab1d7a3c0da
