# Stage1 风险门 legacy 诊断（无 recheck 结论，中文结论）
- 生产参考：窗口 2023-01-04..2026-08-07（900 行），成本 10.0bp，source recorder `f6b09330175e41719886d58ee2d11bec`，`worst_rolling_5d_pct=-12.110754878147423`。
- 同窗同身份可比数：0；recheck 结论：`none`。
- 各 legacy recorder：
  - 454e94c9ed5f（2023-01-03..2026-06-26，window_match=False，identity_match=False，cost=unknown，calendar=unknown，可比=not_comparable，qlib=readable）：收益 629.7%，MaxDD -28.11%，worst5d -11.30%
  - 76a6fe6a54cf（2023-01-03..2026-06-26，window_match=False，identity_match=False，cost=unknown，calendar=unknown，可比=not_comparable，qlib=readable）：收益 639.0%，MaxDD -28.11%，worst5d -11.30%
- 撤回声明：This card makes no claim that the production risk failure is a provenance artifact. Legacy committed recorders are reported as diagnostics only; their Qlib readability and window/identity comparability are recorded, and only an exact same-window, same-identity, same-cost run could confirm or overturn the production result.
