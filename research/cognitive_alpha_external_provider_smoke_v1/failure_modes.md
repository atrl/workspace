# Failure Modes

- 这是 Codex provider readiness smoke；Codex CLI 使用 ChatGPT 登录，不要求外部 API key。
- 本卡用 fixture provider 快速证明 Codex exec schema、quality checker、repair loop 和 recorder wiring，未消耗完整 Codex 生成预算。
- fixture provider 故意生成一个 future shift 候选，只用于证明 quality checker 和 repair loop 会进入 recorder。
- SOXL 是单标的时间序列，弱于 CogAlpha 论文股票池横截面设定。
- 本卡不生成生产买卖点，不构成买卖建议。
