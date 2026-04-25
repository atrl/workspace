# risk_budget_usage_v3

**方向**: model
**类别**: risk_budget
**状态**: 🟡 candidate (blocked)

## 假设

Actor 直接接收 mode one-hot（4-dim），学习 mode-specific usage 策略。

## 方法/配置

Profile: `risk_budget_usage_v3`
网络: GRU-64 + [128,64] MLP, obs + 4-dim mode one-hot

## 结果

| 切片 | Return | Sharpe | MaxDD | Trades |
|------|--------|--------|-------|--------|
| test (full) | 434.1% | 0.87 | 47.5% | 295 |
| wf_2020 | 19.3% | 0.29 | 19.5% | 36 |
| wf_2022 | 59.9% | 0.39 | **59.0%** | 32 |
| wf_2023_2024 | 13.0% | 0.22 | 23.4% | 40 |

**❌ PROMOTION BLOCKED.** Walk-forward 未通过。

## 结论

Mode-conditioned input 改善 proxy 指标但不解决泛化。需要诊断：
1. 在 proven baseline 上验证该机制
2. 收紧 budget_usage 映射范围
3. 修复 proxy-to-backtest gap

## 证据

- 详细复盘: `runtime/model/reviews/risk_budget_usage_v3_wf_review.md`
- 连续性记录: `memory/2026-04-24-risk-budget-v3-wf-failure.md`
