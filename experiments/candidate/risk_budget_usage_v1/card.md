# risk_budget_usage_v1

**方向**: model
**类别**: risk_budget
**状态**: 🟡 candidate

## 假设

慢速 Mode Router 决定风险区间（band），快速 Actor 决定区间内的使用比例（usage）。

## 架构

- Mode Router: 4 种 mode，每种有 floor/cap band
- Budget Usage: Actor 5-action delta 映射为 0%-100% band 使用率
- Position: `floor + usage × band_width`，再经过 drawdown cap

## 结果

| 指标 | 数值 |
|------|------|
| Return | 274.4% |
| Sharpe | 0.942 |
| MaxDD | 32.5% |
| 2022 MaxDD | 18.7% |

## 结论

**Candidate.** PPO 发现保守策略，未达 replay 规模。需 supervised warm-start。

## 证据

- 详细复盘: `runtime/model/reviews/risk_budget_usage_v1_full_review.md`
