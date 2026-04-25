# risk_budget_usage_v2

**方向**: model
**类别**: risk_budget
**状态**: 🔴 ablation

## 假设

Mode-conditional reward + dynamic band scaling 引导 PPO 发现更少保守的策略。

## 结果

| 切片 | Return | Sharpe | MaxDD |
|------|--------|--------|-------|
| test_proxy | 158.3% | 0.884 | 20.6% |
| test_2022 | -3.2% | -0.272 | 7.4% |

## 结论

**Ablation.** 过度保守。2022 avg_pos 仅 0.070，proxy Sharpe 0.884 未达 0.95。

## 证据

- Landscape: `docs/experiment-landscape-review.md`
